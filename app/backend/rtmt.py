import asyncio
import json
import logging
import os
from enum import Enum
# Added defaultdict for easier argument aggregation
from collections import defaultdict
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import web
# from ragtools import Tool, ToolResult, ToolResultDirection # Ensure ragtools defines these

logger = logging.getLogger("voicerag")

class ToolResultDirection(Enum):
    TO_SERVER = 1
    TO_CLIENT = 2

class ToolResult:
    text: str
    destination: ToolResultDirection

    def __init__(self, text: str, destination: ToolResultDirection):
        self.text = text
        self.destination = destination

    def to_text(self) -> str:
        if self.text is None:
            return ""
        return self.text if type(self.text) == str else json.dumps(self.text)

class Tool:
    target: Callable[..., ToolResult]
    schema: Any

    def __init__(self, target: Any, schema: Any):
        self.target = target
        self.schema = schema

class RTToolCall:
    tool_call_id: str
    previous_id: str

    def __init__(self, tool_call_id: str, previous_id: str):
        self.tool_call_id = tool_call_id
        self.previous_id = previous_id

class PendingToolCall:
    call_id: str
    tool_name: str
    arguments_str: str # Accumulated arguments as string
    previous_item_id: Optional[str] # For client-side grounding correlation

    def __init__(self, call_id: str, tool_name: str, previous_item_id: Optional[str] = None):
        self.call_id = call_id
        self.tool_name = tool_name
        self.arguments_str = ""
        self.previous_item_id = previous_item_id

class RTMiddleTier:
    # 注意：这里需要修改为openai的api地址，并将之前的配置注释掉
    openai_api_key: str
    model: str
    websocket_base_url: str = "wss://api.openai.com/v1/realtime"
    # endpoint: str
    # deployment: str
    # key: Optional[str] = None
    
    # Tools are server-side only for now, though the case could be made for client-side tools
    # in addition to server-side tools that are invisible to the client
    tools: dict[str, Tool] = {}

    # Server-enforced configuration, if set, these will override the client's configuration
    # Typically at least the model name and system message will be set by the server
    # model: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    disable_audio: Optional[bool] = None
    voice_choice: Optional[str] = None
    # api_version: str = "2024-10-01-preview"
    # _tools_pending = {}
    # Use defaultdict to store accumulating arguments for pending tool calls
    _pending_tool_calls: defaultdict[str, PendingToolCall] = defaultdict(lambda: None)
    # Keep track of item_id -> call_id mapping from function_call items
    _item_to_call_id: dict[str, str] = {}
    # _token_provider = None

    # 修改init函数
    # def __init__(self, endpoint: str, deployment: str, credentials: AzureKeyCredential | DefaultAzureCredential, voice_choice: Optional[str] = None):
    #     self.endpoint = endpoint
    #     self.deployment = deployment
    #     self.voice_choice = voice_choice
    #     if voice_choice is not None:
    #         logger.info("Realtime voice choice set to %s", voice_choice)
    #     if isinstance(credentials, AzureKeyCredential):
    #         self.key = credentials.key
    #     else:
    #         self._token_provider = get_bearer_token_provider(credentials, "https://cognitiveservices.azure.com/.default")
    #         self._token_provider() # Warm up during startup so we have a token cached when the first request arrives
    def __init__(self, openai_api_key: str, model: str, system_message: Optional[str] = None, voice_choice: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.model = model # Store the model ID
        self.system_message = system_message # Store system message if passed
        self.voice_choice = voice_choice
        if voice_choice is not None:
            logger.info("Realtime voice choice set to %s", voice_choice)
        self._pending_tool_calls = defaultdict(lambda: None) # Ensure it's reset on init
        self._item_to_call_id = {}

    # --- _process_message_to_client (Needs Full Rewrite) ---
    async def _process_message_to_client(self, msg_data: str, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        """Processes messages received FROM OpenAI server BEFORE sending to client."""
        try:
            message = json.loads(msg_data)
            message_type = message.get("type")
            logger.debug(f"OpenAI Server -> Client: Type={message_type}, Data={message}")

            # Default: forward the message
            updated_message_str = msg_data

            # ** Handle OpenAI Server Events for Tool Calling **
            match message_type:
                case "response.output_item.added":
                    item = message.get("item", {})
                    if item.get("type") == "function_call":
                        # Store the mapping and initialize pending call data structure if not already started by delta
                        call_id = item.get("call_id")
                        item_id = item.get("id")
                        tool_name = item.get("name")
                        if call_id and item_id and tool_name:
                            logger.info(f"Tool call started: call_id={call_id}, item_id={item_id}, name={tool_name}")
                            self._item_to_call_id[item_id] = call_id
                            if not self._pending_tool_calls[call_id]:
                                # Store previous_item_id if available for grounding correlation
                                prev_item_id = message.get("previous_item_id")
                                self._pending_tool_calls[call_id] = PendingToolCall(call_id, tool_name, prev_item_id)
                        else:
                             logger.warning(f"Received function_call item without sufficient info: {item}")
                        updated_message_str = None # Don't forward this internal marker to client

                case "response.function_call_arguments.delta":
                    call_id = message.get("call_id")
                    delta = message.get("delta", "")
                    if call_id and call_id in self._pending_tool_calls:
                         self._pending_tool_calls[call_id].arguments_str += delta
                         # Maybe initialize PendingToolCall here if not already done by output_item.added
                         item_id = message.get("item_id") # Correlate back if needed
                         if not self._pending_tool_calls[call_id].tool_name and item_id:
                              # This path might be less common if output_item.added always comes first
                              logger.warning(f"Received arguments delta for call_id {call_id} before function_call item?")
                              # We might need the tool name here, which isn't in this event.
                              # Rely on output_item.added or function_call_arguments.done
                    elif call_id:
                         logger.warning(f"Received arguments delta for unknown or completed call_id: {call_id}")
                    updated_message_str = None # Don't forward deltas to client

                case "response.function_call_arguments.done":
                    call_id = message.get("call_id")
                    arguments_final = message.get("arguments") # Final arguments string
                    if call_id and call_id in self._pending_tool_calls:
                        logger.info(f"Completed receiving arguments for call_id={call_id}")
                        # Override any delta accumulation with the final complete arguments
                        self._pending_tool_calls[call_id].arguments_str = arguments_final
                        # Now we have everything needed to execute the tool
                        await self._execute_tool_call(call_id, server_ws, client_ws)
                    else:
                        logger.warning(f"Received arguments done for unknown or completed call_id: {call_id}")
                    updated_message_str = None # Don't forward this internal event

                case "response.output_item.done":
                     item = message.get("item", {})
                     if item.get("type") == "function_call":
                         # This might also signify the end of a function call if arguments didn't stream
                         call_id = item.get("call_id")
                         tool_name = item.get("name")
                         arguments = item.get("arguments") # Complete arguments might be here
                         item_id = item.get("id")

                         if call_id and tool_name and arguments is not None:
                             logger.info(f"Tool call item done: call_id={call_id}, name={tool_name}")
                             # Ensure pending call exists (might have been created by output_item.added)
                             if not self._pending_tool_calls[call_id]:
                                 # Need previous_item_id if grounding tool result goes to client
                                 # We lack previous_item_id here. Maybe find item_id's predecessor? Risky.
                                 # For now, assume grounding source correlation might be lost if we hit this path first.
                                 logger.warning(f"Initializing pending call {call_id} from output_item.done without previous_item_id.")
                                 self._pending_tool_calls[call_id] = PendingToolCall(call_id, tool_name)

                             # If arguments were provided directly (not streamed delta), use them
                             if self._pending_tool_calls[call_id].arguments_str == "":
                                 self._pending_tool_calls[call_id].arguments_str = arguments

                         else:
                              logger.warning(f"Received function_call item done without sufficient info: {item}")
                         updated_message_str = None # Don't forward internal processing

                case "response.done":
                    # The response cycle is finished.
                    # Check if we still have pending tool calls that finished *after* the last tool call execution
                    # but before the final response.done. This might happen with complex interleaving.
                    # The original logic of triggering a new response if pending calls exist seems plausible.
                    if len(self._pending_tool_calls) > 0:
                        logger.warning(f"Response done, but {len(self._pending_tool_calls)} tool calls still marked as pending. Clearing them.")
                        # Potentially trigger a final response.create here if needed based on app logic
                        # await server_ws.send_json({"type": "response.create"})
                        self._pending_tool_calls.clear()
                        self._item_to_call_id.clear()
                    # Filter out any function_call items from the final response sent to client
                    response_data = message.get("response", {})
                    if "output" in response_data:
                        original_outputs = response_data.get("output", [])
                        filtered_outputs = [item for item in original_outputs if item.get("type") != "function_call"]
                        if len(filtered_outputs) < len(original_outputs):
                            logger.info("Filtering function_call item(s) from final response.done event before sending to client.")
                            message["response"]["output"] = filtered_outputs
                            updated_message_str = json.dumps(message)
                    # Pass other response types through

                case "error":
                    logger.error(f"Received error from OpenAI: {message.get('error')}")
                    # Forward the error to the client

                # --- Default: Forward other message types ---
                case _:
                    # logger.debug(f"Forwarding unhandled server event type: {message_type}")
                    pass

            return updated_message_str
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from OpenAI server: {msg_data}")
            return None
        except Exception as e:
            logger.exception(f"Error processing message to client: {e}")
            # Optionally send an error message to the client
            # await client_ws.send_json({"type": "error", "error": {"message": "Internal server error processing message"}})
            return None # Avoid sending potentially malformed data

    async def _execute_tool_call(self, call_id: str, server_ws: web.WebSocketResponse, client_ws: web.WebSocketResponse):
        """Executes the tool function and sends result back."""
        pending_call = self._pending_tool_calls.pop(call_id, None) # Remove as we process it

        if not pending_call:
            logger.error(f"Attempted to execute non-existent or already completed tool call: {call_id}")
            return

        tool_name = pending_call.tool_name
        arguments_str = pending_call.arguments_str

        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' requested by model is not registered.")
            # Send error back to model? OpenAI API doesn't specify this, maybe just send empty result?
            # For now, just log it.
            return

        tool = self.tools[tool_name]
        logger.info(f"Executing tool '{tool_name}' for call_id={call_id} with args: {arguments_str}")

        try:
            # Parse arguments string as JSON
            args = json.loads(arguments_str)
            # Execute the tool's target function (e.g., _search_tool)
            result: ToolResult = await tool.target(args)

            # Send result back
            if result.destination == ToolResultDirection.TO_SERVER:
                 logger.info(f"Sending tool result for call_id {call_id} back to OpenAI server.")
                 await server_ws.send_json({
                     "type": "conversation.item.create",
                     "item": {
                         "type": "function_call_output",
                         "call_id": call_id,
                         "output": result.to_text() # Use the formatted text/JSON string
                     }
                 })
                 # After sending tool result, immediately ask for the next response
                 logger.info(f"Requesting next response from OpenAI after tool call {call_id}.")
                 await server_ws.send_json({"type": "response.create"})

            elif result.destination == ToolResultDirection.TO_CLIENT:
                logger.info(f"Sending tool result for call_id {call_id} directly to client.")
                # Use a custom message type to send structured data to client
                # Ensure previous_item_id exists if client needs it for correlation
                client_payload = {
                     "type": "extension.middle_tier_tool_response",
                     "tool_name": tool_name,
                     "tool_result": result.to_text() # Send parsed JSON back to client
                 }
                if pending_call.previous_item_id:
                    client_payload["previous_item_id"] = pending_call.previous_item_id
                else:
                    logger.warning(f"Missing previous_item_id for client-destined tool result {call_id}")

                await client_ws.send_json(client_payload)
                # Do we need to trigger a new response from OpenAI after client-side tool? Depends on app logic.
                # Maybe not, if the grounding info is just for display.

        except json.JSONDecodeError:
            logger.error(f"Failed to parse arguments for tool '{tool_name}' (call_id={call_id}): {arguments_str}")
            # Inform model? Log only for now.
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}' (call_id={call_id}): {e}")
            # Inform model? Log only for now.

        # Clean up item_id mapping related to this call_id
        items_to_remove = [item_id for item_id, c_id in self._item_to_call_id.items() if c_id == call_id]
        for item_id in items_to_remove:
            del self._item_to_call_id[item_id]


    # --- _process_message_to_server (Handles Client Events -> Server) ---
    async def _process_message_to_server(self, msg_data: str, client_ws: web.WebSocketResponse) -> Optional[str]:
        """Processes messages received FROM client BEFORE sending to OpenAI server."""
        try:
            message = json.loads(msg_data)
            message_type = message.get("type")
            # logger.debug(f"Client -> OpenAI Server: Type={message_type}, Data={message}")

            updated_message = message # Default: forward

            # Check if tools are attached and update tool_choice/tools if necessary
            has_tools = bool(self.tools)
            tool_schemas = [tool.schema for tool in self.tools.values()] if has_tools else []
            effective_tool_choice = "auto" if has_tools else "none"

            match message_type:
                case "session.update":
                    session_data = updated_message.get("session", {})
                    # Override tool settings based on attached tools
                    session_data["tools"] = tool_schemas
                    session_data["tool_choice"] = effective_tool_choice
                    if has_tools:
                         logger.info(f"Advertising {len(tool_schemas)} tools to OpenAI in session.update.")
                    # Apply server-side overrides like system_message
                    if self.system_message is not None:
                        session_data["instructions"] = self.system_message
                    # ... apply other overrides ...
                    updated_message["session"] = session_data
                case "response.create":
                    response_data = updated_message.get("response", {})
                    # Override tool settings for this specific response
                    response_data["tools"] = tool_schemas
                    response_data["tool_choice"] = effective_tool_choice
                    if has_tools:
                        logger.info(f"Advertising {len(tool_schemas)} tools to OpenAI in response.create.")
                    # Apply server-side overrides if needed
                    if self.system_message is not None and "instructions" not in response_data:
                        response_data["instructions"] = self.system_message
                    updated_message["response"] = response_data
                case "conversation.item.create":
                    # Prevent client from sending function_call_output directly
                    item = updated_message.get("item", {})
                    if item.get("type") == "function_call_output":
                        logger.error("Client attempted to send function_call_output; blocking.")
                        return None # Don't forward
                # --- Default: Forward other client events ---
                case _:
                     # logger.debug(f"Forwarding unhandled client event type: {message_type}")
                     pass

            return json.dumps(updated_message)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from client: {msg_data}")
            return None
        except Exception as e:
            logger.exception(f"Error processing message to server: {e}")
            return None

    # --- Updated _forward_messages (Connection Logic - Mostly Unchanged from previous OpenAI switch) ---
    async def _forward_messages(self, ws: web.WebSocketResponse):
        target_url = f"{self.websocket_base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        logger.info(f"Connecting to OpenAI Realtime API: {target_url}")

        # Reset state for new connection
        self._pending_tool_calls.clear()
        self._item_to_call_id.clear()

        # Proxy logic can be kept if needed
        proxy_url = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')

        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(
                    target_url,
                    headers=headers,
                    timeout=30.0,
                    # ssl=ssl_context, # Add if SSL verification issues arise
                    proxy=proxy_url # Add proxy argument if env var is set
                ) as target_ws:
                    logger.info("Successfully connected to OpenAI Realtime API.")

                    async def from_client_to_server():
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                new_msg_str = await self._process_message_to_server(msg.data, ws)
                                if new_msg_str is not None:
                                    await target_ws.send_str(new_msg_str)
                                    # logger.debug("Forwarded processed message to OpenAI")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f'Client WebSocket connection closed with exception {ws.exception()!r}')
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("Client WebSocket connection closed normally.")
                                break
                            else:
                                 logger.warning(f"Received unexpected message type from client: {msg.type}")

                        if not target_ws.closed:
                            logger.info("Client disconnected, closing connection to OpenAI.")
                            await target_ws.close()

                    async def from_server_to_client():
                        async for msg in target_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                new_msg_str = await self._process_message_to_client(msg.data, ws, target_ws)
                                if new_msg_str is not None:
                                    await ws.send_str(new_msg_str)
                                    # logger.debug("Forwarded processed message to Client")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f'OpenAI WebSocket connection closed with exception {target_ws.exception()!r}')
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("OpenAI WebSocket connection closed.")
                                break
                            else:
                                logger.warning(f"Received unexpected message type from OpenAI: {msg.type}")

                        if not ws.closed:
                             logger.info("OpenAI disconnected, closing connection to client.")
                             await ws.close()

                    await asyncio.gather(from_client_to_server(), from_server_to_client())

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Failed to connect to OpenAI WebSocket: {e}")
                await ws.close(code=aiohttp.WSCloseCode.TRY_AGAIN_LATER, message=b'Could not connect to backend API')
            except aiohttp.WSServerHandshakeError as e:
                 logger.error(f"WebSocket handshake failed with OpenAI: {e.status} {e.message}")
                 await ws.close(code=aiohttp.WSCloseCode.PROTOCOL_ERROR, message=b'Backend authentication or protocol error')
            except Exception as e:
                logger.exception(f"Unhandled exception in WebSocket forwarding: {e}")
                if not ws.closed:
                    await ws.close(code=aiohttp.WSCloseCode.INTERNAL_ERROR, message=b'Internal server error')

        logger.info("WebSocket forwarding finished.")



    # --- _websocket_handler and attach_to_app remain unchanged ---
    async def _websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("Client WebSocket connection established.")
        await self._forward_messages(ws)
        logger.info("Client WebSocket connection finished.")
        return ws

    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)

"""
请你分析数据库实现的方案，都有哪些方案可选，各有什么特点
"""