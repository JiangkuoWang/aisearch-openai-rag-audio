import asyncio
import json
import logging
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

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
    _tools_pending = {}
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
        # Removed Azure credential handling logic

    # --- _process_message_to_client (Needs Full Rewrite) ---
    async def _process_message_to_client(self, msg_data: str, client_ws: web.WebSocketResponse, server_ws: web.WebSocketResponse) -> Optional[str]:
        """Processes messages received FROM OpenAI server BEFORE sending to client."""
        try:
            message = json.loads(msg_data)
            message_type = message.get("type")
            logger.debug(f"OpenAI Server -> Client: Type={message_type}, Data={message}")

            # Default: forward the message
            updated_message_str = msg_data

            # ** Handle OpenAI Server Events **
            match message_type:
                case "session.created" | "session.updated":
                    # Forward session status
                    pass
                case "response.created":
                     # Log start of response
                     pass
                case "response.text.delta" | "response.audio.delta":
                    # Forward content deltas
                    pass
                case "response.function_call_arguments.delta" | "response.function_call_arguments.done":
                    logger.warning(f"Ignoring server event '{message_type}' as tools are disabled.")
                    updated_message_str = None # Don't forward
                case "response.output_item.added" | "response.output_item.done":
                    item = message.get("item", {})
                    if item.get("type") == "function_call":
                        logger.warning(f"Ignoring server event '{message_type}' with function_call item as tools are disabled.")
                        updated_message_str = None # Don't forward
                    # else: forward other item types
                case "response.done":
                    # Check if response contains function calls and filter if needed
                    response_data = message.get("response", {})
                    if "output" in response_data:
                        original_outputs = response_data.get("output", [])
                        filtered_outputs = [item for item in original_outputs if item.get("type") != "function_call"]
                        if len(filtered_outputs) < len(original_outputs):
                             logger.warning("Filtering function_call item(s) from response.done event.")
                             if not filtered_outputs:
                                 # If only function calls were present, maybe don't send 'done'? Or send with empty output?
                                 # Let's filter and send modified event for now.
                                 message["response"]["output"] = filtered_outputs
                                 updated_message_str = json.dumps(message)
                             else:
                                 message["response"]["output"] = filtered_outputs
                                 updated_message_str = json.dumps(message)
                case "error":
                    logger.error(f"Received error from OpenAI: {message.get('error')}")
                    # Forward the error to the client
                # Add cases for other relevant server events based on OpenAI docs
                case _:
                    logger.debug(f"Forwarding unhandled server event type: {message_type}")

            return updated_message_str
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from OpenAI server: {msg_data}")
            return None
        except Exception as e:
            logger.exception(f"Error processing message to client: {e}")
            return None # Avoid sending potentially malformed data

    # --- _process_message_to_server (Needs Full Rewrite) ---
    async def _process_message_to_server(self, msg_data: str, client_ws: web.WebSocketResponse) -> Optional[str]:
        """Processes messages received FROM client BEFORE sending to OpenAI server."""
        try:
            message = json.loads(msg_data)
            message_type = message.get("type")
            logger.debug(f"Client -> OpenAI Server: Type={message_type}, Data={message}")

            # Default: forward the message
            updated_message = message

            # ** Handle OpenAI Client Events **
            match message_type:
                case "session.update":
                    session_data = updated_message.get("session", {})
                    # Enforce disabling tools if client tries to enable them
                    if "tools" in session_data or "tool_choice" in session_data:
                         logger.warning("Client attempted to modify tools/tool_choice in session.update; overriding to disabled.")
                         session_data["tools"] = []
                         session_data["tool_choice"] = "none"
                    # Apply server-side overrides if needed (e.g., self.system_message)
                    if self.system_message is not None:
                         session_data["instructions"] = self.system_message
                    # ... apply other overrides like temperature, max_tokens if desired ...
                    updated_message["session"] = session_data
                case "input_audio_buffer.append":
                    # Usually forward directly
                    pass
                case "conversation.item.create":
                    item = updated_message.get("item", {})
                    if item.get("type") == "function_call_output":
                        logger.error("Client attempted to send function_call_output; blocking as tools are disabled.")
                        return None # Don't forward
                    # else: forward user messages, etc.
                case "response.create":
                    response_data = updated_message.get("response", {})
                    # Enforce disabling tools if client tries to enable them
                    if "tools" in response_data or "tool_choice" in response_data:
                        logger.warning("Client attempted to set tools/tool_choice in response.create; overriding to disabled.")
                        response_data["tools"] = []
                        response_data["tool_choice"] = "none"
                    # Apply server-side overrides if needed
                    if self.system_message is not None and "instructions" not in response_data:
                        response_data["instructions"] = self.system_message
                    updated_message["response"] = response_data
                # Add cases for other client events if needed (e.g., commit, clear buffer)
                case _:
                     logger.debug(f"Forwarding unhandled client event type: {message_type}")

            return json.dumps(updated_message)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from client: {msg_data}")
            return None
        except Exception as e:
            logger.exception(f"Error processing message to server: {e}")
            return None

    # --- Updated _forward_messages ---
    async def _forward_messages(self, ws: web.WebSocketResponse):
        # Construct the target WebSocket URL with model parameter
        target_url = f"{self.websocket_base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        logger.info(f"Connecting to OpenAI Realtime API: {target_url}")

        import ssl
        ssl_context = ssl.create_default_context()


        # Remove base_url from session, connect directly to full URL
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(target_url, headers=headers, timeout=30.0, ssl=ssl_context) as target_ws:
                    logger.info("Successfully connected to OpenAI Realtime API.")
                    # The gather logic remains conceptually the same
                    async def from_client_to_server():
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                # Use the REWRITTEN processing function
                                new_msg_str = await self._process_message_to_server(msg.data, ws)
                                if new_msg_str is not None:
                                    await target_ws.send_str(new_msg_str)
                                    logger.debug("Forwarded processed message to OpenAI")
                                else:
                                     logger.debug("Message processing decided not to forward to OpenAI")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f'Client WebSocket connection closed with exception {ws.exception()!r}')
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("Client WebSocket connection closed normally.")
                                break
                        # Close target if client closes
                        if not target_ws.closed:
                            logger.info("Client disconnected, closing connection to OpenAI.")
                            await target_ws.close()

                    async def from_server_to_client():
                        async for msg in target_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                # Use the REWRITTEN processing function
                                new_msg_str = await self._process_message_to_client(msg.data, ws, target_ws)
                                if new_msg_str is not None:
                                    await ws.send_str(new_msg_str)
                                    logger.debug("Forwarded processed message to Client")
                                else:
                                     logger.debug("Message processing decided not to forward to Client")

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f'OpenAI WebSocket connection closed with exception {target_ws.exception()!r}')
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("OpenAI WebSocket connection closed.")
                                break
                        # Close client ws if server closes
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



    async def _websocket_handler(self, request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await self._forward_messages(ws)
        return ws
    
    def attach_to_app(self, app, path):
        app.router.add_get(path, self._websocket_handler)
