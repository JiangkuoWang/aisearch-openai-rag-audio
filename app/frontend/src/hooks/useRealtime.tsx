import { useState, useEffect } from "react";
import useWebSocket from "react-use-websocket";
import { useLogto } from "@logto/react";

import {
    InputAudioBufferAppendCommand,
    InputAudioBufferClearCommand,
    Message,
    ResponseAudioDelta,
    ResponseAudioTranscriptDelta,
    ResponseDone,
    SessionUpdateCommand,
    ExtensionMiddleTierToolResponse,
    ResponseInputAudioTranscriptionCompleted
} from "@/types";

type Parameters = {
    useDirectAoaiApi?: boolean; // If true, the middle tier will be skipped and the AOAI ws API will be called directly
    aoaiEndpointOverride?: string;
    aoaiApiKeyOverride?: string;
    aoaiModelOverride?: string;

    enableInputAudioTranscription?: boolean;
    onWebSocketOpen?: () => void;
    onWebSocketClose?: () => void;
    onWebSocketError?: (event: Event) => void;
    onWebSocketMessage?: (event: MessageEvent<any>) => void;

    onReceivedResponseAudioDelta?: (message: ResponseAudioDelta) => void;
    onReceivedInputAudioBufferSpeechStarted?: (message: Message) => void;
    onReceivedResponseDone?: (message: ResponseDone) => void;
    onReceivedExtensionMiddleTierToolResponse?: (message: ExtensionMiddleTierToolResponse) => void;
    onReceivedResponseAudioTranscriptDelta?: (message: ResponseAudioTranscriptDelta) => void;
    onReceivedInputAudioTranscriptionCompleted?: (message: ResponseInputAudioTranscriptionCompleted) => void;
    onReceivedError?: (message: Message) => void;
};

export default function useRealTime({
    useDirectAoaiApi,
    aoaiEndpointOverride,
    aoaiApiKeyOverride,
    aoaiModelOverride,
    enableInputAudioTranscription,
    onWebSocketOpen,
    onWebSocketClose,
    onWebSocketError,
    onWebSocketMessage,
    onReceivedResponseDone,
    onReceivedResponseAudioDelta,
    onReceivedResponseAudioTranscriptDelta,
    onReceivedInputAudioBufferSpeechStarted,
    onReceivedExtensionMiddleTierToolResponse,
    onReceivedInputAudioTranscriptionCompleted,
    onReceivedError
}: Parameters) {
    const [socketUrl, setSocketUrl] = useState<string | (() => Promise<string>) | null>(null);
    const { getAccessToken, isAuthenticated } = useLogto();

    useEffect(() => {
        const setupSocketUrl = async () => {
            if (useDirectAoaiApi) {
                setSocketUrl(`${aoaiEndpointOverride}/openai/realtime?api-key=${aoaiApiKeyOverride}&deployment=${aoaiModelOverride}&api-version=2024-10-01-preview`);
            } else {
                if (isAuthenticated) {
                    try {
                        const token = await getAccessToken();
                        console.log("Attempting to connect WebSocket. Logto Token:", token ? "Token Acquired" : "No Token");
                        if (token) {
                            setSocketUrl(`ws://localhost:8765/realtime?token=${token}`);
                        } else {
                            console.log("No auth token from Logto. WebSocket connection will not be attempted.");
                            setSocketUrl(null);
                        }
                    } catch (error) {
                        console.error("Error getting Logto access token for WebSocket:", error);
                        setSocketUrl(null);
                    }
                } else {
                    console.log("User not authenticated via Logto. WebSocket connection will not be attempted.");
                    setSocketUrl(null);
                }
            }
        };

        setupSocketUrl();
    }, [
        isAuthenticated, 
        getAccessToken, 
        useDirectAoaiApi, 
        aoaiEndpointOverride, 
        aoaiApiKeyOverride, 
        aoaiModelOverride
    ]);

    const { sendJsonMessage } = useWebSocket(socketUrl, {
        onOpen: () => {
            console.log("WebSocket connection opened."); // 保留用于确认连接成功
            onWebSocketOpen?.();
        },
        onClose: () => {
            console.log("WebSocket connection closed.");
            onWebSocketClose?.();
        },
        onError: event => {
            console.error("WebSocket error:", event);
            onWebSocketError?.(event);
        },
        onMessage: event => onMessageReceived(event),
        shouldReconnect: () => true
    });

    const startSession = () => {
        const command: SessionUpdateCommand = {
            type: "session.update",
            session: {
                turn_detection: {
                    type: "server_vad"
                }
            }
        };

        if (enableInputAudioTranscription) {
            command.session.input_audio_transcription = {
                model: "whisper-1"
            };
        }

        sendJsonMessage(command);
    };

    const addUserAudio = (base64Audio: string) => {
        const command: InputAudioBufferAppendCommand = {
            type: "input_audio_buffer.append",
            audio: base64Audio
        };

        sendJsonMessage(command);
    };

    const inputAudioBufferClear = () => {
        const command: InputAudioBufferClearCommand = {
            type: "input_audio_buffer.clear"
        };

        sendJsonMessage(command);
    };

    const onMessageReceived = (event: MessageEvent<any>) => {
        onWebSocketMessage?.(event);

        let message: Message;
        try {
            message = JSON.parse(event.data);
        } catch (e) {
            console.error("Failed to parse JSON message:", e);
            throw e;
        }

        switch (message.type) {
            case "response.done":
                onReceivedResponseDone?.(message as ResponseDone);
                break;
            case "response.audio.delta":
                onReceivedResponseAudioDelta?.(message as ResponseAudioDelta);
                break;
            case "response.audio_transcript.delta":
                onReceivedResponseAudioTranscriptDelta?.(message as ResponseAudioTranscriptDelta);
                break;
            case "input_audio_buffer.speech_started":
                onReceivedInputAudioBufferSpeechStarted?.(message);
                break;
            case "conversation.item.input_audio_transcription.completed":
                onReceivedInputAudioTranscriptionCompleted?.(message as ResponseInputAudioTranscriptionCompleted);
                break;
            case "extension.middle_tier_tool_response":
                onReceivedExtensionMiddleTierToolResponse?.(message as ExtensionMiddleTierToolResponse);
                break;
            case "error":
                onReceivedError?.(message);
                break;
        }
    };

    return { startSession, addUserAudio, inputAudioBufferClear };
}
