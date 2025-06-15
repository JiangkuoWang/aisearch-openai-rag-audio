import { useState, useCallback } from "react";

import useRealTime from "@/hooks/useRealtime";
import useAudioRecorder from "@/hooks/useAudioRecorder";
import useAudioPlayer from "@/hooks/useAudioPlayer";

import { GroundingFile, ToolResult } from "@/types";

interface UseAudioSessionParams {
  isAuthenticated: boolean;
  onGroundingFilesUpdate: (files: GroundingFile[]) => void;
}

interface UseAudioSessionReturn {
  isRecording: boolean;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  playAudio: (delta: string) => void;
  stopAudio: () => void;
  resetAudio: () => void;
}

/**
 * 自定义Hook：音频会话管理
 * 封装音频录制、播放和WebSocket通信的完整逻辑
 */
export function useAudioSession({
  isAuthenticated,
  onGroundingFilesUpdate
}: UseAudioSessionParams): UseAudioSessionReturn {
  const [isRecording, setIsRecording] = useState(false);

  // 初始化音频相关hooks
  const { reset: resetAudioPlayer, play: playAudio, stop: stopAudioPlayer } = useAudioPlayer();

  // WebSocket事件处理函数 - 使用useCallback确保引用稳定
  const handleWebSocketOpen = useCallback(() => {
    console.log("WebSocket connection opened");
  }, []);

  const handleWebSocketClose = useCallback(() => {
    console.log("WebSocket connection closed");
  }, []);

  const handleWebSocketError = useCallback((event: Event) => {
    console.error("WebSocket error:", event);
  }, []);

  const handleReceivedError = useCallback((message: any) => {
    console.error("error", message);
  }, []);

  const handleReceivedResponseAudioDelta = useCallback((message: any) => {
    if (isRecording) {
      playAudio(message.delta);
    }
  }, [isRecording, playAudio]);

  const handleReceivedInputAudioBufferSpeechStarted = useCallback(() => {
    stopAudioPlayer();
  }, [stopAudioPlayer]);

  const handleReceivedExtensionMiddleTierToolResponse = useCallback((message: any) => {
    const result: ToolResult = JSON.parse(message.tool_result);
    const files: GroundingFile[] = result.sources.map(x => {
      return { id: x.chunk_id, name: x.title, content: x.chunk };
    });
    onGroundingFilesUpdate(files);
  }, [onGroundingFilesUpdate]);

  // 初始化WebSocket和录音相关hooks
  const { startSession, addUserAudio, inputAudioBufferClear } = useRealTime({
    onWebSocketOpen: handleWebSocketOpen,
    onWebSocketClose: handleWebSocketClose,
    onWebSocketError: handleWebSocketError,
    onReceivedError: handleReceivedError,
    onReceivedResponseAudioDelta: handleReceivedResponseAudioDelta,
    onReceivedInputAudioBufferSpeechStarted: handleReceivedInputAudioBufferSpeechStarted,
    onReceivedExtensionMiddleTierToolResponse: handleReceivedExtensionMiddleTierToolResponse
  });

  const { start: startAudioRecording, stop: stopAudioRecording } = useAudioRecorder({
    onAudioRecorded: addUserAudio
  });

  // 开始录音 - 使用useCallback优化性能
  const startRecording = useCallback(async () => {
    if (!isAuthenticated) {
      console.warn("User not authenticated. Please log in to start recording.");
      return;
    }

    if (!isRecording) {
      startSession();
      await startAudioRecording();
      resetAudioPlayer();
      setIsRecording(true);
    }
  }, [
    isAuthenticated,
    isRecording,
    startSession,
    startAudioRecording,
    resetAudioPlayer
  ]);

  // 停止录音 - 使用useCallback优化性能
  const stopRecording = useCallback(async () => {
    if (isRecording) {
      await stopAudioRecording();
      stopAudioPlayer();
      inputAudioBufferClear();
      setIsRecording(false);
    }
  }, [
    isRecording,
    stopAudioRecording,
    stopAudioPlayer,
    inputAudioBufferClear
  ]);

  // 停止音频播放 - 使用useCallback优化性能
  const stopAudio = useCallback(() => {
    stopAudioPlayer();
  }, [stopAudioPlayer]);

  // 重置音频播放器 - 使用useCallback优化性能
  const resetAudio = useCallback(() => {
    resetAudioPlayer();
  }, [resetAudioPlayer]);

  return {
    isRecording,
    startRecording,
    stopRecording,
    playAudio,
    stopAudio,
    resetAudio
  };
}
