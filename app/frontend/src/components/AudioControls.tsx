import React, { useCallback, useMemo } from "react";
import { Mic, MicOff } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { useAudioSession } from "@/hooks/useAudioSession";

import { GroundingFile } from "@/types";

interface AudioControlsProps {
  isAuthenticated: boolean;
  onGroundingFilesUpdate: (files: GroundingFile[]) => void;
}

export const AudioControls: React.FC<AudioControlsProps> = React.memo(({
  isAuthenticated,
  onGroundingFilesUpdate
}) => {
  const { t } = useTranslation();

  // 使用自定义Hook管理音频会话
  const {
    isRecording,
    startRecording,
    stopRecording
  } = useAudioSession({
    isAuthenticated,
    onGroundingFilesUpdate
  });

  // 录音切换处理函数 - 使用useCallback优化性能
  const onToggleListening = useCallback(async () => {
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // 使用useMemo优化按钮样式计算
  const buttonClassName = useMemo(() => {
    return `h-12 w-60 transition-all duration-200 ${
      isRecording
        ? "bg-destructive hover:bg-destructive/90 text-destructive-foreground"
        : "bg-purple-500 hover:bg-purple-600 text-white dark:bg-purple-600 dark:hover:bg-purple-700"
    }`;
  }, [isRecording]);

  // 使用useMemo优化aria-label计算
  const ariaLabel = useMemo(() => {
    return isRecording ? t("app.stopRecording") : t("app.startRecording");
  }, [isRecording, t]);

  return (
    <Button
      onClick={onToggleListening}
      className={buttonClassName}
      aria-label={ariaLabel}
    >
      {isRecording ? (
        <>
          <MicOff className="mr-2 h-4 w-4" />
          {t("app.stopConversation")}
        </>
      ) : (
        <>
          <Mic className="mr-2 h-6 w-6" />
        </>
      )}
    </Button>
  );
});

AudioControls.displayName = "AudioControls";
