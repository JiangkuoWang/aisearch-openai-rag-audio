import { useState, useCallback } from "react";
import { useLogto } from "@logto/react";

import { RagProviderType } from "@/types";

interface UseFileUploadReturn {
  ragType: RagProviderType;
  setRagType: (type: RagProviderType) => void;
  files: FileList | null;
  setFiles: (files: FileList | null) => void;
  uploading: boolean;
  doUpload: () => Promise<boolean>;
  uploadError: string | null;
  clearError: () => void;
}

/**
 * 自定义Hook：文件上传和RAG配置管理
 * 封装RAG类型管理、文件选择、上传进度、错误处理等逻辑
 */
export function useFileUpload(): UseFileUploadReturn {
  const [ragType, setRagType] = useState<RagProviderType>("none");
  const [files, setFiles] = useState<FileList | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { isAuthenticated, getAccessToken } = useLogto();

  // 清除错误信息 - 使用useCallback优化性能
  const clearError = useCallback(() => {
    setUploadError(null);
  }, []);

  // 获取认证头部 - 使用useCallback优化性能
  const getAuthHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: HeadersInit = { "Content-Type": "application/json" };
    
    try {
      const token = await getAccessToken("https://your-api-resource-indicator");
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    } catch (err) {
      console.error("Failed to get access token:", err);
      throw new Error("Authentication failed");
    }
    
    return headers;
  }, [getAccessToken]);

  // 获取文件上传认证头部 - 使用useCallback优化性能
  const getFileUploadHeaders = useCallback(async (): Promise<HeadersInit> => {
    const headers: HeadersInit = {};
    
    try {
      const token = await getAccessToken("https://your-api-resource-indicator");
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
    } catch (err) {
      console.error("Failed to get access token for file upload:", err);
      throw new Error("Authentication failed for file upload");
    }
    
    return headers;
  }, [getAccessToken]);

  // 设置RAG配置 - 使用useCallback优化性能
  const setRagConfig = useCallback(async (providerType: RagProviderType): Promise<boolean> => {
    try {
      const headers = await getAuthHeaders();
      
      const configResponse = await fetch("/rag-config", {
        method: "POST",
        headers: headers,
        body: JSON.stringify({ provider_type: providerType })
      });

      if (!configResponse.ok) {
        const errorText = await configResponse.text();
        console.error("Failed to set RAG config:", errorText);
        setUploadError(`Failed to set RAG configuration: ${errorText}`);
        return false;
      }

      return true;
    } catch (error) {
      console.error("RAG config error:", error);
      setUploadError(`RAG configuration error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return false;
    }
  }, [getAuthHeaders]);

  // 上传文件 - 使用useCallback优化性能
  const uploadFiles = useCallback(async (fileList: FileList): Promise<boolean> => {
    try {
      const headers = await getFileUploadHeaders();
      
      const fd = new FormData();
      Array.from(fileList).forEach(f => fd.append("files", f));

      const uploadResponse = await fetch("/upload", {
        method: "POST",
        body: fd,
        headers: headers
      });

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        console.error("Upload failed:", errorText);
        setUploadError(`File upload failed: ${errorText}`);
        return false;
      }

      return true;
    } catch (error) {
      console.error("Upload error:", error);
      setUploadError(`Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return false;
    }
  }, [getFileUploadHeaders]);

  // 执行上传 - 使用useCallback优化性能
  const doUpload = useCallback(async (): Promise<boolean> => {
    // 清除之前的错误
    setUploadError(null);

    // 检查认证状态
    if (!isAuthenticated) {
      const errorMsg = "User not authenticated. Please log in to upload files.";
      console.warn(errorMsg);
      setUploadError(errorMsg);
      return false;
    }

    try {
      setUploading(true);

      // 设置RAG配置
      const configSuccess = await setRagConfig(ragType);
      if (!configSuccess) {
        return false;
      }

      // 如果需要上传文件
      if (ragType !== "none" && files && files.length > 0) {
        const uploadSuccess = await uploadFiles(files);
        if (!uploadSuccess) {
          return false;
        }
      }

      // 上传成功
      console.log("Upload completed successfully");
      return true;
    } catch (error) {
      console.error("Upload error:", error);
      setUploadError(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      return false;
    } finally {
      setUploading(false);
    }
  }, [isAuthenticated, ragType, files, setRagConfig, uploadFiles]);

  return {
    ragType,
    setRagType,
    files,
    setFiles,
    uploading,
    doUpload,
    uploadError,
    clearError
  };
}
