import React, { useCallback, useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

import GroundingFileView from "@/components/ui/grounding-file-view";

import { GroundingFile, RagProviderType } from "@/types";
import RagSelector from "@/RagSelector";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onOpen: () => void;
  ragType: RagProviderType;
  setRagType: (type: RagProviderType) => void;
  files: FileList | null;
  setFiles: (files: FileList | null) => void;
  uploading: boolean;
  doUpload: () => Promise<boolean>;
  selectedFile: GroundingFile | null;
  setSelectedFile: (file: GroundingFile | null) => void;
}

export const Sidebar: React.FC<SidebarProps> = React.memo(({
  isOpen,
  onClose,
  onOpen,
  ragType,
  setRagType,
  files,
  setFiles,
  uploading,
  doUpload,
  selectedFile,
  setSelectedFile
}) => {
  // 侧栏关闭处理函数 - 使用useCallback优化性能
  const handleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  // 侧栏展开处理函数 - 使用useCallback优化性能
  const handleOpen = useCallback(() => {
    onOpen();
  }, [onOpen]);



  // 文件视图关闭处理函数 - 使用useCallback优化性能
  const handleFileViewClose = useCallback(() => {
    setSelectedFile(null);
  }, [setSelectedFile]);

  // 使用useMemo优化侧栏面板样式计算
  const sidebarClassName = useMemo(() => {
    return `fixed top-0 right-0 h-screen w-96 bg-background/95 backdrop-blur-sm border-l border-border rounded-l-2xl shadow-lg p-6 overflow-y-auto z-40 transform transition-all duration-300 ease-in-out ${
      isOpen ? 'translate-x-0' : 'translate-x-full'
    }`;
  }, [isOpen]);

  // 使用useMemo优化文件列表渲染
  const fileListItems = useMemo(() => {
    if (!files || files.length === 0) return null;

    return Array.from(files).map(f => (
      <li key={f.name} className="truncate">{f.name}</li>
    ));
  }, [files]);

  return (
    <>
      {/* 收起时显示的展开按钮 */}
      {!isOpen && (
        <button
          onClick={handleOpen}
          className="fixed top-1/2 right-0 z-40 p-2 bg-background border border-border rounded-l-md shadow-md flex items-center justify-center transform -translate-y-1/2 hover:bg-accent transition-colors"
        >
          <ChevronLeft className="h-5 w-5 text-foreground" />
        </button>
      )}

      {/* 侧栏面板 */}
      <aside className={sidebarClassName}>
        <button
          onClick={handleClose}
          className="absolute top-4 left-4 p-2 bg-background hover:bg-accent border border-border rounded-full shadow-md transition-colors"
          aria-label="关闭侧栏"
        >
          <ChevronRight className="h-5 w-5 text-foreground" />
        </button>
        <h2 className="text-2xl font-semibold mb-4">Select RAG Mode & Upload</h2>
        <RagSelector
          ragType={ragType}
          setRagType={setRagType}
          files={files}
          setFiles={setFiles}
          uploading={uploading}
          doUpload={doUpload}
        />
        {files && files.length > 0 && (
          <div className="mt-6 bg-card rounded-lg shadow border border-border p-4">
            <h3 className="font-semibold mb-2 text-card-foreground">Selected Files</h3>
            <ul className="space-y-1 text-sm text-muted-foreground">
              {fileListItems}
            </ul>
          </div>
        )}
      </aside>

      {/* GroundingFileView模态框 */}
      <GroundingFileView groundingFile={selectedFile} onClosed={handleFileViewClose} />
    </>
  );
});

Sidebar.displayName = "Sidebar";
