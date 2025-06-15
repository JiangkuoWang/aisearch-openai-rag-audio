import { Check, FileText, X } from "lucide-react";

export interface FileStatus {
    name: string;
    progress: number; // 0-100
    status: 'pending' | 'uploading' | 'success' | 'error';
    error?: string;
}

interface FileUploadItemProps {
    file: File;
    status: FileStatus;
    onRemove: () => void;
}

export default function FileUploadItem({ file, status, onRemove }: FileUploadItemProps) {
    // 文件大小格式化
    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    // 文件类型图标颜色 - 主题适配版本
    const getFileColor = (fileName: string) => {
        const ext = fileName.split('.').pop()?.toLowerCase();
        if (['pdf'].includes(ext || '')) return 'text-red-500 dark:text-red-400';
        if (['doc', 'docx'].includes(ext || '')) return 'text-blue-500 dark:text-blue-400';
        if (['txt'].includes(ext || '')) return 'text-muted-foreground';
        return 'text-green-500 dark:text-green-400';
    };

    return (
        <div className="bg-card rounded-lg shadow-sm border border-border p-3 mb-2 relative transition-all duration-200">
            <div className="flex items-center gap-3">
                <FileText className={`h-6 w-6 ${getFileColor(file.name)}`} />

                <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-center">
                        <h3 className="font-medium text-sm truncate pr-6 text-card-foreground">{file.name}</h3>
                        <span className="text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
                    </div>
                    
                    {/* 进度条 - 仅在上传中或错误时显示 */}
                    {(status.status === 'pending' || status.status === 'uploading' || status.status === 'error') && (
                        <div className="w-full h-1.5 bg-secondary rounded-full mt-2 overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-300 ${
                                    status.status === 'error'
                                        ? 'bg-destructive'
                                        : 'bg-primary'
                                }`}
                                style={{ width: `${status.progress}%` }}
                            />
                        </div>
                    )}

                    {/* 状态信息与删除按钮 */}
                    <div className="flex justify-between items-center mt-1">
                        <span className="text-xs text-muted-foreground">
                            {status.status === 'pending' && '等待处理...'}
                            {status.status === 'uploading' && `上传中 ${status.progress}%`}
                            {status.status === 'success' && (
                                <span className="text-green-600 dark:text-green-400 flex items-center">
                                    <Check className="h-4 w-4 mr-1" />上传成功
                                </span>
                            )}
                            {status.status === 'error' && (
                                <span className="text-destructive flex items-center">
                                    <X className="h-4 w-4 mr-1" />{status.error || '上传失败'}
                                </span>
                            )}
                        </span>

                        {/* 移除按钮 - 现在与状态行在同一行 */}
                        <button
                            onClick={onRemove}
                            className="text-muted-foreground hover:text-foreground transition-colors p-1"
                            aria-label="移除文件"
                        >
                            <X className="h-4 w-4" />
                        </button>
                    </div>
                </div>
            </div>
            
            {/* 移除按钮 - 改到状态行上 */}
        </div>
    );
}
