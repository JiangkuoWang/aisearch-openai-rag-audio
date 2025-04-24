import { useRef, useState, useEffect } from "react";
import { Upload, Loader } from "lucide-react";
import { RagProviderType } from "./types";
import FileUploadItem, { FileStatus } from "./FileUploadItem";

interface RagSelectorProps {
    ragType: RagProviderType;
    setRagType: (t: RagProviderType) => void;
    files: FileList | null;
    setFiles: (f: FileList | null) => void;
    uploading: boolean;
    doUpload: () => Promise<boolean>;
}

const ragTypes: { value: RagProviderType; label: string; color: string; tooltip: string }[] = [
    { value: "in_memory", label: "RAG", color: "bg-purple-300", tooltip: "普通模式：内存存储，适合小规模使用" },
    { value: "llama_index", label: "GraphRAG", color: "bg-blue-300", tooltip: "高级模式：基于 LlamaIndex，支持索引持久化" },
];

export default function RagSelector({ ragType, setRagType, files, setFiles, uploading, doUpload }: RagSelectorProps) {
    const fileInput = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);
    
    // 处理文件输入变化
    const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        console.log("handleFileInputChange triggered");
        if (e.target.files && e.target.files.length > 0) {
            console.log("New files selected:", e.target.files);
            try {
                // 合并现有文件和新选择的文件
                if (files) {
                    const existingFiles = Array.from(files);
                    const newFiles = Array.from(e.target.files);
                    
                    // 创建一个新的 DataTransfer 对象
                    const dataTransfer = new DataTransfer();
                    
                    // 添加现有文件（不包括已经成功上传的文件）
                    existingFiles.forEach(file => {
                        // 如果文件状态不是success，或者没有状态信息，则添加到新的列表中
                        const fileStatus = fileStatuses[file.name];
                        if (!fileStatus || fileStatus.status !== 'success') {
                            dataTransfer.items.add(file);
                        } else {
                            // 已上传成功的文件也需要保留
                            dataTransfer.items.add(file);
                        }
                    });
                    
                    // 添加新文件（排除重复）
                    newFiles.forEach(newFile => {
                        if (!existingFiles.some(f => f.name === newFile.name)) {
                            dataTransfer.items.add(newFile);
                        }
                    });
                    
                    const finalFiles = dataTransfer.files;
                    console.log("Combined files ready for state update:", finalFiles);
                    setFiles(finalFiles); // This triggers the useEffects
                } else {
                    console.log("Setting initial files:", e.target.files);
                    setFiles(e.target.files);
                }
            } catch (error) {
                console.error("Error in handleFileInputChange:", error);
                // 出错时，至少设置新选择的文件
                setFiles(e.target.files);
            }
        } else {
            console.log("No files selected or event target files are null.");
        }
    };
    const [fileStatuses, setFileStatuses] = useState<{[key: string]: FileStatus}>({});
    
    // 当文件选择变化时，初始化状态
    useEffect(() => {
        if (files) {
            const newStatuses: {[key: string]: FileStatus} = {};
            
            // 首先，保留所有文件的现有状态
            Object.entries(fileStatuses).forEach(([filename, status]) => {
                newStatuses[filename] = status;
            });
            
            // 处理当前文件列表中的文件
            Array.from(files).forEach(file => {
                // 如果这个文件没有状态，或者状态是error，初始化/重置为pending
                if (!fileStatuses[file.name] || fileStatuses[file.name].status === 'error') {
                    newStatuses[file.name] = {
                        name: file.name,
                        progress: 0,
                        status: 'pending'
                    };
                }
                // 其他状态（success, uploading, pending）保持不变
            });
            
            console.log("Updated file statuses after file change:", newStatuses);
            setFileStatuses(newStatuses);
        } else {
            // 当files为null时，清空所有状态
            setFileStatuses({});
        }
    }, [files]);
    
    // 当有新文件且RAG模式不为none时，自动触发上传
    useEffect(() => {
        console.log("Auto-upload effect running. Files:", files, "RAG Type:", ragType, "Uploading:", uploading, "Statuses:", fileStatuses);
        if (files && files.length > 0 && ragType !== 'none' && !uploading) {
            // 找出所有pending状态的文件
            const pendingFiles = Object.values(fileStatuses).filter(s => s.status === 'pending');
            const hasPending = pendingFiles.length > 0;
            console.log("Checking for pending files:", hasPending, "Pending files details:", pendingFiles);

            if (hasPending) {
                console.log("Found pending files, calling handleUpload...");
                // 使用setTimeout避免潜在的状态更新冲突
                setTimeout(() => {
                    if (!uploading) { // 再次检查，确保在延迟后仍未上传中
                        handleUpload();
                    }
                }, 50);
            } else {
                console.log("No pending files found.");
            }
        } else {
            console.log("Auto-upload conditions not met.");
        }
    }, [files, ragType, fileStatuses, uploading]);
    
    // 处理上传过程
    const handleUpload = async () => {
        if (!files || !files.length || ragType === 'none') return;
        
        // 筛选出需要上传的文件（仅处理pending状态的）
        const pendingFilesList = Array.from(files).filter(file => 
            fileStatuses[file.name]?.status === 'pending'
        );
        
        if (pendingFilesList.length === 0) {
            console.log("No pending files to upload");
            return;
        }
        
        console.log("Uploading files:", pendingFilesList.map(f => f.name));
        
        // 更新所有pending状态为uploading
        const updatedStatuses = {...fileStatuses};
        pendingFilesList.forEach(file => {
            if (updatedStatuses[file.name]?.status === 'pending') {
                updatedStatuses[file.name].status = 'uploading';
            }
        });
        setFileStatuses(updatedStatuses);
        
        // 模拟上传进度
        const progressInterval = setInterval(() => {
            setFileStatuses(prev => {
                const updated = {...prev};
                let allDone = true;
                
                pendingFilesList.forEach(file => {
                    const name = file.name;
                    if (updated[name]?.status === 'uploading' && updated[name].progress < 90) {
                        updated[name].progress += Math.floor(Math.random() * 10) + 5;
                        if (updated[name].progress > 90) updated[name].progress = 90;
                        allDone = false;
                    }
                });
                
                if (allDone) clearInterval(progressInterval);
                return updated;
            });
        }, 300);
        
        // 执行上传
        try {
            // 准备FormData - 只包含需要上传的文件
            const formData = new FormData();
            pendingFilesList.forEach(file => formData.append("file", file));
            
            // 确保App组件中的doUpload能够正确处理这批文件
            // 执行上传 - 这里我们不更改files状态，而是直接上传选定的文件
            const success = await doUpload();
            
            // 更新状态为success或error
            clearInterval(progressInterval);
            setFileStatuses(prev => {
                const updated = {...prev};
                pendingFilesList.forEach(file => {
                    const name = file.name;
                    if (updated[name]?.status === 'uploading') {
                        if (success) {
                            updated[name].status = 'success';
                            updated[name].progress = 100;
                        } else {
                            updated[name].status = 'error';
                            updated[name].error = '上传失败，请重试';
                        }
                    }
                });
                return updated;
            });
            
        } catch (error) {
            console.error("Upload error:", error);
            clearInterval(progressInterval);
            // 设置所有uploading状态为error
            setFileStatuses(prev => {
                const updated = {...prev};
                pendingFilesList.forEach(file => {
                    const name = file.name;
                    if (updated[name]?.status === 'uploading') {
                        updated[name].status = 'error';
                        updated[name].error = '上传过程中出错';
                    }
                });
                return updated;
            });
        }
    };
    
    // 移除文件
    const removeFile = (fileName: string) => {
        if (!files) return;
        
        try {
            const newFiles = Array.from(files).filter(f => f.name !== fileName);
            const dataTransfer = new DataTransfer();
            
            // 添加剩余文件到DataTransfer
            newFiles.forEach(file => {
                try {
                    dataTransfer.items.add(file);
                } catch (itemError) {
                    console.error(`Failed to add file ${file.name} to DataTransfer:`, itemError);
                }
            });
            
            // 如果没有文件了，设置为null
            if (newFiles.length === 0) {
                setFiles(null);
            } else {
                setFiles(dataTransfer.files);
            }
            
            // 移除对应状态
            const newStatuses = {...fileStatuses};
            delete newStatuses[fileName];
            setFileStatuses(newStatuses);
        } catch (error) {
            console.error(`Error removing file ${fileName}:`, error);
            
            // 降级方案：如果不能使用DataTransfer，至少更新状态
            try {
                const newStatuses = {...fileStatuses};
                delete newStatuses[fileName]; // 从状态中移除文件
                setFileStatuses(newStatuses);
                
                // 通知用户刷新页面以完全移除文件
                console.warn("File status updated but file list could not be modified. Consider refreshing the page if issues persist.");
            } catch (innerError) {
                console.error("Failed to update file status:", innerError);
            }
        }
    };

    return (
        <div className="mb-8 w-full max-w-md mx-auto p-6 rounded-2xl shadow-lg bg-white/80 animate-fade-in">
            <fieldset className="mb-4 flex flex-col items-center">
                <div className="flex gap-4 justify-center mb-6">
                    {ragTypes.map(rt => (
                        <div
                            key={rt.value}
                            title={rt.tooltip}
                            onClick={() => setRagType(ragType === rt.value ? "none" : rt.value)}
                            className={`flex items-center space-x-2 px-4 py-2 rounded-lg cursor-pointer transition-all duration-200 shadow-md border ${ragType === rt.value 
                                ? rt.color + " scale-110 border-purple-500" 
                                : "bg-white hover:bg-gray-50 border-gray-200"}`}
                        >
                            <span className="font-medium text-base select-none flex items-center">
                                {rt.label}
                                <span className={`ml-2 text-xs px-2 py-0.5 rounded ${rt.value === 'in_memory' ? 'bg-gray-200 text-gray-800' : 'bg-yellow-200 text-yellow-800'}`}>
                                    {rt.value === 'in_memory' ? '普通' : '高级'}
                                </span>
                            </span>
                        </div>
                    ))}
                </div>
            </fieldset>
            <div className="mt-6 w-full">
                {ragType !== "none" ? (
                    <div className="flex flex-col w-full">
                        {/* 文件列表 */}
                        {files && files.length > 0 ? (
                            <div className="w-full">
                                <div className="flex justify-between items-center mb-3">
                                    <h3 className="font-semibold text-gray-700">已选择的文件</h3>
                                    {uploading && (
                                        <div className="flex items-center text-sm text-blue-600">
                                            <Loader className="animate-spin h-4 w-4 mr-1" />
                                            <span>正在处理...</span>
                                        </div>
                                    )}
                                </div>
                                
                                <div className="space-y-2">
                                    {Array.from(files).map(file => (
                                        <FileUploadItem 
                                            key={file.name}
                                            file={file}
                                            status={fileStatuses[file.name] || { name: file.name, progress: 0, status: 'pending' }}
                                            onRemove={() => removeFile(file.name)}
                                        />
                                    ))}
                                </div>
                                
                                {/* 添加新文件的按钮 */}
                                <div 
                                    className="mt-4 border-2 border-dashed border-gray-300 rounded-lg p-3 text-center cursor-pointer hover:bg-gray-50 transition-colors"
                                    onClick={() => {
                                        console.log('"Add more files" button clicked.');
                                        const additionalFileInput = document.getElementById('additionalFileInput') as HTMLInputElement;
                                        if (additionalFileInput) {
                                            console.log('Resetting additional file input value and triggering click...');
                                            additionalFileInput.value = ''; // Reset is important
                                            additionalFileInput.click();
                                        } else {
                                            console.error('Additional file input element not found.');
                                        }
                                    }}
                                >
                                    <input
                                        id="additionalFileInput"
                                        type="file"
                                        multiple
                                        style={{ display: "none" }}
                                        onChange={handleFileInputChange}
                                    />
                                    <Upload className="h-5 w-5 mx-auto text-gray-400 mb-1" />
                                    <p className="text-sm text-gray-500">添加更多文件</p>
                                </div>
                            </div>
                        ) : (
                            <div
                                className={`w-full border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${dragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-gray-50'}`}
                                onDragOver={e => { e.preventDefault(); setDragActive(true); }}
                                onDragLeave={e => { e.preventDefault(); setDragActive(false); }}
                                onDrop={e => {
                                    e.preventDefault();
                                    setDragActive(false);
                                    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                                        setFiles(e.dataTransfer.files);
                                    }
                                }}
                                onClick={() => fileInput.current?.click()}
                                style={{ cursor: "pointer" }}
                            >
                                <input
                                    type="file"
                                    multiple
                                    ref={fileInput}
                                    style={{ display: "none" }}
                                    onChange={handleFileInputChange} // Ensure this uses the logging function
                                />
                                <div className="flex flex-col items-center text-gray-700">
                                    <Upload className="h-10 w-10 text-gray-400 mb-2" />
                                    <p className="font-medium">拖拽文件到此区域或点击选择文件</p>
                                    <p className="text-sm text-gray-500 mt-1">支持 PDF、DOCX、TXT 文件</p>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center text-gray-500 p-4 border border-dashed rounded-lg">
                        请先选择上方的 RAG 模式以启用文件上传
                    </div>
                )}
            </div>
        </div>
    );
}
