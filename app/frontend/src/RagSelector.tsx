import { useRef, useState } from "react";
import { RagProviderType } from "./types";

interface RagSelectorProps {
    ragType: RagProviderType;
    setRagType: (t: RagProviderType) => void;
    files: FileList | null;
    setFiles: (f: FileList | null) => void;
    uploading: boolean;
    doUpload: () => void;
}

const ragTypes: { value: RagProviderType; label: string; color: string }[] = [
    { value: "none", label: "无 (关闭)", color: "bg-gray-300" },
    { value: "in_memory", label: "In-Memory", color: "bg-purple-300" },
    { value: "llama_index", label: "LlamaIndex", color: "bg-blue-300" },
];

export default function RagSelector({ ragType, setRagType, files, setFiles, uploading, doUpload }: RagSelectorProps) {
    const fileInput = useRef<HTMLInputElement>(null);
    const [dragActive, setDragActive] = useState(false);

    return (
        <div className="mb-8 w-full max-w-md mx-auto p-6 rounded-2xl shadow-lg bg-white/80 animate-fade-in">
            <fieldset className="mb-4 flex flex-col items-center">
                <legend className="font-semibold text-lg mb-2 text-gray-700">RAG 模式：</legend>
                <div className="flex gap-4">
                    {ragTypes.map(rt => (
                        <label key={rt.value} className={`flex items-center px-3 py-1 rounded-full cursor-pointer transition-all duration-200 shadow-sm border-2 border-transparent ${ragType === rt.value ? rt.color + ' scale-110 border-purple-500' : 'hover:border-gray-400'}`}>
                            <input
                                type="radio"
                                name="ragMode"
                                value={rt.value}
                                checked={ragType === rt.value}
                                onChange={() => setRagType(rt.value)}
                                className="sr-only"
                            />
                            <span className="font-medium text-base select-none">{rt.label}</span>
                        </label>
                    ))}
                </div>
            </fieldset>
            {ragType !== "none" && (
                <div className="flex flex-col items-center mt-4 w-full">
                    <div
                        className={`w-full border-2 border-dashed rounded-xl p-4 text-center transition-all duration-200 ${dragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 bg-gray-50'}`}
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
                            onChange={e => setFiles(e.target.files)}
                        />
                        <div className="text-gray-700">
                            {files && files.length > 0 ? (
                                <ul className="mb-2 text-sm text-left max-h-20 overflow-y-auto">
                                    {Array.from(files).map(f => (
                                        <li key={f.name} className="truncate">{f.name}</li>
                                    ))}
                                </ul>
                            ) : (
                                <span className="text-gray-400">拖拽文件到此或点击选择文件上传</span>
                            )}
                        </div>
                    </div>
                    <button
                        onClick={doUpload}
                        disabled={!files || uploading}
                        className={`mt-4 px-6 py-2 rounded-xl font-semibold shadow-md transition-all duration-200 text-white ${uploading ? 'bg-gray-400 animate-pulse' : 'bg-gradient-to-r from-purple-500 to-blue-400 hover:from-purple-600 hover:to-blue-500'}`}
                    >
                        {uploading ? "上传中..." : "上传知识库"}
                    </button>
                </div>
            )}
        </div>
    );
}
