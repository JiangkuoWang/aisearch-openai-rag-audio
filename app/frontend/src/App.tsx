import { useState, useEffect } from "react";
import { Mic, MicOff } from "lucide-react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { GroundingFiles } from "@/components/ui/grounding-files";
import GroundingFileView from "@/components/ui/grounding-file-view";

import useRealTime from "@/hooks/useRealtime";
import useAudioRecorder from "@/hooks/useAudioRecorder";
import useAudioPlayer from "@/hooks/useAudioPlayer";

import { GroundingFile, ToolResult, RagProviderType } from "./types";
import RagSelector from "./RagSelector";

// 导入认证相关组件
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { AuthModal } from "./components/AuthModal";
import { UserMenu } from "./components/UserMenu";
import { getAuthStatus } from "./lib/api";

import logo from "./assets/logo.svg";

// 封装应用内容，以便可以使用认证上下文
function AppContent() {
    const [isRecording, setIsRecording] = useState(false);
    const [groundingFiles, setGroundingFiles] = useState<GroundingFile[]>([]);
    const [selectedFile, setSelectedFile] = useState<GroundingFile | null>(null);

    // RAG selection and upload state
    const [ragType, setRagType] = useState<RagProviderType>("none");
    const [files, setFiles] = useState<FileList | null>(null);
    const [uploading, setUploading] = useState(false);
    const [dbConnected, setDbConnected] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    
    // 认证状态
    const [authModalOpen, setAuthModalOpen] = useState(false);
    const { isAuthenticated } = useAuth();
    
    // 检查认证状态
    useEffect(() => {
        const checkAuthStatus = async () => {
            try {
                const status = await getAuthStatus();
                console.log("Auth status:", status);
            } catch (error) {
                console.error("Failed to check auth status:", error);
            }
        };
        
        checkAuthStatus();
    }, []);

    const { startSession, addUserAudio, inputAudioBufferClear } = useRealTime({
        onWebSocketOpen: () => console.log("WebSocket connection opened"),
        onWebSocketClose: () => console.log("WebSocket connection closed"),
        onWebSocketError: event => console.error("WebSocket error:", event),
        onReceivedError: message => console.error("error", message),
        onReceivedResponseAudioDelta: message => {
            isRecording && playAudio(message.delta);
        },
        onReceivedInputAudioBufferSpeechStarted: () => {
            stopAudioPlayer();
        },
        onReceivedExtensionMiddleTierToolResponse: message => {
            const result: ToolResult = JSON.parse(message.tool_result);
            const files: GroundingFile[] = result.sources.map(x => {
                return { id: x.chunk_id, name: x.title, content: x.chunk };
            });

            setGroundingFiles(files);
        }
    });

    const { reset: resetAudioPlayer, play: playAudio, stop: stopAudioPlayer } = useAudioPlayer();
    const { start: startAudioRecording, stop: stopAudioRecording } = useAudioRecorder({ onAudioRecorded: addUserAudio });

    const onToggleListening = async () => {
        if (!isRecording) {
            startSession();
            await startAudioRecording();
            resetAudioPlayer();

            setIsRecording(true);
        } else {
            await stopAudioRecording();
            stopAudioPlayer();
            inputAudioBufferClear();

            setIsRecording(false);
        }
    };

    // Upload RAG config and files
    const doUpload = async (): Promise<boolean> => {
        // 如果未登录，则打开登录模态框
        if (!isAuthenticated) {
            setAuthModalOpen(true);
            return false;
        }
        
        try {
            setUploading(true);
            // Set provider type
            const configResponse = await fetch("/rag-config", {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    // 如果有认证令牌，添加到请求头
                    ...(getAuthHeaders() || {})
                },
                body: JSON.stringify({ provider_type: ragType })
            });
            
            if (!configResponse.ok) {
                console.error("Failed to set RAG config:", await configResponse.text());
                return false;
            }
            
            // Upload files if needed
            if (ragType !== "none" && files) {
                const fd = new FormData();
                Array.from(files).forEach(f => fd.append("file", f));
                const uploadResponse = await fetch("/upload", { 
                    method: "POST", 
                    body: fd,
                    headers: getAuthHeaders() || undefined
                });
                
                if (!uploadResponse.ok) {
                    console.error("Upload failed:", await uploadResponse.text());
                    return false;
                }
            }
            return true;
        } catch (error) {
            console.error("Upload error:", error);
            return false;
        } finally {
            setUploading(false);
        }
    };
    
    // 获取认证请求头
    const getAuthHeaders = () => {
        const token = localStorage.getItem('voice_rag_auth_token');
        return token ? { 'Authorization': `Bearer ${token}` } : null;
    };

    const { t } = useTranslation();

    return (
        <div className="flex min-h-screen flex-col bg-gray-100 text-gray-900">
            <div className="flex justify-between items-center p-4">
                <div>
                    <img src={logo} alt="Azure logo" className="h-16 w-16" />
                </div>
                <div className="mr-4">
                    <UserMenu onLoginClick={() => setAuthModalOpen(true)} />
                </div>
            </div>
            <main className={`flex flex-grow flex-col items-center justify-center transition-all duration-300 ${dbConnected && sidebarOpen ? 'pr-96' : ''}`}>
                <h1 className="mb-8 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-4xl font-bold text-transparent md:text-7xl">
                    {t("app.title")}
                </h1>
                <div className="mb-4 flex flex-col items-center justify-center space-y-4">
                    <Button
                        onClick={onToggleListening}
                        className={`h-12 w-60 ${isRecording ? "bg-red-600 hover:bg-red-700" : "bg-purple-500 hover:bg-purple-600"}`}
                        aria-label={isRecording ? t("app.stopRecording") : t("app.startRecording")}
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
                    <Button
                        onClick={() => { setDbConnected(!dbConnected); setSidebarOpen(true); }}
                        className={`h-12 w-60 ${dbConnected ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white rounded-lg transition`}
                    >
                        {dbConnected ? 'Disconnect Database' : 'Connect to database'}
                    </Button>
                </div>
                {dbConnected && (
                    <>
                        {/* 收起时显示的展开按钮 */}
                        {!sidebarOpen && (
                            <button 
                                onClick={() => setSidebarOpen(true)}
                                className="fixed top-1/2 right-0 z-40 p-2 bg-white rounded-l-md shadow-md flex items-center justify-center transform -translate-y-1/2"
                            >
                                <ChevronLeft className="h-5 w-5" />
                            </button>
                        )}
                        
                        {/* 侧栏面板 */}
                        <aside className={`fixed top-0 right-0 h-screen w-96 bg-white/90 backdrop-blur-sm rounded-l-2xl shadow-lg p-6 overflow-y-auto z-40 transform transition-all duration-300 ease-in-out ${sidebarOpen ? 'translate-x-0' : 'translate-x-full'}`}>
                            <button 
                                onClick={() => setSidebarOpen(false)} 
                                className="absolute top-4 left-4 p-2 bg-white hover:bg-gray-100 rounded-full shadow-md transition-colors"
                                aria-label="关闭侧栏"
                            >
                                <ChevronRight className="h-5 w-5" />
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
                            <div className="mt-6 bg-white rounded-lg shadow p-4">
                                <h3 className="font-semibold mb-2">Selected Files</h3>
                                <ul className="space-y-1 text-sm text-gray-700">
                                    {Array.from(files).map(f => (
                                        <li key={f.name} className="truncate">{f.name}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        </aside>
                    </>
                )}
                <GroundingFiles files={groundingFiles} onSelected={setSelectedFile} />
            </main>

            <footer className="py-4 text-center">
                <p>{t("app.footer")}</p>
            </footer>

            <GroundingFileView groundingFile={selectedFile} onClosed={() => setSelectedFile(null)} />
            
            {/* 认证模态框 */}
            <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} />
        </div>
    );
}

// 主应用组件，使用AuthProvider包装内容
function App() {
    return (
        <AuthProvider>
            <AppContent />
        </AuthProvider>
    );
}

export default App;
