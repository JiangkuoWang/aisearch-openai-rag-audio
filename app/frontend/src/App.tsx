import { useState, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";

import { GroundingFiles } from "@/components/ui/grounding-files";
import { AudioControls } from "@/components/AudioControls";
import { DatabaseControls } from "@/components/DatabaseControls";
import { Sidebar } from "@/components/Sidebar";
import { AppHeader } from "@/components/AppHeader";
import { usePerformanceMonitor } from "@/components/PerformanceMonitor";
import { useFileUpload } from "@/hooks/useFileUpload";

import { GroundingFile } from "./types";

import { useLogto } from "@logto/react";
import { ThemeProvider } from "@/contexts/ThemeContext";

// 封装应用内容，以便可以使用认证上下文http://localhost:8765/callback?code=kriVDFau_yRBz0pmxZtreVIt67VTcd9LOWPJpR5n8xJ&state=mF-hc05QZ7nYW0nRjLalSYbc10bizxqt1tp1sMQhCAfGNQcKlZ28FYo2RcbWGrXpeSB2acOJBfP_xxJhMXiejg&iss=https%3A%2F%2Fousbav.logto.app%2Foidc
function AppContent() {
    // 性能监控
    usePerformanceMonitor('AppContent');

    const [groundingFiles, setGroundingFiles] = useState<GroundingFile[]>([]);
    const [selectedFile, setSelectedFile] = useState<GroundingFile | null>(null);
    const [dbConnected, setDbConnected] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);

    // 认证状态
    const { isAuthenticated } = useLogto();

    // 使用自定义Hook管理文件上传
    const {
        ragType,
        setRagType,
        files,
        setFiles,
        uploading,
        doUpload,
        uploadError,
        clearError
    } = useFileUpload();

    // 数据库连接切换处理函数 - 使用useCallback优化性能
    const handleToggleConnection = useCallback(() => {
        setDbConnected(!dbConnected);
        setSidebarOpen(true);
    }, [dbConnected]);

    // 侧栏控制处理函数 - 使用useCallback优化性能
    const handleSidebarClose = useCallback(() => {
        setSidebarOpen(false);
    }, []);

    const handleSidebarOpen = useCallback(() => {
        setSidebarOpen(true);
    }, []);



    const { t } = useTranslation();

    // 使用useMemo优化主内容区域样式计算
    const mainContentClassName = useMemo(() => {
        return `flex flex-grow flex-col items-center justify-center transition-all duration-300 ${
            dbConnected && sidebarOpen ? 'pr-96' : ''
        }`;
    }, [dbConnected, sidebarOpen]);

    return (
        <div className="flex min-h-screen flex-col bg-background text-foreground">
            <AppHeader />
            {/* 错误提示 */}
            {uploadError && (
                <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-2 mx-4 mt-2 rounded-md flex justify-between items-center">
                    <span className="text-sm">{uploadError}</span>
                    <button
                        onClick={clearError}
                        className="text-destructive hover:text-destructive/80 ml-2"
                        aria-label="Close error message"
                    >
                        ×
                    </button>
                </div>
            )}
            <main className={mainContentClassName}>
                <h1 className="mb-8 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-4xl font-bold text-transparent md:text-7xl dark:from-purple-400 dark:to-pink-400">
                    {t("app.title")}
                </h1>
                <div className="mb-4 flex flex-col items-center justify-center space-y-4">
                    <AudioControls
                        isAuthenticated={isAuthenticated}
                        onGroundingFilesUpdate={setGroundingFiles}
                    />
                    <DatabaseControls
                        dbConnected={dbConnected}
                        isAuthenticated={isAuthenticated}
                        onToggleConnection={handleToggleConnection}
                    />
                </div>
                <GroundingFiles files={groundingFiles} onSelected={setSelectedFile} />
                {dbConnected && (
                    <Sidebar
                        isOpen={sidebarOpen}
                        onClose={handleSidebarClose}
                        onOpen={handleSidebarOpen}
                        ragType={ragType}
                        setRagType={setRagType}
                        files={files}
                        setFiles={setFiles}
                        uploading={uploading}
                        doUpload={doUpload}
                        selectedFile={selectedFile}
                        setSelectedFile={setSelectedFile}
                    />
                )}
            </main>

            <footer className="py-4 text-center border-t border-border">
                <p className="text-muted-foreground">{t("app.footer")}</p>
            </footer>

            {/* 认证模态框 */}
            {/* <AuthModal isOpen={authModalOpen} onClose={() => setAuthModalOpen(false)} /> */}
        </div>
    );
}

// 主应用组件
// 注意：我们不再需要AuthProvider，因为现在使用LogtoProvider
// 但为了保持兼容性，我们暂时保留了原有的认证系统
function App() {
    return (
        <ThemeProvider defaultTheme="system" storageKey="voice-rag-theme">
            <AppContent />
        </ThemeProvider>
    );
}

export default App;
