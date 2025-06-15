import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";

import { Button } from "./button";
import GroundingFile from "./grounding-file";

import { GroundingFile as GroundingFileType, HistoryItem } from "@/types";

import { useTranslation } from "react-i18next";

type Properties = {
    history: HistoryItem[];
    show: boolean;
    onClosed: () => void;
    onSelectedGroundingFile: (file: GroundingFileType) => void;
};

export default function HistoryPanel({ show, history, onClosed, onSelectedGroundingFile }: Properties) {
    const { t } = useTranslation();

    return (
        <AnimatePresence>
            {show && (
                <motion.div
                    initial={{ opacity: 0, x: "100%" }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: "100%" }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="fixed inset-y-0 right-0 z-40 w-full overflow-y-auto bg-background border-l border-border shadow-lg sm:w-96"
                >
                    <div className="p-4">
                        <div className="mb-4 flex items-center justify-between">
                            <h2 className="text-xl font-bold text-foreground">{t("history.answerHistory")}</h2>
                            <Button variant="ghost" size="sm" onClick={onClosed}>
                                <X className="h-5 w-5" />
                            </Button>
                        </div>
                        {history.length > 0 ? (
                            history.map((item, index) => (
                                <div key={index} className="mb-6 border-b border-border pb-6">
                                    <h3 className="mb-2 font-semibold text-foreground">{item.id}</h3>
                                    <pre className="mb-2 overflow-x-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-sm">
                                        <code className="block h-24 overflow-y-auto text-muted-foreground">{item.transcript}</code>
                                    </pre>
                                    <div className="mt-2 flex flex-wrap gap-2">
                                        {item.groundingFiles.map((file, index) => (
                                            <GroundingFile key={index} value={file} onClick={() => onSelectedGroundingFile(file)} />
                                        ))}
                                    </div>
                                </div>
                            ))
                        ) : (
                            <p className="text-muted-foreground">{t("history.noHistory")}</p>
                        )}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
