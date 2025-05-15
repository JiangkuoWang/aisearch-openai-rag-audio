import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { LogtoProvider } from '@logto/react';

import { I18nextProvider } from "react-i18next";
import i18next from "./i18n/config";

import { Router } from "./Router";
import { logtoConfig } from "./logto-config";
import "./index.css";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <LogtoProvider config={logtoConfig}>
            <I18nextProvider i18n={i18next}>
                <Router />
            </I18nextProvider>
        </LogtoProvider>
    </StrictMode>
);
