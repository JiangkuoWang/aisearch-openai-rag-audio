import React from "react";

import { ThemeToggle } from "@/components/ui/theme-toggle";
import { LogtoUserMenu } from "./LogtoUserMenu";

import logo from "@/assets/logo.svg";

interface AppHeaderProps {
  // 头部组件目前不需要外部props，所有功能都是自包含的
  // 如果将来需要传递认证状态或其他数据，可以在这里添加
}

export const AppHeader: React.FC<AppHeaderProps> = React.memo(() => {
  return (
    <div className="flex justify-between items-center p-4 border-b border-border">
      <div>
        <img src={logo} alt="Azure logo" className="h-16 w-16" />
      </div>
      <div className="flex items-center space-x-3 mr-4">
        {/* 主题切换按钮 */}
        <ThemeToggle />

        {/* Logto认证菜单 */}
        <LogtoUserMenu />
      </div>
    </div>
  );
});

AppHeader.displayName = "AppHeader";
