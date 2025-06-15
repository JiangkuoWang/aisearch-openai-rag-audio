import React, { useCallback } from "react";

import { Button } from "@/components/ui/button";

interface DatabaseControlsProps {
  dbConnected: boolean;
  isAuthenticated: boolean;
  onToggleConnection: () => void;
}

export const DatabaseControls: React.FC<DatabaseControlsProps> = React.memo(({
  dbConnected,
  isAuthenticated,
  onToggleConnection
}) => {
  // 数据库连接切换处理函数 - 使用useCallback优化性能
  const handleToggleConnection = useCallback(() => {
    // 如果未登录，则提示用户登录
    if (!isAuthenticated) {
      console.warn("User not authenticated. Please log in to connect to the database.");
      return;
    }
    
    // 调用父组件传递的切换函数
    onToggleConnection();
  }, [isAuthenticated, onToggleConnection]);

  return (
    <Button
      onClick={handleToggleConnection}
      className={`h-12 w-60 text-white rounded-lg transition-all duration-200 ${
        dbConnected
          ? 'bg-red-500 hover:bg-red-600 dark:bg-red-600 dark:hover:bg-red-700'
          : 'bg-green-500 hover:bg-green-600 dark:bg-green-600 dark:hover:bg-green-700'
      }`}
    >
      {dbConnected ? 'Disconnect Database' : 'Connect to database'}
    </Button>
  );
});

DatabaseControls.displayName = "DatabaseControls";
