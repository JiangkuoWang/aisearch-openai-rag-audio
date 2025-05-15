import { useState, useEffect } from 'react';
import { LogtoCallback } from './components/LogtoCallback';
import App from './App';

/**
 * 简单的路由组件
 * 用于处理Logto回调路径
 */
export function Router() {
  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  useEffect(() => {
    // 监听路径变化
    const handleLocationChange = () => {
      setCurrentPath(window.location.pathname);
    };

    window.addEventListener('popstate', handleLocationChange);
    
    return () => {
      window.removeEventListener('popstate', handleLocationChange);
    };
  }, []);

  // 根据路径渲染不同的组件
  if (currentPath === '/callback') {
    return <LogtoCallback />;
  }

  // 默认渲染主应用
  return <App />;
}
