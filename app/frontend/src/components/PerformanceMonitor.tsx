import React, { useEffect, useRef, useState } from 'react';

interface PerformanceMonitorProps {
  componentName: string;
  children: React.ReactNode;
  enabled?: boolean;
}

interface RenderStats {
  renderCount: number;
  lastRenderTime: number;
  averageRenderTime: number;
  totalRenderTime: number;
}

/**
 * 性能监控组件
 * 在开发环境中监控组件的重渲染次数和渲染时间
 */
export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  componentName,
  children,
  enabled = process.env.NODE_ENV === 'development'
}) => {
  const renderCountRef = useRef(0);
  const renderTimesRef = useRef<number[]>([]);
  const startTimeRef = useRef<number>(0);
  const [stats, setStats] = useState<RenderStats>({
    renderCount: 0,
    lastRenderTime: 0,
    averageRenderTime: 0,
    totalRenderTime: 0
  });

  // 记录渲染开始时间
  if (enabled) {
    startTimeRef.current = performance.now();
  }

  useEffect(() => {
    if (!enabled) return;

    const endTime = performance.now();
    const renderTime = endTime - startTimeRef.current;
    
    renderCountRef.current += 1;
    renderTimesRef.current.push(renderTime);
    
    // 只保留最近50次渲染的数据
    if (renderTimesRef.current.length > 50) {
      renderTimesRef.current = renderTimesRef.current.slice(-50);
    }
    
    const totalTime = renderTimesRef.current.reduce((sum, time) => sum + time, 0);
    const averageTime = totalTime / renderTimesRef.current.length;
    
    const newStats: RenderStats = {
      renderCount: renderCountRef.current,
      lastRenderTime: renderTime,
      averageRenderTime: averageTime,
      totalRenderTime: totalTime
    };
    
    setStats(newStats);
    
    // 在控制台输出性能信息
    if (renderTime > 16) { // 超过16ms（60fps阈值）时警告
      console.warn(`🐌 ${componentName} slow render: ${renderTime.toFixed(2)}ms (render #${renderCountRef.current})`);
    } else if (renderCountRef.current % 10 === 0) { // 每10次渲染输出一次统计
      console.log(`📊 ${componentName} stats: ${renderCountRef.current} renders, avg: ${averageTime.toFixed(2)}ms`);
    }
  });

  if (!enabled) {
    return <>{children}</>;
  }

  return (
    <div data-performance-monitor={componentName}>
      {children}
      {/* 开发环境中显示性能统计 */}
      {process.env.NODE_ENV === 'development' && (
        <div 
          style={{
            position: 'fixed',
            top: 10,
            right: 10,
            background: 'rgba(0,0,0,0.8)',
            color: 'white',
            padding: '8px',
            fontSize: '12px',
            borderRadius: '4px',
            zIndex: 9999,
            fontFamily: 'monospace'
          }}
          title={`Performance stats for ${componentName}`}
        >
          <div>{componentName}</div>
          <div>Renders: {stats.renderCount}</div>
          <div>Last: {stats.lastRenderTime.toFixed(2)}ms</div>
          <div>Avg: {stats.averageRenderTime.toFixed(2)}ms</div>
        </div>
      )}
    </div>
  );
};

/**
 * 高阶组件：为组件添加性能监控
 */
export function withPerformanceMonitor<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  componentName?: string
) {
  const displayName = componentName || WrappedComponent.displayName || WrappedComponent.name || 'Component';
  
  const WithPerformanceMonitor = (props: P) => (
    <PerformanceMonitor componentName={displayName}>
      <WrappedComponent {...props} />
    </PerformanceMonitor>
  );
  
  WithPerformanceMonitor.displayName = `withPerformanceMonitor(${displayName})`;
  
  return WithPerformanceMonitor;
}

/**
 * Hook：用于在组件内部监控性能
 */
export function usePerformanceMonitor(componentName: string, enabled = process.env.NODE_ENV === 'development') {
  const renderCountRef = useRef(0);
  const startTimeRef = useRef<number>(0);
  
  if (enabled) {
    startTimeRef.current = performance.now();
  }
  
  useEffect(() => {
    if (!enabled) return;
    
    const endTime = performance.now();
    const renderTime = endTime - startTimeRef.current;
    renderCountRef.current += 1;
    
    if (renderTime > 16) {
      console.warn(`🐌 ${componentName} slow render: ${renderTime.toFixed(2)}ms (render #${renderCountRef.current})`);
    }
  });
  
  return {
    renderCount: renderCountRef.current,
    logRender: (message?: string) => {
      if (enabled) {
        console.log(`🔄 ${componentName} render #${renderCountRef.current}${message ? `: ${message}` : ''}`);
      }
    }
  };
}
