import { type Theme } from '@/contexts/ThemeContext';

/**
 * 主题工具函数集合
 */

// 存储键常量
export const THEME_STORAGE_KEY = 'voice-rag-theme';

/**
 * 获取系统主题偏好
 * @returns 'light' | 'dark'
 */
export function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') {
    return 'light';
  }
  
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * 从localStorage获取保存的主题设置
 * @param defaultTheme 默认主题
 * @returns 保存的主题或默认主题
 */
export function getStoredTheme(defaultTheme: Theme = 'system'): Theme {
  if (typeof window === 'undefined') {
    return defaultTheme;
  }
  
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored && ['light', 'dark', 'system'].includes(stored)) {
      return stored as Theme;
    }
  } catch (error) {
    console.warn('Failed to read theme from localStorage:', error);
  }
  
  return defaultTheme;
}

/**
 * 保存主题设置到localStorage
 * @param theme 要保存的主题
 */
export function setStoredTheme(theme: Theme): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.warn('Failed to save theme to localStorage:', error);
  }
}

/**
 * 解析主题为实际应用的主题
 * @param theme 主题设置
 * @returns 实际的主题 ('light' | 'dark')
 */
export function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme === 'system') {
    return getSystemTheme();
  }
  return theme;
}

/**
 * 应用主题到DOM
 * @param theme 要应用的主题
 */
export function applyTheme(theme: 'light' | 'dark'): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  const root = window.document.documentElement;
  
  // 移除之前的主题类
  root.classList.remove('light', 'dark');
  
  // 添加新的主题类
  root.classList.add(theme);
  
  // 设置data属性用于CSS选择器
  root.setAttribute('data-theme', theme);
}

/**
 * 创建系统主题变化监听器
 * @param callback 主题变化时的回调函数
 * @returns 清理函数
 */
export function createSystemThemeListener(callback: (theme: 'light' | 'dark') => void): () => void {
  if (typeof window === 'undefined') {
    return () => {};
  }
  
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  
  const handleChange = (e: MediaQueryListEvent) => {
    callback(e.matches ? 'dark' : 'light');
  };
  
  mediaQuery.addEventListener('change', handleChange);
  
  return () => {
    mediaQuery.removeEventListener('change', handleChange);
  };
}

/**
 * 获取主题的显示名称（用于UI显示）
 * @param theme 主题
 * @returns 显示名称
 */
export function getThemeDisplayName(theme: Theme): string {
  switch (theme) {
    case 'light':
      return '浅色主题';
    case 'dark':
      return '深色主题';
    case 'system':
      return '跟随系统';
    default:
      return '未知主题';
  }
}

/**
 * 获取下一个主题（用于循环切换）
 * @param currentTheme 当前主题
 * @returns 下一个主题
 */
export function getNextTheme(currentTheme: Theme): Theme {
  const themes: Theme[] = ['light', 'dark', 'system'];
  const currentIndex = themes.indexOf(currentTheme);
  return themes[(currentIndex + 1) % themes.length];
}
