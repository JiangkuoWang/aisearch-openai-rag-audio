// 重新导出ThemeContext中的useTheme hook
// 这样可以保持hooks目录的一致性，同时避免重复实现

export { useTheme, type Theme } from '@/contexts/ThemeContext';
