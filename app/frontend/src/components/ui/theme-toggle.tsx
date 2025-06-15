import * as React from "react";
import { Moon, Sun, Monitor } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

// 主题切换按钮的属性接口
interface ThemeToggleProps {
  className?: string;
  variant?: "default" | "outline" | "ghost" | "secondary";
  size?: "default" | "sm" | "lg" | "icon";
  showLabel?: boolean;
  showTooltip?: boolean;
}

// 主题图标映射
const themeIcons = {
  light: Sun,
  dark: Moon,
  system: Monitor,
} as const;

// 主题标签映射
const themeLabels = {
  light: "浅色主题",
  dark: "深色主题", 
  system: "跟随系统",
} as const;

/**
 * 主题切换按钮组件
 * 支持在浅色、深色和系统主题之间循环切换
 */
export function ThemeToggle({
  className,
  variant = "ghost",
  size = "icon",
  showLabel = false,
  showTooltip = true,
}: ThemeToggleProps) {
  const { theme, setTheme, actualTheme } = useTheme();

  // 获取下一个主题
  const getNextTheme = () => {
    const themes = ['light', 'dark', 'system'] as const;
    const currentIndex = themes.indexOf(theme);
    return themes[(currentIndex + 1) % themes.length];
  };

  // 处理主题切换
  const handleToggle = () => {
    const nextTheme = getNextTheme();
    setTheme(nextTheme);
  };

  // 获取当前主题图标
  const CurrentIcon = themeIcons[theme];
  
  // 获取当前主题标签
  const currentLabel = themeLabels[theme];

  // 构建按钮内容
  const buttonContent = (
    <>
      <CurrentIcon className={cn("h-4 w-4", showLabel && "mr-2")} />
      {showLabel && (
        <span className="text-sm font-medium">{currentLabel}</span>
      )}
      <span className="sr-only">切换主题</span>
    </>
  );

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleToggle}
      className={cn(
        "transition-all duration-200",
        // 根据当前实际主题调整样式
        actualTheme === 'dark' 
          ? "text-yellow-400 hover:text-yellow-300" 
          : "text-gray-600 hover:text-gray-900",
        className
      )}
      title={showTooltip ? `当前: ${currentLabel}，点击切换` : undefined}
      aria-label={`当前主题: ${currentLabel}，点击切换到下一个主题`}
    >
      {buttonContent}
    </Button>
  );
}

/**
 * 主题切换下拉菜单组件
 * 提供更详细的主题选择界面
 */
export function ThemeToggleDropdown({
  className,
}: {
  className?: string;
}) {
  const { theme, setTheme, actualTheme } = useTheme();
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // 点击外部关闭下拉菜单
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const themes = [
    { value: 'light' as const, label: '浅色主题', icon: Sun },
    { value: 'dark' as const, label: '深色主题', icon: Moon },
    { value: 'system' as const, label: '跟随系统', icon: Monitor },
  ];

  const CurrentIcon = themeIcons[theme];

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "transition-all duration-200",
          actualTheme === 'dark' 
            ? "text-yellow-400 hover:text-yellow-300" 
            : "text-gray-600 hover:text-gray-900"
        )}
        aria-label="选择主题"
      >
        <CurrentIcon className="h-4 w-4" />
        <span className="sr-only">选择主题</span>
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-background border border-border rounded-md shadow-lg py-1 z-50">
          {themes.map((themeOption) => {
            const Icon = themeOption.icon;
            const isSelected = theme === themeOption.value;
            
            return (
              <button
                key={themeOption.value}
                onClick={() => {
                  setTheme(themeOption.value);
                  setIsOpen(false);
                }}
                className={cn(
                  "flex items-center w-full px-3 py-2 text-sm transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  isSelected && "bg-accent text-accent-foreground font-medium"
                )}
              >
                <Icon className="mr-3 h-4 w-4" />
                <span>{themeOption.label}</span>
                {isSelected && (
                  <span className="ml-auto text-xs text-muted-foreground">
                    当前
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
