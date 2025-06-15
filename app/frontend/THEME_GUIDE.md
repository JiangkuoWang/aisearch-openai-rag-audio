# 主题切换功能使用指南

## 📖 概述

本应用已集成完整的主题切换功能，支持浅色主题、深色主题和跟随系统主题三种模式。主题系统基于 shadcn/ui 设计系统和 Tailwind CSS 构建，提供了一致的用户体验和开发体验。

## 🎨 功能特性

### ✨ 核心功能
- **三种主题模式**：浅色、深色、跟随系统
- **持久化存储**：用户偏好自动保存到 localStorage
- **系统主题检测**：自动检测并响应系统主题变化
- **平滑过渡动画**：主题切换时的优雅过渡效果
- **完整组件适配**：所有UI组件都支持主题切换

### 🎯 用户界面
- **主题切换按钮**：位于顶部导航栏，支持循环切换
- **主题下拉菜单**：提供详细的主题选择界面（可选）
- **视觉反馈**：当前主题的图标和状态显示

## 🚀 使用方法

### 用户操作
1. **快速切换**：点击顶部导航栏的主题切换按钮
2. **循环模式**：浅色 → 深色 → 跟随系统 → 浅色...
3. **自动保存**：选择的主题偏好会自动保存

### 开发者集成

#### 1. 基础使用
```tsx
import { useTheme } from '@/hooks/useTheme';

function MyComponent() {
  const { theme, setTheme, actualTheme } = useTheme();
  
  return (
    <div className="bg-background text-foreground">
      <p>当前主题: {theme}</p>
      <p>实际应用: {actualTheme}</p>
    </div>
  );
}
```

#### 2. 主题切换组件
```tsx
import { ThemeToggle } from '@/components/ui/theme-toggle';

function Header() {
  return (
    <header>
      <ThemeToggle />
    </header>
  );
}
```

#### 3. 条件样式
```tsx
function ConditionalStyling() {
  const { actualTheme } = useTheme();
  
  return (
    <div className={`
      p-4 rounded-lg
      ${actualTheme === 'dark' ? 'shadow-lg' : 'shadow-md'}
    `}>
      内容
    </div>
  );
}
```

## 🎨 样式系统

### CSS 变量
主题系统使用以下 CSS 变量：

```css
:root {
  --background: 0 0% 100%;
  --foreground: 0 0% 3.9%;
  --primary: 0 0% 9%;
  --primary-foreground: 0 0% 98%;
  --secondary: 0 0% 96.1%;
  --secondary-foreground: 0 0% 9%;
  --muted: 0 0% 96.1%;
  --muted-foreground: 0 0% 45.1%;
  --accent: 0 0% 96.1%;
  --accent-foreground: 0 0% 9%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 0 0% 98%;
  --border: 0 0% 89.8%;
  --input: 0 0% 89.8%;
  --ring: 0 0% 3.9%;
}

.dark {
  --background: 0 0% 3.9%;
  --foreground: 0 0% 98%;
  /* ... 其他深色主题变量 */
}
```

### Tailwind 类名
推荐使用语义化的 Tailwind 类名：

```tsx
// ✅ 推荐：使用语义化类名
<div className="bg-background text-foreground border-border">

// ❌ 避免：硬编码颜色
<div className="bg-white text-black border-gray-200">
```

### 常用类名映射

| 用途 | 浅色主题类名 | 语义化类名 |
|------|-------------|-----------|
| 背景色 | `bg-white` | `bg-background` |
| 文字色 | `text-gray-900` | `text-foreground` |
| 边框色 | `border-gray-200` | `border-border` |
| 次要文字 | `text-gray-500` | `text-muted-foreground` |
| 卡片背景 | `bg-white` | `bg-card` |
| 按钮背景 | `bg-blue-600` | `bg-primary` |

## 🔧 自定义主题

### 修改颜色变量
在 `src/index.css` 中修改 CSS 变量：

```css
:root {
  --primary: 220 100% 50%; /* 自定义主色调 */
}

.dark {
  --primary: 220 100% 60%; /* 深色主题的主色调 */
}
```

### 添加新的主题变量
```css
:root {
  --custom-color: 180 100% 50%;
}

.dark {
  --custom-color: 180 100% 60%;
}
```

在 `tailwind.config.js` 中注册：
```js
module.exports = {
  theme: {
    extend: {
      colors: {
        custom: "hsl(var(--custom-color))",
      }
    }
  }
}
```

## 🧪 测试功能

### 运行主题测试
```typescript
import { runAllThemeTests } from '@/lib/theme-test';

// 在浏览器控制台中运行
runAllThemeTests();
```

### 手动测试清单
- [ ] 主题切换按钮正常工作
- [ ] 三种主题模式都能正确显示
- [ ] 主题偏好能够持久化保存
- [ ] 系统主题变化时能自动响应
- [ ] 所有组件在不同主题下显示正常
- [ ] 过渡动画流畅自然

## 🐛 故障排除

### 常见问题

1. **主题切换不生效**
   - 检查是否正确包装了 `ThemeProvider`
   - 确认 CSS 变量是否正确定义

2. **某些组件没有适配主题**
   - 检查是否使用了硬编码的颜色类名
   - 替换为语义化的主题类名

3. **过渡动画卡顿**
   - 检查是否有大量元素同时进行过渡
   - 考虑为特定元素添加 `no-transition` 类

4. **localStorage 错误**
   - 检查浏览器是否支持 localStorage
   - 确认没有隐私模式限制

### 调试工具
```typescript
// 检查当前主题状态
console.log('当前主题:', document.documentElement.className);
console.log('存储的主题:', localStorage.getItem('voice-rag-theme'));

// 检查CSS变量
const style = getComputedStyle(document.documentElement);
console.log('背景色变量:', style.getPropertyValue('--background'));
```

## 📚 技术架构

### 文件结构
```
src/
├── contexts/
│   └── ThemeContext.tsx      # 主题上下文和Provider
├── hooks/
│   └── useTheme.ts          # 主题Hook
├── components/ui/
│   └── theme-toggle.tsx     # 主题切换组件
├── lib/
│   ├── theme.ts            # 主题工具函数
│   └── theme-test.ts       # 测试工具
└── index.css               # 主题CSS变量
```

### 依赖关系
- **React Context API**：全局状态管理
- **Tailwind CSS**：样式系统
- **shadcn/ui**：组件库
- **Lucide React**：图标库

## 🔄 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的主题切换功能
- ✅ 三种主题模式支持
- ✅ 持久化存储
- ✅ 系统主题检测
- ✅ 平滑过渡动画
- ✅ 全组件主题适配
- ✅ 测试工具和文档

---

如有问题或建议，请联系开发团队。
