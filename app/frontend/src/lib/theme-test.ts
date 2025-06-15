/**
 * 主题功能测试工具
 * 用于验证主题切换功能的正确性
 */

import { getSystemTheme, getStoredTheme, setStoredTheme, resolveTheme } from './theme';
import type { Theme } from '@/contexts/ThemeContext';

/**
 * 测试主题功能的基本操作
 */
export function testThemeFunctionality(): boolean {
  console.log('🧪 开始测试主题功能...');
  
  try {
    // 测试系统主题检测
    const systemTheme = getSystemTheme();
    console.log('✅ 系统主题检测:', systemTheme);
    
    // 测试主题存储
    const testThemes: Theme[] = ['light', 'dark', 'system'];
    
    testThemes.forEach(theme => {
      setStoredTheme(theme);
      const stored = getStoredTheme();
      if (stored !== theme) {
        throw new Error(`主题存储测试失败: 期望 ${theme}, 实际 ${stored}`);
      }
      console.log(`✅ 主题存储测试通过: ${theme}`);
    });
    
    // 测试主题解析
    testThemes.forEach(theme => {
      const resolved = resolveTheme(theme);
      if (theme === 'system') {
        if (resolved !== systemTheme) {
          throw new Error(`系统主题解析失败: 期望 ${systemTheme}, 实际 ${resolved}`);
        }
      } else {
        if (resolved !== theme) {
          throw new Error(`主题解析失败: 期望 ${theme}, 实际 ${resolved}`);
        }
      }
      console.log(`✅ 主题解析测试通过: ${theme} -> ${resolved}`);
    });
    
    console.log('🎉 所有主题功能测试通过!');
    return true;
    
  } catch (error) {
    console.error('❌ 主题功能测试失败:', error);
    return false;
  }
}

/**
 * 测试CSS变量是否正确设置
 */
export function testCSSVariables(): boolean {
  console.log('🧪 开始测试CSS变量...');
  
  try {
    if (typeof window === 'undefined') {
      console.log('⚠️ 非浏览器环境，跳过CSS变量测试');
      return true;
    }
    
    const root = document.documentElement;
    const computedStyle = getComputedStyle(root);
    
    // 测试关键CSS变量
    const requiredVariables = [
      '--background',
      '--foreground',
      '--primary',
      '--primary-foreground',
      '--secondary',
      '--secondary-foreground',
      '--muted',
      '--muted-foreground',
      '--accent',
      '--accent-foreground',
      '--destructive',
      '--destructive-foreground',
      '--border',
      '--input',
      '--ring'
    ];
    
    const missingVariables: string[] = [];
    
    requiredVariables.forEach(variable => {
      const value = computedStyle.getPropertyValue(variable).trim();
      if (!value) {
        missingVariables.push(variable);
      } else {
        console.log(`✅ CSS变量存在: ${variable} = ${value}`);
      }
    });
    
    if (missingVariables.length > 0) {
      throw new Error(`缺少CSS变量: ${missingVariables.join(', ')}`);
    }
    
    console.log('🎉 所有CSS变量测试通过!');
    return true;
    
  } catch (error) {
    console.error('❌ CSS变量测试失败:', error);
    return false;
  }
}

/**
 * 测试主题切换的DOM操作
 */
export function testThemeDOM(): boolean {
  console.log('🧪 开始测试主题DOM操作...');
  
  try {
    if (typeof window === 'undefined') {
      console.log('⚠️ 非浏览器环境，跳过DOM测试');
      return true;
    }
    
    const root = document.documentElement;
    
    // 测试浅色主题
    root.classList.remove('light', 'dark');
    root.classList.add('light');
    root.setAttribute('data-theme', 'light');
    
    if (!root.classList.contains('light') || root.getAttribute('data-theme') !== 'light') {
      throw new Error('浅色主题DOM设置失败');
    }
    console.log('✅ 浅色主题DOM设置成功');
    
    // 测试深色主题
    root.classList.remove('light', 'dark');
    root.classList.add('dark');
    root.setAttribute('data-theme', 'dark');
    
    if (!root.classList.contains('dark') || root.getAttribute('data-theme') !== 'dark') {
      throw new Error('深色主题DOM设置失败');
    }
    console.log('✅ 深色主题DOM设置成功');
    
    console.log('🎉 所有DOM操作测试通过!');
    return true;
    
  } catch (error) {
    console.error('❌ DOM操作测试失败:', error);
    return false;
  }
}

/**
 * 运行所有主题测试
 */
export function runAllThemeTests(): boolean {
  console.log('🚀 开始运行完整的主题功能测试套件...');
  
  const tests = [
    testThemeFunctionality,
    testCSSVariables,
    testThemeDOM
  ];
  
  let allPassed = true;
  
  tests.forEach((test, index) => {
    console.log(`\n--- 测试 ${index + 1}/${tests.length} ---`);
    const passed = test();
    if (!passed) {
      allPassed = false;
    }
  });
  
  console.log('\n' + '='.repeat(50));
  if (allPassed) {
    console.log('🎉 所有主题测试通过! 主题功能工作正常。');
  } else {
    console.log('❌ 部分主题测试失败! 请检查相关功能。');
  }
  console.log('='.repeat(50));
  
  return allPassed;
}
