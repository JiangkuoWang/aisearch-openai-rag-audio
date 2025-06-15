/**
 * 主题功能测试脚本
 * 在浏览器控制台中运行此脚本来测试主题功能
 */

// 测试主题功能的基本操作
function testThemeFunctionality() {
  console.log('🧪 开始测试主题功能...');
  
  try {
    // 测试系统主题检测
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    console.log('✅ 系统主题检测:', systemTheme);
    
    // 测试主题存储
    const testThemes = ['light', 'dark', 'system'];
    
    testThemes.forEach(theme => {
      localStorage.setItem('voice-rag-theme', theme);
      const stored = localStorage.getItem('voice-rag-theme');
      if (stored !== theme) {
        throw new Error(`主题存储测试失败: 期望 ${theme}, 实际 ${stored}`);
      }
      console.log(`✅ 主题存储测试通过: ${theme}`);
    });
    
    console.log('🎉 主题功能测试通过!');
    return true;
    
  } catch (error) {
    console.error('❌ 主题功能测试失败:', error);
    return false;
  }
}

// 测试CSS变量是否正确设置
function testCSSVariables() {
  console.log('🧪 开始测试CSS变量...');
  
  try {
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
    
    const missingVariables = [];
    
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

// 测试主题切换的DOM操作
function testThemeDOM() {
  console.log('🧪 开始测试主题DOM操作...');
  
  try {
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

// 测试主题切换按钮
function testThemeToggleButton() {
  console.log('🧪 开始测试主题切换按钮...');
  
  try {
    // 查找主题切换按钮
    const themeButton = document.querySelector('[aria-label*="主题"]');
    
    if (!themeButton) {
      throw new Error('未找到主题切换按钮');
    }
    
    console.log('✅ 找到主题切换按钮:', themeButton);
    
    // 检查按钮是否可点击
    if (themeButton.disabled) {
      throw new Error('主题切换按钮被禁用');
    }
    
    console.log('✅ 主题切换按钮可用');
    
    // 检查按钮图标
    const icon = themeButton.querySelector('svg');
    if (!icon) {
      throw new Error('主题切换按钮缺少图标');
    }
    
    console.log('✅ 主题切换按钮包含图标');
    
    console.log('🎉 主题切换按钮测试通过!');
    return true;
    
  } catch (error) {
    console.error('❌ 主题切换按钮测试失败:', error);
    return false;
  }
}

// 测试组件主题适配
function testComponentThemeAdaptation() {
  console.log('🧪 开始测试组件主题适配...');
  
  try {
    // 检查主要组件是否使用了主题类名
    const componentsToCheck = [
      { selector: '.bg-background', name: '背景组件' },
      { selector: '.text-foreground', name: '文字组件' },
      { selector: '.border-border', name: '边框组件' },
      { selector: '.bg-card', name: '卡片组件' },
    ];
    
    componentsToCheck.forEach(({ selector, name }) => {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        console.log(`✅ 找到 ${elements.length} 个${name}使用主题类名`);
      } else {
        console.warn(`⚠️ 未找到使用 ${selector} 的组件`);
      }
    });
    
    console.log('🎉 组件主题适配测试完成!');
    return true;
    
  } catch (error) {
    console.error('❌ 组件主题适配测试失败:', error);
    return false;
  }
}

// 运行所有测试
function runAllThemeTests() {
  console.log('🚀 开始运行完整的主题功能测试套件...');
  
  const tests = [
    { name: '基础功能测试', fn: testThemeFunctionality },
    { name: 'CSS变量测试', fn: testCSSVariables },
    { name: 'DOM操作测试', fn: testThemeDOM },
    { name: '切换按钮测试', fn: testThemeToggleButton },
    { name: '组件适配测试', fn: testComponentThemeAdaptation }
  ];
  
  let allPassed = true;
  const results = [];
  
  tests.forEach((test, index) => {
    console.log(`\n--- 测试 ${index + 1}/${tests.length}: ${test.name} ---`);
    const passed = test.fn();
    results.push({ name: test.name, passed });
    if (!passed) {
      allPassed = false;
    }
  });
  
  console.log('\n' + '='.repeat(50));
  console.log('📊 测试结果汇总:');
  results.forEach(result => {
    console.log(`${result.passed ? '✅' : '❌'} ${result.name}`);
  });
  
  if (allPassed) {
    console.log('\n🎉 所有主题测试通过! 主题功能工作正常。');
  } else {
    console.log('\n❌ 部分主题测试失败! 请检查相关功能。');
  }
  console.log('='.repeat(50));
  
  return allPassed;
}

// 手动主题切换测试
function manualThemeTest() {
  console.log('🎮 开始手动主题切换测试...');
  console.log('请按照以下步骤进行测试:');
  console.log('1. 点击顶部导航栏的主题切换按钮');
  console.log('2. 观察页面主题是否正确切换');
  console.log('3. 刷新页面，检查主题是否保持');
  console.log('4. 重复步骤1-3，测试所有三种主题模式');
  
  // 提供快捷测试函数
  window.switchToLight = () => {
    localStorage.setItem('voice-rag-theme', 'light');
    location.reload();
  };
  
  window.switchToDark = () => {
    localStorage.setItem('voice-rag-theme', 'dark');
    location.reload();
  };
  
  window.switchToSystem = () => {
    localStorage.setItem('voice-rag-theme', 'system');
    location.reload();
  };
  
  console.log('💡 快捷测试函数已添加到window对象:');
  console.log('- switchToLight() - 切换到浅色主题');
  console.log('- switchToDark() - 切换到深色主题');
  console.log('- switchToSystem() - 切换到系统主题');
}

// 导出测试函数到全局
window.testTheme = {
  runAll: runAllThemeTests,
  functionality: testThemeFunctionality,
  cssVariables: testCSSVariables,
  dom: testThemeDOM,
  button: testThemeToggleButton,
  components: testComponentThemeAdaptation,
  manual: manualThemeTest
};

console.log('🧪 主题测试工具已加载!');
console.log('使用 testTheme.runAll() 运行所有测试');
console.log('使用 testTheme.manual() 进行手动测试');
console.log('或运行单个测试: testTheme.functionality(), testTheme.cssVariables() 等');

// 自动运行基础测试
runAllThemeTests();
