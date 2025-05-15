# Logto安装指南

要将Logto集成到您的前端项目中，请按照以下步骤操作：

## 1. 安装必要的依赖

在项目根目录下运行以下命令安装Logto SDK：

```bash
npm install @logto/react
```

## 2. 配置Logto

1. 在Logto管理控制台中创建一个应用
2. 获取应用ID和Logto端点URL
3. 在`app/frontend/src/logto-config.ts`文件中更新配置：

```typescript
export const logtoConfig: LogtoConfig = {
  endpoint: 'YOUR_LOGTO_ENDPOINT', // 例如: https://your-logto-instance.logto.app
  appId: 'YOUR_APPLICATION_ID',
  // ...其他配置
};
```

## 3. 配置重定向URI

在Logto管理控制台中，为您的应用添加以下重定向URI：

- 回调URI: `http://localhost:8765/callback`
- 登出后重定向URI: `http://localhost:8765`

## 4. 构建并启动应用

```bash
npm run build
```

## 5. 测试Logto登录

1. 启动应用
2. 点击"Logto 登录"按钮
3. 完成Logto认证流程
4. 验证是否成功重定向回应用并显示用户信息

## 注意事项

- 确保后端服务器正确配置，能够处理前端路由
- 如果遇到CORS问题，请检查Logto控制台中的CORS设置
- 对于生产环境，请使用HTTPS协议
