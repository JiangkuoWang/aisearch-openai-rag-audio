/**
 * Logto配置文件
 * 包含Logto SDK的配置信息
 */
import { LogtoConfig, UserScope } from '@logto/react';

// Logto配置
export const logtoConfig: LogtoConfig = {
  endpoint: 'https://ousbav.logto.app/', // 例如: https://your-logto-instance.logto.app
  appId: '406h22fnkxkpz793c722l',
  scopes: [
    UserScope.Email,
  ],
  // 如果需要访问API资源，可以在这里添加
  // resources: ['https://your-api.com/api'],
};
