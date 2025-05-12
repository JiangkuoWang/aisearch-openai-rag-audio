/**
 * 认证服务
 * 处理用户登录、注册和令牌管理
 */

// 认证服务器地址（独立的认证服务）
const AUTH_API_URL = 'http://localhost:8765/auth';
// 主应用API地址
const MAIN_API_URL = 'http://localhost:8765';

// 本地存储键名
const TOKEN_KEY = 'voice_rag_auth_token';
const USER_KEY = 'voice_rag_user';

// 用户类型定义
export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
}

// 登录请求类型
export interface LoginRequest {
  username: string;
  password: string;
}

// 注册请求类型
export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

// 认证响应类型
export interface AuthResponse {
  access_token: string;
  token_type: string;
}

/**
 * 注册新用户
 */
export async function register(data: RegisterRequest): Promise<User> {
  // 确保使用正确的注册端点 /auth/register
  const response = await fetch(`${AUTH_API_URL}/register`, {
    method: 'POST', // 确保方法是 POST
    headers: {
      'Content-Type': 'application/json', // 确保 Content-Type 是 application/json
    },
    // 确保请求体是包含 username, email, password 的 JSON 字符串
    body: JSON.stringify({
        username: data.username,
        email: data.email,
        password: data.password
    }),
  });

  if (!response.ok) {
    // 保留错误处理逻辑
    const errorData = await response.json().catch(() => ({ detail: '注册请求失败' }));
    throw new Error(errorData.detail || '注册失败');
  }

  // 保留成功处理逻辑
  const user = await response.json();
  return user;
}

/**
 * 用户登录
 */
export async function login(data: LoginRequest): Promise<User> {
  const formData = new URLSearchParams();
  formData.append('username', data.username);
  formData.append('password', data.password);

  const response = await fetch(`${AUTH_API_URL}/login/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: '登录失败' }));
    throw new Error(errorData.detail || '登录失败');
  }

  const responseData = await response.json();
  
  localStorage.setItem(TOKEN_KEY, responseData.access_token);
  
  if (responseData.user) {
    localStorage.setItem(USER_KEY, JSON.stringify(responseData.user));
    return responseData.user as User;
  } else {
    const user = await getCurrentUser();
    if (!user) {
      logout();
      throw new Error('获取用户信息失败');
    }
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    return user;
  }
}

/**
 * 获取当前登录的用户信息
 */
export async function getCurrentUser(): Promise<User | null> {
  const token = getToken();
  if (!token) {
    return null;
  }

  try {
    const response = await fetch(`${AUTH_API_URL}/users/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      // 如果获取用户信息失败，清除本地存储
      logout();
      return null;
    }

    const user = await response.json();
    return user;
  } catch (error) {
    console.error('获取用户信息出错:', error);
    return null;
  }
}

/**
 * 从主应用获取认证状态
 */
export async function checkAuthStatus(): Promise<{ authenticated: boolean; user?: User; auth_url?: string }> {
  try {
    const response = await fetch(`${MAIN_API_URL}/auth-status`);
    if (!response.ok) {
      return { authenticated: false };
    }
    return await response.json();
  } catch (error) {
    console.error('检查认证状态出错:', error);
    return { authenticated: false };
  }
}

/**
 * 用户登出
 */
export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  
  // 可以在这里添加重定向到登录页或其他逻辑
  window.location.href = '/';
}

/**
 * 获取保存的认证令牌
 */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

/**
 * 获取本地存储的用户信息
 */
export function getStoredUser(): User | null {
  const userJson = localStorage.getItem(USER_KEY);
  if (!userJson) {
    return null;
  }
  
  try {
    return JSON.parse(userJson);
  } catch (error) {
    console.error('解析用户信息出错:', error);
    return null;
  }
}

/**
 * 检查用户是否已登录
 */
export function isAuthenticated(): boolean {
  return !!getToken();
} 