import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, getStoredUser, isAuthenticated, login as authLogin, logout as authLogout, register as authRegister, LoginRequest, RegisterRequest, getCurrentUser } from '../lib/auth';

// 认证上下文类型
interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  isAuthenticated: boolean;
  login: (data: LoginRequest) => Promise<void>;
  logout: () => void;
  register: (data: RegisterRequest) => Promise<void>;
  clearError: () => void;
}

// 创建认证上下文
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// AuthContext Provider组件
interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(getStoredUser());
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // 检查用户认证状态
  useEffect(() => {
    const checkAuth = async () => {
      if (isAuthenticated() && !user) {
        try {
          const currentUser = await getCurrentUser();
          setUser(currentUser);
        } catch (error) {
          console.error('获取用户信息失败:', error);
        }
      }
    };
    
    checkAuth();
  }, []);
  
  // 登录方法
  const login = async (data: LoginRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const loggedInUser = await authLogin(data);
      setUser(loggedInUser);
    } catch (error) {
      setError((error as Error).message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  // 注册方法
  const register = async (data: RegisterRequest) => {
    setIsLoading(true);
    setError(null);
    
    try {
      await authRegister(data);
      // 注册成功后自动登录
      await login({ username: data.username, password: data.password });
    } catch (error) {
      setError((error as Error).message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };
  
  // 登出方法
  const logout = () => {
    authLogout();
    setUser(null);
  };
  
  // 清除错误
  const clearError = () => {
    setError(null);
  };
  
  // 提供上下文值
  const contextValue: AuthContextType = {
    user,
    isLoading,
    error,
    isAuthenticated: !!user,
    login,
    logout,
    register,
    clearError,
  };
  
  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

// 自定义钩子，方便使用认证上下文
export function useAuth() {
  const context = useContext(AuthContext);
  
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
} 