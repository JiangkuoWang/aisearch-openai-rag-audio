import { useState, useRef, useEffect } from 'react';
import { useLogto } from '@logto/react';

// 不需要任何props
export function LogtoUserMenu() {
  const { signIn, signOut, isAuthenticated, getIdTokenClaims } = useLogto();
  const [user, setUser] = useState<any>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // 获取用户信息
  useEffect(() => {
    const fetchUserInfo = async () => {
      if (isAuthenticated) {
        const claims = await getIdTokenClaims();
        setUser(claims);
      }
    };

    fetchUserInfo();
  }, [isAuthenticated, getIdTokenClaims]);

  // 点击外部关闭菜单
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // 处理登录
  const handleLogin = () => {
    signIn('http://localhost:8765/callback');
  };

  if (!isAuthenticated) {
    return (
      <button
        onClick={handleLogin}
        className="flex items-center px-4 py-2 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-md transition"
      >
        Sign in/Sign up
      </button>
    );
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        className="flex items-center px-3 py-2 text-sm text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition"
      >
        <span className="mr-2">{user?.name || user?.sub}</span>
        <svg
          className={`w-5 h-5 transform ${isMenuOpen ? 'rotate-180' : ''} transition-transform`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isMenuOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
          <div className="px-4 py-2 border-b">
            <p className="text-sm font-medium">{user?.name || user?.sub}</p>
            <p className="text-xs text-gray-500 truncate">{user?.email}</p>
          </div>

          <button
            onClick={() => {
              // Construct the post-logout redirect URI to be the root of the current application.
              // This ensures that after logging out from Logto, the user is redirected back to your app's home page.
              const postLogoutRedirectUri = window.location.origin + '/'; // e.g., 'http://localhost:8765/'
              signOut(postLogoutRedirectUri);
              setIsMenuOpen(false);
            }}
            className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-gray-100"
          >
            退出登录
          </button>
        </div>
      )}
    </div>
  );
}
