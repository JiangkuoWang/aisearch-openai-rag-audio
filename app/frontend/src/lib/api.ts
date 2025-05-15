/**
 * HTTP客户端封装
 * 自动向请求中添加认证令牌
 */
// import { getToken } from './auth';

// 主应用API地址
const API_BASE_URL = 'http://localhost:8765';

/**
 * 封装的fetch函数，自动添加认证头
 */
export async function fetchWithAuth(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  // 获取认证令牌
  // const token = getToken();
  
  // 准备请求头
  const headers = new Headers(options.headers || {});
  
  // 如果有令牌，添加到认证头
  // if (token) {
  //   headers.set('Authorization', `Bearer ${token}`);
  // }
  
  // 确保请求包含完整的URL
  const fullUrl = url.startsWith('http') ? url : `${API_BASE_URL}${url}`;
  
  // 发送请求
  return fetch(fullUrl, {
    ...options,
    headers,
  });
}

/**
 * GET请求封装
 */
export async function get<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetchWithAuth(url, {
    ...options,
    method: 'GET',
  });
  
  if (!response.ok) {
    throw new Error(`请求失败: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * POST请求封装
 */
export async function post<T>(
  url: string,
  data: any,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers || {});
  
  // 如果没有指定Content-Type且不是FormData，默认使用JSON
  if (!headers.has('Content-Type') && !(data instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  
  const response = await fetchWithAuth(url, {
    ...options,
    method: 'POST',
    headers,
    body: data instanceof FormData ? data : JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error(`请求失败: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * 上传文件封装
 */
export async function uploadFiles(
  url: string,
  files: File[],
  options: RequestInit = {}
): Promise<any> {
  const formData = new FormData();
  
  // 添加所有文件到FormData
  files.forEach((file) => {
    formData.append('file', file);
  });
  
  return post(url, formData, options);
}

/**
 * 设置RAG提供程序类型
 */
export async function setRagProviderType(providerType: string): Promise<any> {
  return post('/rag-config', { provider_type: providerType });
}

/**
 * 获取当前用户认证状态
 */
// export async function getAuthStatus(): Promise<any> {
//   return get('/auth-status');
// } 