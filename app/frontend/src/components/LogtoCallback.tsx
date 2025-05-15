import { useHandleSignInCallback } from '@logto/react';
import { useEffect } from 'react';

/**
 * Logto回调组件
 * 处理Logto认证后的回调
 */
export function LogtoCallback() {
  console.log('LogtoCallback: Component Instantiated/Re-rendered (Top Level)');

  // isAuthProcessed (renamed from isAuthenticated) indicates if the callback was processed by the SDK.
  // The actual isAuthenticated state for the app should be sourced from useLogto() in other components.
  const { isLoading, error, isAuthenticated: isAuthProcessed } = useHandleSignInCallback(() => {
    // This callback is executed after Logto SDK finishes processing the sign-in.
    // It's generally not recommended to perform immediate redirection here,
    // as the SDK has already updated the authentication state.
    // Navigation should be handled based on the updated global isAuthenticated state
    // in your main application router or relevant components.
    console.log('LogtoCallback: ---- Logto SDK has finished processing sign-in callback. isAuthProcessed: ----', isAuthProcessed);
  });

  useEffect(() => {
    console.log('LogtoCallback: useEffect triggered.');
    console.log('LogtoCallback: isLoading from useEffect:', isLoading);
    console.log('LogtoCallback: error from useEffect:', error);
    console.log('LogtoCallback: isAuthProcessed from useEffect:', isAuthProcessed);

    if (error) {
      console.error('LogtoCallback: Detailed error object from useEffect:', JSON.stringify(error, null, 2));
      // Optionally, redirect to an error page or show an error message.
      // For now, the component's render method will handle showing the error.
    } else if (!isLoading && isAuthProcessed) {
      // If no longer loading, no error, and the callback has been processed by the SDK
      // (isAuthProcessed is true), then it's safe to redirect to the main application.
      // The global isAuthenticated state (from useLogto()) should now be true in other parts of the app.
      console.log('LogtoCallback: Redirecting to home page as auth process is complete.');
      window.location.href = '/';
    }
  }, [isLoading, error, isAuthProcessed]); // Dependencies

  console.log('LogtoCallback: State before rendering UI - isLoading:', isLoading, 'error:', error, 'isAuthProcessed (from hook):', isAuthProcessed);

  if (isLoading) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center">
        {/* Simplified loading UI */}
        <p className="text-lg font-medium text-gray-700">LogtoCallback: Processing authentication...</p>
        <div className="mt-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-500 border-t-transparent"></div>
        <p className="mt-2 text-sm text-gray-500">Please wait while we securely sign you in.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-red-50 p-4">
        <div className="w-full max-w-md rounded-lg bg-white p-6 text-center shadow-xl">
          <h2 className="mb-3 text-2xl font-semibold text-red-700">LogtoCallback: Authentication Error</h2>
          <p className="mb-4 text-sm text-gray-600">
            An error occurred during the authentication process. Please try signing in again.
          </p>
          <pre className="mb-4 whitespace-pre-wrap rounded-md bg-red-100 p-3 text-left text-xs text-red-800">
            {error.message || (typeof error === 'object' ? JSON.stringify(error, null, 2) : String(error))}
          </pre>
          <button
            onClick={() => window.location.href = '/'} // Redirect to home or login page
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Go to Homepage and Try Again
          </button>
        </div>
      </div>
    );
  }

  // This state should ideally not be reached if redirection in useEffect works correctly
  // when !isLoading && !error && isAuthProcessed.
  // If it is reached, it means the user is waiting for the redirect in useEffect.
  return (
    <div className="flex min-h-screen flex-col items-center justify-center">
      <p className="text-lg font-medium text-gray-700">LogtoCallback: Finalizing sign-in...</p>
      <p className="mt-2 text-sm text-gray-500">You will be redirected shortly.</p>
      {/* You might want to add a small spinner here too if this state is visible for a noticeable time */}
    </div>
  );
}
