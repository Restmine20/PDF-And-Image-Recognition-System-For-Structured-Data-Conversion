import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import { authApi } from '@/api/auth';
import { tokenStorage } from '@/api/client';
import { ApiError, type User } from '@/types';

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  /** true пока проверяем сохранённую сессию на старте. */
  isInitializing: boolean;
  login(email: string, password: string): Promise<void>;
  register(email: string, password: string): Promise<void>;
  logout(): Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isInitializing, setIsInitializing] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    const token = tokenStorage.get();
    if (!token) {
      setIsInitializing(false);
      return;
    }
    authApi
      .me()
      .then(({ user: fetched }) => {
        if (!cancelled) setUser(fetched);
      })
      .catch((err: unknown) => {
        // 401 — токен невалиден, чистим. Сетевые ошибки игнорируем,
        // позволим пользователю работать как гость.
        if (err instanceof ApiError && err.status === 401) {
          tokenStorage.clear();
        }
      })
      .finally(() => {
        if (!cancelled) setIsInitializing(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const { token, user: loggedIn } = await authApi.login(email, password);
    tokenStorage.set(token);
    setUser(loggedIn);
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const { token, user: registered } = await authApi.register(email, password);
    tokenStorage.set(token);
    setUser(registered);
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      /* даже если сервер недоступен, локально разлогиниваем */
    }
    tokenStorage.clear();
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: user !== null,
      isInitializing,
      login,
      register,
      logout,
    }),
    [user, isInitializing, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within <AuthProvider>');
  }
  return ctx;
}
