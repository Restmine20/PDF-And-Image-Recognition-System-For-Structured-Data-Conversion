import { request } from './client';
import type { AuthResponse, User } from '@/types';

export const authApi = {
  register(email: string, password: string): Promise<AuthResponse> {
    return request<AuthResponse>('/api/auth/register', {
      method: 'POST',
      body: { email, password },
      anonymous: true,
    });
  },
  login(email: string, password: string): Promise<AuthResponse> {
    return request<AuthResponse>('/api/auth/login', {
      method: 'POST',
      body: { email, password },
      anonymous: true,
    });
  },
  logout(): Promise<void> {
    return request<void>('/api/auth/logout', { method: 'POST' });
  },
  me(): Promise<{ user: User }> {
    return request<{ user: User }>('/api/auth/me');
  },
};
