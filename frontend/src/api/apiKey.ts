import { request } from './client';
import type { ApiKeyInfo } from '@/types';

export const apiKeyApi = {
  generate(): Promise<ApiKeyInfo> {
    return request<ApiKeyInfo>('/api/api-key', { method: 'POST' });
  },
};
