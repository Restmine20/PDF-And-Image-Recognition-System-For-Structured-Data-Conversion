import { request, requestBlob } from './client';
import type { HistoryItem, Job } from '@/types';

export const documentsApi = {
  upload(files: File[], signal?: AbortSignal): Promise<{ jobId: string }> {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file, file.name);
    }
    return request<{ jobId: string }>('/api/documents', {
      method: 'POST',
      body: formData,
      signal,
    });
  },

  getJob(jobId: string, signal?: AbortSignal): Promise<Job> {
    return request<Job>(`/api/documents/${encodeURIComponent(jobId)}`, { signal });
  },

  history(signal?: AbortSignal): Promise<HistoryItem[]> {
    return request<HistoryItem[]>('/api/documents/history', { signal });
  },

  downloadBlob(url: string, signal?: AbortSignal): Promise<Blob> {
    return requestBlob(url, signal);
  },

  async triggerDownload(url: string, filename: string): Promise<void> {
    const isAbsolute = /^https?:\/\//i.test(url);
    if (isAbsolute) {
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      return;
    }
    const blob = await requestBlob(url);
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.setTimeout(() => URL.revokeObjectURL(objUrl), 1000);
  },
};
