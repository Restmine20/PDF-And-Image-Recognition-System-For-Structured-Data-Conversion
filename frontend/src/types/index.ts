export interface User {
  email: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export type JobStatus = 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE';

export interface JobFile {
  name: string;
  resultUrl?: string;
}

export interface Job {
  jobId: string;
  status: JobStatus;
  errorMessage?: string;
  files: JobFile[];
}

export interface HistoryItem {
  jobId: string;
  filename: string;
  status: JobStatus;
  resultUrl?: string;
}

export interface ApiKeyInfo {
  key: string;
}

export interface ApiErrorPayload {
  message: string;
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}
