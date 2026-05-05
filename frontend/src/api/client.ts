import { ApiError, ApiErrorPayload } from '@/types';
import { MOCKS_ENABLED, mockFetch } from './mocks';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const TOKEN_STORAGE_KEY = 'recognitionsystem.session.token';

const fetchImpl: typeof fetch = MOCKS_ENABLED
  ? ((input, init) => mockFetch(input as string, init as RequestInit))
  : fetch.bind(window);

export const tokenStorage = {
  get(): string | null {
    try {
      return localStorage.getItem(TOKEN_STORAGE_KEY);
    } catch {
      return null;
    }
  },
  set(token: string): void {
    try {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } catch {
      /* noop */
    }
  },
  clear(): void {
    try {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    } catch {
      /* noop */
    }
  },
};

type Body = Record<string, unknown> | FormData | undefined;

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  body?: Body;
  signal?: AbortSignal;
  anonymous?: boolean;
}

async function parseError(response: Response): Promise<ApiError> {
  let payload: ApiErrorPayload | null = null;
  try {
    payload = (await response.json()) as ApiErrorPayload;
  } catch {
  }

  const message =
    payload?.message ||
    defaultMessageForStatus(response.status) ||
    `Ошибка ${response.status}`;

  return new ApiError(message, response.status);
}

function defaultMessageForStatus(status: number): string | undefined {
  switch (status) {
    case 400:
      return 'Некорректный запрос. Проверьте введённые данные.';
    case 401:
      return 'Требуется вход в систему.';
    case 403:
      return 'Недостаточно прав для этого действия.';
    case 404:
      return 'Ресурс не найден.';
    case 409:
      return 'Конфликт данных (возможно, такой email уже зарегистрирован).';
    case 413:
      return 'Файл слишком большой.';
    case 415:
      return 'Неподдерживаемый формат файла.';
    case 429:
      return 'Слишком много запросов. Попробуйте позже.';
    case 500:
    case 502:
    case 503:
    case 504:
      return 'Ошибка на сервере. Попробуйте позже.';
    default:
      return undefined;
  }
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, anonymous = false } = options;

  const headers: Record<string, string> = {};
  let payload: BodyInit | undefined;

  if (body instanceof FormData) {
    payload = body;
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  if (!anonymous) {
    const token = tokenStorage.get();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetchImpl(`${BASE_URL}${path}`, {
      method,
      headers,
      body: payload,
      signal,
      credentials: 'include',
    });
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    throw new ApiError(
      'Не удалось подключиться к серверу. Проверьте интернет-соединение.',
      0,
    );
  }

  if (response.status === 401 && !anonymous) {
    tokenStorage.clear();
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('application/json')) {
    return (await response.json()) as T;
  }

  return (await response.text()) as unknown as T;
}

export async function requestBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  const token = tokenStorage.get();
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const response = await fetchImpl(`${BASE_URL}${path}`, {
    method: 'GET',
    headers,
    signal,
    credentials: 'include',
  });

  if (!response.ok) {
    throw await parseError(response);
  }

  return await response.blob();
}
