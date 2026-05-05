/** Моки (тестовый вариант) */

import type { ApiKeyInfo, HistoryItem, Job } from '@/types';

const STORAGE_KEY = 'recognitionsystem.mocks.db';
const NETWORK_DELAY_MS = 300;
const MOCK_JOB_DURATION_MS = 6000;
const HISTORY_LIMIT = 20;

interface MockUser {
  id: string;
  email: string;
  password: string;
}


interface MockJob extends Job {
  createdAt: string;
}

interface MockDb {
  users: MockUser[];
  sessions: Record<string, string>;
  jobs: Record<string, MockJob>;
  apiKeys: Record<string, ApiKeyInfo>;
  historyByUser: Record<string, HistoryItem[]>;
}

function loadDb(): MockDb {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw) as MockDb;
  } catch {
    /* ignore */
  }
  return { users: [], sessions: {}, jobs: {}, apiKeys: {}, historyByUser: {} };
}

function saveDb(db: MockDb): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(db));
}

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function makeId(prefix = ''): string {
  return prefix + Math.random().toString(36).slice(2, 11);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function errorResponse(message: string, status = 400): Response {
  return jsonResponse({ message }, status);
}

function getAuthToken(init: RequestInit | undefined): string | null {
  const headers = init?.headers as Record<string, string> | undefined;
  const auth = headers?.['Authorization'] ?? headers?.['authorization'] ?? '';
  return auth.startsWith('Bearer ') ? auth.slice(7) : null;
}

function publicJob(job: MockJob): Job {
  const { createdAt: _createdAt, ...rest } = job;
  return rest;
}

export async function mockFetch(input: string, init?: RequestInit): Promise<Response> {
  await delay(NETWORK_DELAY_MS);

  const method = (init?.method ?? 'GET').toUpperCase();
  const path = new URL(input, window.location.origin).pathname;
  const db = loadDb();

  const token = getAuthToken(init);
  const currentUserId = token ? db.sessions[token] ?? null : null;
  const currentUser = currentUserId
    ? db.users.find((u) => u.id === currentUserId) ?? null
    : null;

  if (method === 'POST' && path === '/api/auth/register') {
    const { email, password } = JSON.parse(init?.body as string);
    if (db.users.some((u) => u.email === email)) {
      return errorResponse('Пользователь с таким email уже зарегистрирован.', 409);
    }
    const user: MockUser = { id: makeId('user_'), email, password };
    db.users.push(user);
    const newToken = makeId('tok_');
    db.sessions[newToken] = user.id;
    saveDb(db);
    return jsonResponse({ token: newToken, user: { email: user.email } });
  }

  if (method === 'POST' && path === '/api/auth/login') {
    const { email, password } = JSON.parse(init?.body as string);
    const user = db.users.find((u) => u.email === email && u.password === password);
    if (!user) return errorResponse('Неверный email или пароль.', 401);
    const newToken = makeId('tok_');
    db.sessions[newToken] = user.id;
    saveDb(db);
    return jsonResponse({ token: newToken, user: { email: user.email } });
  }

  if (method === 'POST' && path === '/api/auth/logout') {
    if (token) {
      delete db.sessions[token];
      saveDb(db);
    }
    return new Response(null, { status: 204 });
  }

  if (method === 'GET' && path === '/api/auth/me') {
    if (!currentUser) return errorResponse('Требуется вход.', 401);
    return jsonResponse({ user: { email: currentUser.email } });
  }

  if (method === 'POST' && path === '/api/documents') {
    const formData = init?.body as FormData;
    const files: { name: string }[] = [];
    formData.forEach((value) => {
      if (value instanceof File) files.push({ name: value.name });
    });

    const jobId = makeId('job_');
    const now = new Date().toISOString();
    const job: MockJob = {
      jobId,
      status: 'STARTED',
      createdAt: now,
      files: files.map((f) => ({ name: f.name })),
    };
    db.jobs[jobId] = job;

    if (currentUserId) {
      db.historyByUser[currentUserId] = db.historyByUser[currentUserId] ?? [];
      for (const f of files) {
        db.historyByUser[currentUserId].push({
          jobId,
          filename: f.name,
          status: 'STARTED',
        });
      }
    }
    saveDb(db);
    return jsonResponse({ jobId });
  }

  if (method === 'GET' && path === '/api/documents/history') {
    if (!currentUserId) return errorResponse('Требуется вход.', 401);
    const list = db.historyByUser[currentUserId] ?? [];
    return jsonResponse(list.slice().reverse().slice(0, HISTORY_LIMIT));
  }

  const jobMatch = path.match(/^\/api\/documents\/([^/]+)$/);
  if (method === 'GET' && jobMatch) {
    const jobId = jobMatch[1];
    const job = db.jobs[jobId];
    if (!job) return errorResponse('Задача не найдена.', 404);

    if (job.status === 'STARTED' || job.status === 'PENDING') {
      const elapsedMs = Date.now() - new Date(job.createdAt).getTime();
      if (elapsedMs >= MOCK_JOB_DURATION_MS) {
        job.status = 'SUCCESS';
        job.files = job.files.map((f, i) => ({
          ...f,
          resultUrl: `/api/documents/${jobId}/files/${i}/result`,
        }));
        for (const userId of Object.keys(db.historyByUser)) {
          db.historyByUser[userId] = db.historyByUser[userId].map((h) =>
            h.jobId === jobId
              ? { ...h, status: 'SUCCESS', resultUrl: job.files[0]?.resultUrl }
              : h,
          );
        }
        saveDb(db);
      }
    }
    return jsonResponse(publicJob(job));
  }

  if (method === 'GET' && /^\/api\/documents\/[^/]+\/files\/\d+\/result$/.test(path)) {
    const content = 'RecognitionSystem,mock,result\nrow1,foo,bar\nrow2,baz,qux';
    return new Response(content, {
      status: 200,
      headers: {
        'Content-Type':
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      },
    });
  }

  if (method === 'POST' && path === '/api/api-key') {
    if (!currentUserId) return errorResponse('Требуется вход.', 401);
    db.apiKeys[currentUserId] = { key: 'recsys_live_' + makeId() };
    saveDb(db);
    return jsonResponse(db.apiKeys[currentUserId]);
  }

  return errorResponse(`Mock не знает, как обработать ${method} ${path}.`, 404);
}

export const MOCKS_ENABLED = import.meta.env.VITE_USE_MOCKS === 'true';

if (MOCKS_ENABLED && typeof window !== 'undefined') {
  console.info(
    '%c[RecognitionSystem] Mock-режим включён',
    'background:#e5322d;color:#fff;padding:2px 6px;border-radius:4px;',
    '— все запросы идут в src/api/mocks.ts, бэкенд не требуется.',
  );
}
