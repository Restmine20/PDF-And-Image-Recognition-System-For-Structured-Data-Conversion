import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { Spinner } from '@/components/Spinner/Spinner';
import { Button } from '@/components/Button/Button';
import { documentsApi } from '@/api/documents';
import { ApiError, type Job } from '@/types';

import styles from './ProcessingPage.module.css';

const POLL_INTERVAL_MS = 1500;
const MAX_POLL_DURATION_MS = 10 * 60 * 1000; // 10 минут

export function ProcessingPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startedAtRef = useRef<number>(Date.now());

  useEffect(() => {
    if (!jobId) {
      setError('Не указан идентификатор задачи.');
      return;
    }

    const controller = new AbortController();
    let timeoutId: number | null = null;
    let cancelled = false;

    async function poll() {
      try {
        const fresh = await documentsApi.getJob(jobId!, controller.signal);
        if (cancelled) return;
        setJob(fresh);

        if (fresh.status === 'SUCCESS') {
          navigate(`/result/${jobId}`, { replace: true });
          return;
        }

        if (fresh.status === 'FAILURE') {
          setError(fresh.errorMessage ?? 'Сервер не смог обработать документы.');
          return;
        }

        if (Date.now() - startedAtRef.current > MAX_POLL_DURATION_MS) {
          setError('Обработка идёт слишком долго. Попробуйте позже на странице истории.');
          return;
        }

        timeoutId = window.setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        if (cancelled || (err as Error).name === 'AbortError') return;
        const message =
          err instanceof ApiError
            ? err.message
            : 'Не удалось получить статус обработки.';
        setError(message);
      }
    }

    poll();

    return () => {
      cancelled = true;
      controller.abort();
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    };
  }, [jobId, navigate]);

  if (error) {
    return (
      <div className="container">
        <div className={[styles.card, styles.errorCard].join(' ')}>
          <h1 className={styles.title}>Что-то пошло не так</h1>
          <p className={styles.errorText}>{error}</p>
          <div className={styles.actions}>
            <Button variant="secondary" onClick={() => navigate('/')}>
              На главную
            </Button>
            <Button onClick={() => navigate('/history')}>Открыть историю</Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className={styles.card}>
        <Spinner size="xl" />
        <h1 className={styles.title}>Обрабатываем ваши документы</h1>
        <p className={styles.subtitle}>
          Это может занять до нескольких минут. Не закрывайте страницу — после
          завершения вы автоматически попадёте на результат.
        </p>
        {job && (
          <p className={styles.meta}>
            Идентификатор задачи: <code>{job.jobId}</code>
          </p>
        )}
      </div>
    </div>
  );
}
