import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { Button } from '@/components/Button/Button';
import { Spinner } from '@/components/Spinner/Spinner';
import { CheckIcon, DownloadIcon, FileIcon } from '@/components/icons';
import { documentsApi } from '@/api/documents';
import { useToast } from '@/context/ToastContext';
import { ApiError, type Job } from '@/types';

import styles from './ResultPage.module.css';

export function ResultPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const toast = useToast();

  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingUrl, setDownloadingUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) {
      setError('Не указан идентификатор задачи.');
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    documentsApi
      .getJob(jobId, controller.signal)
      .then(setJob)
      .catch((err: unknown) => {
        if ((err as Error).name === 'AbortError') return;
        setError(err instanceof ApiError ? err.message : 'Не удалось загрузить результат.');
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [jobId]);

  async function handleDownload(url: string, suggestedName: string) {
    setDownloadingUrl(url);
    try {
      await documentsApi.triggerDownload(url, suggestedName);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Не удалось скачать файл.';
      toast.showError(message);
    } finally {
      setDownloadingUrl(null);
    }
  }

  if (loading) {
    return (
      <div className="container">
        <div className={styles.card}>
          <Spinner size="lg" label="Загружаем результат…" />
        </div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="container">
        <div className={[styles.card, styles.errorCard].join(' ')}>
          <h1 className={styles.title}>Не удалось открыть результат</h1>
          <p className={styles.subtitle}>{error ?? 'Задача не найдена.'}</p>
          <Button onClick={() => navigate('/')}>На главную</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className={styles.card}>
        <span className={styles.successBadge}>
          <CheckIcon width={20} height={20} /> Готово
        </span>
        <h1 className={styles.title}>Документы обработаны</h1>
        <p className={styles.subtitle}>
          Скачайте результат — таблицы в формате XLSX. Для зарегистрированных
          пользователей результаты сохраняются в истории.
        </p>

        <ul className={styles.fileList}>
          {job.files.map((file, idx) => (
            <li key={`${file.name}-${idx}`} className={styles.fileItem}>
              <span className={styles.fileIcon} aria-hidden>
                <FileIcon width={28} height={28} />
              </span>
              <div className={styles.fileMeta}>
                <span className={styles.fileName} title={file.name}>
                  {file.name}
                </span>
              </div>
              <Button
                size="sm"
                variant="primary"
                leftIcon={<DownloadIcon width={18} height={18} />}
                disabled={!file.resultUrl}
                loading={downloadingUrl === file.resultUrl}
                onClick={() =>
                  file.resultUrl && handleDownload(file.resultUrl, replaceExtension(file.name, 'xlsx'))
                }
              >
                Скачать XLSX
              </Button>
            </li>
          ))}
        </ul>

        <div className={styles.actions}>
          <Button variant="secondary" onClick={() => navigate('/')}>
            Обработать ещё
          </Button>
          <Button variant="ghost" onClick={() => navigate('/history')}>
            История загрузок
          </Button>
        </div>
      </div>
    </div>
  );
}

function replaceExtension(name: string, ext: string): string {
  const idx = name.lastIndexOf('.');
  const base = idx >= 0 ? name.slice(0, idx) : name;
  return `${base}.${ext}`;
}

