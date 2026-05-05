import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import { Button } from '@/components/Button/Button';
import { Spinner } from '@/components/Spinner/Spinner';
import { DownloadIcon, FileIcon, RefreshIcon } from '@/components/icons';
import { documentsApi } from '@/api/documents';
import { useToast } from '@/context/ToastContext';
import { ApiError, type HistoryItem, type JobStatus } from '@/types';

import styles from './HistoryPage.module.css';

const HISTORY_LIMIT = 20;

const STATUS_LABEL: Record<JobStatus, string> = {
  PENDING: 'В очереди',
  STARTED: 'Обработка',
  SUCCESS: 'Готово',
  FAILURE: 'Ошибка',
};

export function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const toast = useToast();

  async function load() {
    setError(null);
    setRefreshing(true);
    try {
      const data = await documentsApi.history();
      setItems(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Не удалось загрузить историю.');
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function handleDownload(item: HistoryItem) {
    if (!item.resultUrl) return;
    setDownloadingId(item.jobId);
    try {
      const filename = item.filename.replace(/\.[^.]+$/, '') + '.xlsx';
      await documentsApi.triggerDownload(item.resultUrl, filename);
    } catch (err) {
      toast.showError(err instanceof ApiError ? err.message : 'Не удалось скачать файл.');
    } finally {
      setDownloadingId(null);
    }
  }

  return (
    <div className="container">
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>История загрузок</h1>
          <p className={styles.subtitle}>
            Все ваши документы и результаты распознавания в одном месте.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          leftIcon={<RefreshIcon width={18} height={18} />}
          onClick={() => void load()}
          loading={refreshing}
        >
          Обновить
        </Button>
      </header>

      {error && (
        <div className={styles.errorBanner}>
          <p>{error}</p>
          <Button variant="ghost" size="sm" onClick={() => void load()}>
            Повторить
          </Button>
        </div>
      )}

      {items === null && !error ? (
        <div className={styles.center}>
          <Spinner size="lg" label="Загружаем историю…" />
        </div>
      ) : items && items.length === 0 ? (
        <div className={styles.empty}>
          <h2 className={styles.emptyTitle}>Здесь пока ничего нет</h2>
          <p className={styles.emptyText}>
            После загрузки документов они появятся в этом списке.
          </p>
          <Link to="/" className={styles.emptyLink}>
            Перейти к загрузке
          </Link>
        </div>
      ) : items ? (
        <>
          <ul className={styles.list}>
            {items.map((item) => (
              <li key={item.jobId} className={styles.row}>
                <span className={styles.fileIcon} aria-hidden>
                  <FileIcon width={22} height={22} />
                </span>
                <div className={styles.meta}>
                  <span className={styles.fileName} title={item.filename}>
                    {item.filename}
                  </span>
                </div>
                <StatusPill status={item.status} />
                <div className={styles.rowActions}>
                  {item.resultUrl && (
                    <Button
                      size="sm"
                      variant="primary"
                      leftIcon={<DownloadIcon width={16} height={16} />}
                      loading={downloadingId === item.jobId}
                      onClick={() => handleDownload(item)}
                    >
                      XLSX
                    </Button>
                  )}
                </div>
              </li>
            ))}
          </ul>
          {items.length >= HISTORY_LIMIT && (
            <p className={styles.limitHint}>
              Показаны последние {HISTORY_LIMIT} загрузок.
            </p>
          )}
        </>
      ) : null}
    </div>
  );
}

function StatusPill({ status }: { status: JobStatus }) {
  return (
    <span className={[styles.pill, styles[`pill_${status}`]].join(' ')}>
      {STATUS_LABEL[status]}
    </span>
  );
}

