import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { Button } from '@/components/Button/Button';
import { FileDropzone } from '@/components/FileDropzone/FileDropzone';
import { FilePreviewList } from '@/components/FilePreview/FilePreview';
import { UploadIcon } from '@/components/icons';
import { documentsApi } from '@/api/documents';
import { useToast } from '@/context/ToastContext';
import { ApiError } from '@/types';

import styles from './HomePage.module.css';

export function HomePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();

  function handleAccepted(accepted: File[]) {
    setFiles((prev) => [...prev, ...accepted]);
  }

  function handleRejected(issues: { file: File; reason: string }[]) {
    // Группируем причины, чтобы не показать 10 одинаковых тостов.
    const reasons = new Set(issues.map((i) => i.reason));
    reasons.forEach((reason) => toast.showError(reason));
  }

  function handleRemove(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleSubmit() {
    if (files.length === 0) {
      toast.showError('Выберите хотя бы один файл для отправки.');
      return;
    }

    setSubmitting(true);
    try {
      const { jobId } = await documentsApi.upload(files);
      navigate(`/processing/${jobId}`);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : 'Не удалось отправить файлы на обработку. Попробуйте ещё раз.';
      toast.showError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="container">
      <section className={styles.hero}>
        <h1 className={styles.title}>
          Превратите снимок таблицы в <span className={styles.accent}>Excel</span>
        </h1>
        <p className={styles.subtitle}>
          Загрузите фотографии или сканы документов — мы распознаем таблицы и вернём готовый XLSX.
        </p>
      </section>

      <section className={styles.uploadCard}>
        <FileDropzone
          existingFiles={files}
          onAccepted={handleAccepted}
          onRejected={handleRejected}
          disabled={submitting}
        />

        {files.length > 0 && (
          <div className={styles.previewBlock}>
            <header className={styles.previewHeader}>
              <h2 className={styles.previewTitle}>
                Выбрано файлов: {files.length}
              </h2>
              <button
                type="button"
                className={styles.clearLink}
                onClick={() => setFiles([])}
                disabled={submitting}
              >
                Очистить
              </button>
            </header>
            <FilePreviewList files={files} onRemove={handleRemove} disabled={submitting} />
          </div>
        )}

        <div className={styles.submitRow}>
          <Button
            size="lg"
            variant="primary"
            disabled={files.length === 0}
            loading={submitting}
            onClick={handleSubmit}
            leftIcon={<UploadIcon width={20} height={20} />}
          >
            Отправить
          </Button>
        </div>
      </section>

      <section className={styles.steps}>
        <Step number={1} title="Загрузите" description="Перетащите PDF или фото таблицы — поддерживается несколько файлов сразу." />
        <Step number={2} title="Дождитесь" description="Документы анализируются на сервере — это занимает от нескольких секунд." />
        <Step number={3} title="Скачайте" description="Готовый XLSX появится на странице результата и в истории загрузок." />
      </section>
    </div>
  );
}

function Step({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <div className={styles.step}>
      <div className={styles.stepNumber}>{number}</div>
      <h3 className={styles.stepTitle}>{title}</h3>
      <p className={styles.stepText}>{description}</p>
    </div>
  );
}
