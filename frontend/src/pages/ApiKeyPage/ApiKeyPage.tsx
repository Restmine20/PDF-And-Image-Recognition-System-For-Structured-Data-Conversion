import { useState } from 'react';

import { Button } from '@/components/Button/Button';
import { CopyIcon, RefreshIcon } from '@/components/icons';
import { apiKeyApi } from '@/api/apiKey';
import { useToast } from '@/context/ToastContext';
import { ApiError, type ApiKeyInfo } from '@/types';

import styles from './ApiKeyPage.module.css';

export function ApiKeyPage() {
  const [info, setInfo] = useState<ApiKeyInfo | null>(null);
  const [generating, setGenerating] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const toast = useToast();

  async function handleCopy() {
    if (!info?.key) return;
    try {
      await navigator.clipboard.writeText(info.key);
      toast.showSuccess('Ключ скопирован в буфер обмена.');
    } catch {
      toast.showError('Не удалось скопировать ключ. Скопируйте вручную.');
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    try {
      const fresh = await apiKeyApi.generate();
      setInfo(fresh);
      toast.showSuccess('Новый ключ сгенерирован. Сохраните его — повторно посмотреть нельзя.');
    } catch (err) {
      toast.showError(err instanceof ApiError ? err.message : 'Не удалось сгенерировать ключ.');
    } finally {
      setGenerating(false);
      setConfirming(false);
    }
  }

  return (
    <div className="container">
      <header className={styles.header}>
        <h1 className={styles.title}>API-ключ</h1>
        <p className={styles.subtitle}>
          Используйте ключ для вызова REST API из своих скриптов и интеграций.
          Передавайте его в заголовке <code>X-API-Key: &lt;ключ&gt;</code>.
        </p>
      </header>

      <div className={styles.card}>
        {info ? (
          <>
            <label className={styles.label}>Ваш новый ключ</label>
            <div className={styles.keyRow}>
              <code className={styles.keyValue}>{info.key}</code>
              <Button
                size="sm"
                variant="secondary"
                leftIcon={<CopyIcon width={16} height={16} />}
                onClick={handleCopy}
              >
                Копировать
              </Button>
            </div>
            <p className={styles.meta}>
              Сохраните ключ сейчас — повторно посмотреть его нельзя. Если потеряете,
              сгенерируйте новый (старый перестанет работать).
            </p>
          </>
        ) : (
          <p className={styles.subtitle}>
            Просмотреть существующий ключ нельзя. Сгенерируйте новый — он будет показан
            один раз, после чего сохраните его у себя. Любая повторная генерация
            аннулирует предыдущий ключ.
          </p>
        )}

        <div className={styles.divider} />

        <div className={styles.dangerBlock}>
          <div>
            <h2 className={styles.dangerTitle}>
              {info ? 'Сгенерировать ещё один ключ' : 'Сгенерировать ключ'}
            </h2>
            <p className={styles.dangerText}>
              {info
                ? 'Текущий ключ перестанет работать. Все ваши интеграции потребуют обновления.'
                : 'Если у вас уже есть ключ, он будет аннулирован. Все ваши интеграции потребуют обновления.'}
            </p>
          </div>
          {confirming ? (
            <div className={styles.confirmActions}>
              <Button variant="ghost" size="sm" onClick={() => setConfirming(false)}>
                Отмена
              </Button>
              <Button
                variant="danger"
                size="sm"
                loading={generating}
                onClick={handleGenerate}
              >
                Подтвердить
              </Button>
            </div>
          ) : (
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<RefreshIcon width={16} height={16} />}
              onClick={() => setConfirming(true)}
            >
              {info ? 'Сгенерировать новый' : 'Сгенерировать ключ'}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
