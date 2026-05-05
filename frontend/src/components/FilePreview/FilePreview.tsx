import { useEffect, useState } from 'react';

import { FileIcon, TrashIcon } from '@/components/icons';
import { IconButton } from '@/components/IconButton/IconButton';
import { formatFileSize } from '@/hooks/useFileValidation';

import styles from './FilePreview.module.css';

interface FilePreviewProps {
  file: File;
  onRemove(): void;
  disabled?: boolean;
}

export function FilePreview({ file, onRemove, disabled = false }: FilePreviewProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className={styles.card}>
      <div className={styles.thumb} aria-hidden>
        {previewUrl ? (
          <img src={previewUrl} alt="" />
        ) : (
          <span className={styles.thumbFallback}>
            <FileIcon width={28} height={28} />
            <span className={styles.thumbExt}>
              {file.name.split('.').pop()?.toUpperCase() ?? 'FILE'}
            </span>
          </span>
        )}
      </div>
      <div className={styles.meta}>
        <span className={styles.name} title={file.name}>
          {file.name}
        </span>
        <span className={styles.size}>{formatFileSize(file.size)}</span>
      </div>
      <IconButton label={`Удалить файл ${file.name}`} onClick={onRemove} disabled={disabled}>
        <TrashIcon />
      </IconButton>
    </div>
  );
}

interface FilePreviewListProps {
  files: File[];
  onRemove(index: number): void;
  disabled?: boolean;
}

export function FilePreviewList({ files, onRemove, disabled }: FilePreviewListProps) {
  if (files.length === 0) return null;
  return (
    <ul className={styles.list}>
      {files.map((file, idx) => (
        <li key={`${file.name}-${file.size}-${idx}`}>
          <FilePreview file={file} onRemove={() => onRemove(idx)} disabled={disabled} />
        </li>
      ))}
    </ul>
  );
}
