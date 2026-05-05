import {
  useCallback,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent,
} from 'react';

import { UploadIcon } from '@/components/icons';
import {
  ACCEPTED_EXTENSIONS,
  MAX_FILE_SIZE_BYTES,
  MAX_FILES,
  validateFiles,
  type ValidationIssue,
} from '@/hooks/useFileValidation';

import styles from './FileDropzone.module.css';

interface FileDropzoneProps {
  existingFiles: File[];
  onAccepted(files: File[]): void;
  onRejected?(issues: ValidationIssue[]): void;
  disabled?: boolean;
}

export function FileDropzone({
  existingFiles,
  onAccepted,
  onRejected,
  disabled = false,
}: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const openDialog = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  const handleFiles = useCallback(
    (incoming: FileList | File[]) => {
      const list = Array.from(incoming);
      if (list.length === 0) return;
      const { accepted, rejected } = validateFiles(list, existingFiles);
      if (accepted.length > 0) onAccepted(accepted);
      if (rejected.length > 0 && onRejected) onRejected(rejected);
    },
    [existingFiles, onAccepted, onRejected],
  );

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) handleFiles(e.target.files);
    e.target.value = '';
  }

  function onDragEnter(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }
  function onDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }
  function onDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled && !isDragging) setIsDragging(true);
  }
  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openDialog();
    }
  }

  const remaining = MAX_FILES - existingFiles.length;
  const maxSizeMb = (MAX_FILE_SIZE_BYTES / (1024 * 1024)).toFixed(0);

  return (
    <div
      className={[
        styles.dropzone,
        isDragging ? styles.active : '',
        disabled ? styles.disabled : '',
      ]
        .filter(Boolean)
        .join(' ')}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onClick={openDialog}
      onKeyDown={onKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Перетащите файлы сюда или нажмите, чтобы выбрать"
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_EXTENSIONS.join(',')}
        onChange={onChange}
        className="visually-hidden"
        tabIndex={-1}
        disabled={disabled}
      />
      <div className={styles.iconCircle} aria-hidden>
        <UploadIcon width={36} height={36} />
      </div>
      <h3 className={styles.title}>Перетащите файлы сюда</h3>
      <p className={styles.subtitle}>
        или <span className={styles.linkLike}>выберите с компьютера</span>
      </p>
      <p className={styles.hint}>
        PDF, PNG, JPG · до {maxSizeMb} МБ ·{' '}
        ещё {Math.max(0, remaining)} из {MAX_FILES}
      </p>
    </div>
  );
}
