import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import styles from '@/components/Toast/Toast.module.css';

export type ToastKind = 'success' | 'error' | 'info';

interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastContextValue {
  showToast(message: string, kind?: ToastKind): void;
  showError(message: string): void;
  showSuccess(message: string): void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

const DEFAULT_DURATION_MS = 4500;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idCounter = useRef(0);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    (message: string, kind: ToastKind = 'info') => {
      const id = ++idCounter.current;
      setToasts((current) => [...current, { id, kind, message }]);
      window.setTimeout(() => dismiss(id), DEFAULT_DURATION_MS);
    },
    [dismiss],
  );

  const value = useMemo<ToastContextValue>(
    () => ({
      showToast,
      showError: (m) => showToast(m, 'error'),
      showSuccess: (m) => showToast(m, 'success'),
    }),
    [showToast],
  );

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within <ToastProvider>');
  }
  return ctx;
}

function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss(id: number): void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className={styles.viewport} role="status" aria-live="polite">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={() => onDismiss(t.id)} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss(): void }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const id = window.requestAnimationFrame(() => setVisible(true));
    return () => window.cancelAnimationFrame(id);
  }, []);

  return (
    <div
      className={[
        styles.toast,
        styles[toast.kind],
        visible ? styles.visible : '',
      ].join(' ')}
    >
      <span className={styles.message}>{toast.message}</span>
      <button
        type="button"
        className={styles.close}
        aria-label="Закрыть уведомление"
        onClick={onDismiss}
      >
        ×
      </button>
    </div>
  );
}
