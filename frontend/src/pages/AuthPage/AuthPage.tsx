import { FormEvent, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { Button } from '@/components/Button/Button';
import { useAuth } from '@/context/AuthContext';
import { useToast } from '@/context/ToastContext';
import { ApiError } from '@/types';

import styles from './AuthPage.module.css';

type Mode = 'login' | 'register';

const PASSWORD_MIN_LENGTH = 8;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

interface LocationState {
  from?: string;
}

export function AuthPage() {
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordRepeat, setPasswordRepeat] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const { login, register } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const location = useLocation();
  const from = useMemo(() => {
    const state = location.state as LocationState | null;
    return state?.from && state.from !== '/auth' ? state.from : '/';
  }, [location.state]);

  function switchMode(next: Mode) {
    setMode(next);
    setPasswordRepeat('');
  }

  function validate(): string | null {
    if (!EMAIL_RE.test(email.trim())) return 'Введите корректный адрес электронной почты.';
    if (password.length < PASSWORD_MIN_LENGTH) {
      return `Пароль должен быть не короче ${PASSWORD_MIN_LENGTH} символов.`;
    }
    if (mode === 'register' && password !== passwordRepeat) {
      return 'Пароли не совпадают.';
    }
    return null;
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const validationError = validate();
    if (validationError) {
      toast.showError(validationError);
      return;
    }

    setSubmitting(true);
    try {
      if (mode === 'login') {
        await login(email.trim(), password);
        toast.showSuccess('С возвращением!');
      } else {
        await register(email.trim(), password);
        toast.showSuccess('Аккаунт создан. Добро пожаловать!');
      }
      navigate(from, { replace: true });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : 'Не удалось выполнить запрос. Попробуйте ещё раз.';
      toast.showError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="container">
      <div className={styles.card}>
        <header className={styles.header}>
          <h1 className={styles.title}>
            {mode === 'login' ? 'Вход в RecognitionSystem' : 'Создание аккаунта'}
          </h1>
          <p className={styles.subtitle}>
            {mode === 'login'
              ? 'Войдите, чтобы получить API-ключ и видеть историю своих загрузок.'
              : 'Регистрация открывает доступ к API-ключу и истории обработок.'}
          </p>
        </header>

        <div className={styles.tabs} role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={[styles.tab, mode === 'login' ? styles.tabActive : ''].join(' ')}
            onClick={() => switchMode('login')}
          >
            Вход
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            className={[styles.tab, mode === 'register' ? styles.tabActive : ''].join(' ')}
            onClick={() => switchMode('register')}
          >
            Регистрация
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          <label className={styles.field}>
            <span className={styles.label}>Email</span>
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={styles.input}
              placeholder="you@example.com"
              required
              disabled={submitting}
            />
          </label>

          <label className={styles.field}>
            <span className={styles.label}>Пароль</span>
            <input
              type="password"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={styles.input}
              placeholder={`Не короче ${PASSWORD_MIN_LENGTH} символов`}
              minLength={PASSWORD_MIN_LENGTH}
              required
              disabled={submitting}
            />
          </label>

          {mode === 'register' && (
            <label className={styles.field}>
              <span className={styles.label}>Повторите пароль</span>
              <input
                type="password"
                autoComplete="new-password"
                value={passwordRepeat}
                onChange={(e) => setPasswordRepeat(e.target.value)}
                className={styles.input}
                placeholder="Введите пароль ещё раз"
                minLength={PASSWORD_MIN_LENGTH}
                required
                disabled={submitting}
              />
            </label>
          )}

          <Button type="submit" size="lg" fullWidth loading={submitting}>
            {mode === 'login' ? 'Войти' : 'Создать аккаунт'}
          </Button>
        </form>

        <p className={styles.footnote}>
          {mode === 'login' ? (
            <>
              Ещё нет аккаунта?{' '}
              <button
                type="button"
                className={styles.linkButton}
                onClick={() => switchMode('register')}
              >
                Зарегистрируйтесь
              </button>
            </>
          ) : (
            <>
              Уже есть аккаунт?{' '}
              <button
                type="button"
                className={styles.linkButton}
                onClick={() => switchMode('login')}
              >
                Войти
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
