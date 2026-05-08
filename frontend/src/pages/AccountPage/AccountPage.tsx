import { FormEvent, useState } from 'react';

import { Button } from '@/components/Button/Button';
import { useAuth } from '@/context/AuthContext';
import { useToast } from '@/context/ToastContext';
import { ApiError } from '@/types';

import styles from './AccountPage.module.css';

const PASSWORD_MIN_LENGTH = 8;

export function AccountPage() {
  const { user, changePassword } = useAuth();
  const toast = useToast();

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newPasswordRepeat, setNewPasswordRepeat] = useState('');
  const [submitting, setSubmitting] = useState(false);

  function validate(): string | null {
    if (!currentPassword) return 'Введите текущий пароль.';
    if (newPassword.length < PASSWORD_MIN_LENGTH) {
      return `Новый пароль должен быть не короче ${PASSWORD_MIN_LENGTH} символов.`;
    }
    if (newPassword === currentPassword) {
      return 'Новый пароль должен отличаться от текущего.';
    }
    if (newPassword !== newPasswordRepeat) {
      return 'Новые пароли не совпадают.';
    }
    return null;
  }

  function reset() {
    setCurrentPassword('');
    setNewPassword('');
    setNewPasswordRepeat('');
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
      await changePassword(currentPassword, newPassword);
      toast.showSuccess('Пароль обновлён.');
      reset();
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Не удалось сменить пароль.';
      toast.showError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="container">
      <header className={styles.header}>
        <h1 className={styles.title}>Профиль</h1>
        <p className={styles.subtitle}>
          Вы вошли как <strong>{user?.email}</strong>.
        </p>
      </header>

      <div className={styles.card}>
        <h2 className={styles.cardTitle}>Смена пароля</h2>
        <p className={styles.cardSubtitle}>
          После смены пароля старая сессия продолжит работать на этом устройстве.
          Если хотите завершить сессию — выйдите и войдите снова.
        </p>

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          <label className={styles.field}>
            <span className={styles.label}>Текущий пароль</span>
            <input
              type="password"
              autoComplete="current-password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className={styles.input}
              required
              disabled={submitting}
            />
          </label>

          <label className={styles.field}>
            <span className={styles.label}>Новый пароль</span>
            <input
              type="password"
              autoComplete="new-password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className={styles.input}
              placeholder={`Не короче ${PASSWORD_MIN_LENGTH} символов`}
              minLength={PASSWORD_MIN_LENGTH}
              required
              disabled={submitting}
            />
          </label>

          <label className={styles.field}>
            <span className={styles.label}>Повторите новый пароль</span>
            <input
              type="password"
              autoComplete="new-password"
              value={newPasswordRepeat}
              onChange={(e) => setNewPasswordRepeat(e.target.value)}
              className={styles.input}
              minLength={PASSWORD_MIN_LENGTH}
              required
              disabled={submitting}
            />
          </label>

          <div className={styles.actions}>
            <Button type="submit" loading={submitting}>
              Сменить пароль
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
