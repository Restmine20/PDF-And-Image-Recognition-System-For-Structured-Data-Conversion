import type { ButtonHTMLAttributes, ReactNode } from 'react';
import styles from './Button.module.css';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  fullWidth?: boolean;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

export function Button({
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  loading = false,
  disabled,
  leftIcon,
  rightIcon,
  children,
  className,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={[
        styles.button,
        styles[variant],
        styles[size],
        fullWidth ? styles.fullWidth : '',
        loading ? styles.loading : '',
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {loading && <span className={styles.spinner} aria-hidden />}
      {!loading && leftIcon && <span className={styles.icon}>{leftIcon}</span>}
      <span className={styles.label}>{children}</span>
      {!loading && rightIcon && <span className={styles.icon}>{rightIcon}</span>}
    </button>
  );
}
