import type { ButtonHTMLAttributes, ReactNode } from 'react';
import styles from './IconButton.module.css';

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  children: ReactNode;
  active?: boolean;
}

export function IconButton({
  label,
  children,
  active = false,
  className,
  ...rest
}: IconButtonProps) {
  return (
    <button
      {...rest}
      type={rest.type ?? 'button'}
      aria-label={label}
      title={label}
      className={[
        styles.iconButton,
        active ? styles.active : '',
        className ?? '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </button>
  );
}
