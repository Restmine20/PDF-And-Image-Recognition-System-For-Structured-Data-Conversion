import styles from './Spinner.module.css';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  label?: string;
  progress?: number;
}

export function Spinner({ size = 'md', label, progress }: SpinnerProps) {
  const hasProgress = typeof progress === 'number' && progress >= 0;

  return (
    <div className={styles.wrapper}>
      {hasProgress ? (
        <ProgressRing progress={Math.min(100, Math.max(0, progress))} size={size} />
      ) : (
        <div
          className={[styles.spinner, styles[size]].join(' ')}
          role="status"
          aria-live="polite"
        />
      )}
      {label && <p className={styles.label}>{label}</p>}
    </div>
  );
}

function ProgressRing({ progress, size }: { progress: number; size: NonNullable<SpinnerProps['size']> }) {
  const dimension = { sm: 28, md: 48, lg: 72, xl: 120 }[size];
  const stroke = { sm: 3, md: 4, lg: 6, xl: 8 }[size];
  const radius = (dimension - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className={styles.ringWrap} style={{ width: dimension, height: dimension }}>
      <svg
        width={dimension}
        height={dimension}
        viewBox={`0 0 ${dimension} ${dimension}`}
        className={styles.ring}
      >
        <circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke="var(--color-border)"
          strokeWidth={stroke}
          fill="none"
        />
        <circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke="var(--color-primary)"
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${dimension / 2} ${dimension / 2})`}
        />
      </svg>
      <span className={styles.ringValue}>{Math.round(progress)}%</span>
    </div>
  );
}
