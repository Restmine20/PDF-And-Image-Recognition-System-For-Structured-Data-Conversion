import { useNavigate } from 'react-router-dom';

import { Button } from '@/components/Button/Button';
import styles from './NotFoundPage.module.css';

export function NotFoundPage() {
  const navigate = useNavigate();
  return (
    <div className="container">
      <div className={styles.card}>
        <div className={styles.code}>404</div>
        <h1 className={styles.title}>Страница не найдена</h1>
        <p className={styles.subtitle}>
          Проверьте адрес или вернитесь на главную, чтобы продолжить.
        </p>
        <Button onClick={() => navigate('/')}>На главную</Button>
      </div>
    </div>
  );
}
