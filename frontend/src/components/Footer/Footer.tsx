import styles from './Footer.module.css';

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={[styles.inner, 'container'].join(' ')}>
        <span className={styles.copy}>
          © {new Date().getFullYear()} RecognitionSystem — оцифровка таблиц
        </span>
        <span className={styles.note}>
          Файлы обрабатываются на сервере и удаляются автоматически по истечении срока хранения.
        </span>
      </div>
    </footer>
  );
}
