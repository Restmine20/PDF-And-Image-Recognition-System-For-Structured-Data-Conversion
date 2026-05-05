import { Outlet } from 'react-router-dom';

import { NavBar } from '@/components/NavBar/NavBar';
import { Footer } from '@/components/Footer/Footer';

import styles from './Layout.module.css';

export function Layout() {
  return (
    <div className={styles.page}>
      <NavBar />
      <main className={styles.main}>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
