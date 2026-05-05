import { NavLink, useNavigate } from 'react-router-dom';
import type { ReactNode } from 'react';

import { useAuth } from '@/context/AuthContext';
import { IconButton } from '@/components/IconButton/IconButton';
import { Button } from '@/components/Button/Button';
import {
  ClockIcon,
  HomeIcon,
  KeyIcon,
  LogoutIcon,
  UserIcon,
} from '@/components/icons';

import styles from './NavBar.module.css';

export function NavBar() {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate('/', { replace: true });
  }

  return (
    <header className={styles.navbar}>
      <div className={styles.inner}>
        <NavLink to="/" className={styles.brand} aria-label="На главную">
          <span className={styles.brandIcon} aria-hidden>
            <HomeIcon width={20} height={20} />
          </span>
          <span className={styles.brandName}>RecognitionSystem</span>
        </NavLink>

        <nav className={styles.actions} aria-label="Основная навигация">
          {isAuthenticated && (
            <>
              <NavIconLink to="/history" label="История загрузок">
                <ClockIcon />
              </NavIconLink>
              <NavIconLink to="/api-key" label="API-ключ">
                <KeyIcon />
              </NavIconLink>
            </>
          )}

          {isAuthenticated ? (
            <div className={styles.userBlock}>
              <span className={styles.userEmail} title={user?.email}>
                {user?.email}
              </span>
              <IconButton label="Выйти" onClick={handleLogout}>
                <LogoutIcon />
              </IconButton>
            </div>
          ) : (
            <Button
              variant="primary"
              size="sm"
              leftIcon={<UserIcon width={18} height={18} />}
              onClick={() => navigate('/auth')}
            >
              Регистрация / Вход
            </Button>
          )}
        </nav>
      </div>
    </header>
  );
}

/**
 * Ссылка-иконка для навбара. Использует NavLink для определения активного маршрута,
 * но рендерится как `<a>` (а не вложенный `<button>`), чтобы оставаться семантически
 * корректной ссылкой.
 */
interface NavIconLinkProps {
  to: string;
  label: string;
  children: ReactNode;
}

function NavIconLink({ to, label, children }: NavIconLinkProps) {
  return (
    <NavLink
      to={to}
      end={to === '/'}
      aria-label={label}
      title={label}
      className={({ isActive }) =>
        [styles.iconLink, isActive ? styles.iconLinkActive : ''].filter(Boolean).join(' ')
      }
    >
      {children}
    </NavLink>
  );
}
