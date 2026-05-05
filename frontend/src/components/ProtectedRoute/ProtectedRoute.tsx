import { Navigate, Outlet, useLocation } from 'react-router-dom';

import { useAuth } from '@/context/AuthContext';
import { Spinner } from '@/components/Spinner/Spinner';

export function ProtectedRoute() {
  const { isAuthenticated, isInitializing } = useAuth();
  const location = useLocation();

  if (isInitializing) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '40vh',
        }}
      >
        <Spinner size="lg" label="Проверяем сессию…" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/auth" state={{ from: location.pathname }} replace />;
  }

  return <Outlet />;
}
