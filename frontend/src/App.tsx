import { Route, Routes } from 'react-router-dom';

import { AuthProvider } from '@/context/AuthContext';
import { ToastProvider } from '@/context/ToastContext';

import { Layout } from '@/components/Layout/Layout';
import { ProtectedRoute } from '@/components/ProtectedRoute/ProtectedRoute';

import { HomePage } from '@/pages/HomePage/HomePage';
import { AuthPage } from '@/pages/AuthPage/AuthPage';
import { HistoryPage } from '@/pages/HistoryPage/HistoryPage';
import { ApiKeyPage } from '@/pages/ApiKeyPage/ApiKeyPage';
import { AccountPage } from '@/pages/AccountPage/AccountPage';
import { ProcessingPage } from '@/pages/ProcessingPage/ProcessingPage';
import { ResultPage } from '@/pages/ResultPage/ResultPage';
import { NotFoundPage } from '@/pages/NotFoundPage/NotFoundPage';

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="/auth" element={<AuthPage />} />
            <Route path="/processing/:jobId" element={<ProcessingPage />} />
            <Route path="/result/:jobId" element={<ResultPage />} />

            <Route element={<ProtectedRoute />}>
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/api-key" element={<ApiKeyPage />} />
              <Route path="/account" element={<AccountPage />} />
            </Route>

            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </ToastProvider>
    </AuthProvider>
  );
}
