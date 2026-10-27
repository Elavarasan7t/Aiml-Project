import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import Results from './pages/Results';
import { checkHealth, getModelInfo } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [modelInfo, setModelInfo] = useState(null);

  const verifyBackend = useCallback(async () => {
    try {
      await checkHealth();
      setBackendStatus('online');
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err) {
      console.warn('Backend offline or health check failed:', err);
      setBackendStatus('offline');
    }
  }, []);

  useEffect(() => {
    verifyBackend();
    // Poll backend health every 15 seconds
    const interval = setInterval(verifyBackend, 15000);
    return () => clearInterval(interval);
  }, [verifyBackend]);

  const handleProcessingSuccess = () => {
    // When processing succeeds, we can automatically or manually view results
    // Let user stay on dashboard or navigate
  };

  return (
    <div className="app-container">
      <Navbar
        activePage={activePage}
        onNavigate={setActivePage}
        backendStatus={backendStatus}
        modelInfo={modelInfo}
      />

      <main className="main-content">
        {activePage === 'dashboard' ? (
          <Dashboard
            modelInfo={modelInfo}
            backendStatus={backendStatus}
            onNavigate={setActivePage}
            onProcessingSuccess={handleProcessingSuccess}
          />
        ) : (
          <Results onNavigate={setActivePage} />
        )}
      </main>

      <Footer onNavigate={setActivePage} />
    </div>
  );
}
