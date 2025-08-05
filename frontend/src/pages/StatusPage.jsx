import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { checkStatus } from '../api';

export default function StatusPage() {
  const location = useLocation();
  const initialStatus = location.state?.initialStatus;
  const [status, setStatus] = useState(initialStatus?.status || 'processing');

  useEffect(() => {
    let interval;
    if (initialStatus?.task_id) {
      interval = setInterval(async () => {
        try {
          const data = await checkStatus(initialStatus.task_id);
          setStatus(data.status);
          if (data.status === 'done') {
            clearInterval(interval);
          }
        } catch (err) {
          console.error(err);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [initialStatus]);

  return (
    <div className="container">
      <h2>Processing Status</h2>
      <p>Status: {status}</p>
    </div>
  );
}
