import { useLocation, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { submitExplanations } from '../api';

export default function ReviewPage() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const uploadData = state?.uploadData;
  const [explanations, setExplanations] = useState({});

  if (!uploadData) {
    return <p>No upload data available.</p>;
  }

  const accounts = [
    ...(uploadData.accounts?.negative_accounts || []),
    ...(uploadData.accounts?.open_accounts_with_issues || []),
  ];

  const handleChange = (key, value) => {
    setExplanations((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    try {
      await submitExplanations({
        session_id: uploadData.session_id,
        filename: uploadData.filename,
        email: uploadData.email,
        explanations,
      });
      navigate('/status');
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="container">
      <h2>Explain Your Situation</h2>
      {accounts.map((acc, idx) => (
        <div key={idx} className="account-block">
          <p>
            <strong>{acc.name}</strong> ({acc.account_number})
          </p>
          <textarea
            value={explanations[acc.name] || ''}
            onChange={(e) => handleChange(acc.name, e.target.value)}
            placeholder="Your explanation"
          />
        </div>
      ))}
      <button onClick={handleSubmit}>Generate Letters</button>
    </div>
  );
}
