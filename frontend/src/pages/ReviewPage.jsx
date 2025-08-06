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

  const getProblems = (acc) => {
    const issues = [];
    const late = acc.late_payments;
    if (late) {
      Object.values(late).forEach((counts) => {
        Object.entries(counts).forEach(([days, count]) => {
          if (count > 0) {
            issues.push({
              text: `${count} ${days}-day late payment${count > 1 ? 's' : ''}`,
              severity: 'warning',
            });
          }
        });
      });
    }
    if (acc.status) {
      issues.push({
        text: `Status: ${acc.status}`,
        severity: /collection|chargeoff/i.test(acc.status) ? 'critical' : 'normal',
      });
    }
    if (acc.balance) {
      issues.push({ text: `Balance: ${acc.balance}`, severity: 'normal' });
    }
    return issues;
  };

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
      {accounts.map((acc, idx) => {
        const problems = getProblems(acc);
        return (
          <div key={idx} className="account-block">
            <p>
              <strong>{acc.name}</strong> ({acc.account_number})
            </p>
            {problems.length > 0 && (
              <ul className="problem-list">
                {problems.map((p, i) => (
                  <li key={i} className={`problem-item ${p.severity}`}>
                    {p.severity === 'critical' ? '❌' : p.severity === 'warning' ? '⚠️' : '•'}{' '}
                    {p.text}
                  </li>
                ))}
              </ul>
            )}
            <textarea
              value={explanations[acc.name] || ''}
              onChange={(e) => handleChange(acc.name, e.target.value)}
              placeholder="Your explanation"
            />
          </div>
        );
      })}
      <button onClick={handleSubmit}>Generate Letters</button>
    </div>
  );
}
