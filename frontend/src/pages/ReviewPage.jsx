import { useLocation, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { submitExplanations, getSummaries } from '../api';

export default function ReviewPage() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const uploadData = state?.uploadData;
  const [explanations, setExplanations] = useState({});
  const [summaries, setSummaries] = useState({});
  const [showSummary, setShowSummary] = useState({});

  useEffect(() => {
    if (uploadData?.session_id) {
      getSummaries(uploadData.session_id)
        .then((res) => setSummaries(res.summaries || {}))
        .catch(() => {});
    }
  }, [uploadData?.session_id]);

  if (!uploadData) {
    return <p>No upload data available.</p>;
  }

  const accounts = [
    ...(uploadData.accounts?.negative_accounts ?? uploadData.accounts?.disputes ?? []),
    ...(uploadData.accounts?.open_accounts_with_issues ?? uploadData.accounts?.goodwill ?? []),
  ].filter((acc) => acc.issue_types && acc.issue_types.length);

  const dedupedAccounts = Array.from(
    accounts
      .reduce((map, acc) => {
        const key = acc.account_id ?? acc.name?.toLowerCase();
        const existing = map.get(key);
        if (existing) {
          existing.late_payments = { ...existing.late_payments, ...acc.late_payments };
          return map;
        }
        map.set(key, acc);
        return map;
      }, new Map())
      .values(),
  );

  const getProblems = (acc) => {
    return acc.issue_types.map((type) => {
      switch (type) {
        case 'late_payment':
          return { text: 'Late payments detected', severity: 'warning' };
        case 'collection':
          return { text: 'Account in collections', severity: 'critical' };
        case 'charge_off':
          return { text: 'Account charged off', severity: 'critical' };
        default:
          return { text: type, severity: 'normal' };
      }
    });
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
      {dedupedAccounts.map((acc, idx) => {
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
            <small className="helper-text">
              We’ll use your explanation as context to better understand your case. It will
              not be copied word-for-word into your dispute letter.
            </small>
            <div className="summary-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={!!showSummary[acc.account_id]}
                  onChange={(e) =>
                    setShowSummary((prev) => ({ ...prev, [acc.account_id]: e.target.checked }))
                  }
                />
                Show how the system understood your explanation
              </label>
            </div>
            {showSummary[acc.account_id] && summaries[acc.account_id] && (
              <pre className="summary-box">
                {JSON.stringify(summaries[acc.account_id], null, 2)}
              </pre>
            )}
          </div>
        );
      })}
      <button onClick={handleSubmit}>Generate Letters</button>
    </div>
  );
}
