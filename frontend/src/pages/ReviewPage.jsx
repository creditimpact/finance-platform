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
  const debugEvidence = (() => {
    try {
      return (
        process.env.VITE_DEBUG_EVIDENCE === '1' ||
        new Function('return import.meta.env?.VITE_DEBUG_EVIDENCE')() === '1'
      );
    } catch {
      return false;
    }
  })();

  useEffect(() => {
    if (uploadData?.session_id) {
      (async () => {
        try {
          const res = await getSummaries(uploadData.session_id);
          setSummaries(res?.summaries ?? {});
        } catch (err) {
          console.warn('failed to fetch summaries', err);
          setSummaries({});
        }
      })();
    }
  }, [uploadData?.session_id]);

  if (!uploadData) {
    return <p>No upload data available.</p>;
  }

  const accounts =
    uploadData.accounts?.problem_accounts ?? [
      ...(uploadData.accounts?.negative_accounts ??
        uploadData.accounts?.disputes ??
        []),
      ...(uploadData.accounts?.open_accounts_with_issues ??
        uploadData.accounts?.goodwill ??
        []),
    ];

  // Debug: log first card's props
  if (accounts[0]) {
    console.debug('review-card-props', {
      primary_issue: accounts[0].primary_issue,
      issue_types: accounts[0].issue_types,
      last4: accounts[0].account_number_last4,
      original_creditor: accounts[0].original_creditor,
    });
  }

  const dedupedAccounts = Array.from(
    accounts
      .reduce((map, acc) => {
        const identifier = acc.account_number_last4 ?? acc.account_fingerprint ?? '';
        const key = `${
          acc.normalized_name ?? acc.name?.toLowerCase() ?? ''
        }|${identifier}`;
        const existing = map.get(key);
        if (existing) {
          existing.late_payments = {
            ...(existing.late_payments ?? {}),
            ...(acc.late_payments ?? {}),
          };
          return map;
        }
        map.set(key, acc);
        return map;
      }, new Map())
      .values(),
  );

  const formatIssueType = (type) => {
    switch (type) {
      case 'late_payment':
        return 'Late Payment';
      case 'collection':
        return 'Collection';
      case 'charge_off':
        return 'Charge-Off';
      default:
        return type ? type.charAt(0).toUpperCase() + type.slice(1) : type;
    }
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
        const issues = acc.issue_types ?? [];
        const primary = acc.primary_issue;
        const idLast4 = acc.account_number_last4 ?? null;
        const fingerprint = acc.account_fingerprint ?? null;
        const displayId = idLast4 ? `••••${idLast4}` : fingerprint ?? '';
        const secondaryIssues = issues.filter((t) => t !== primary);
        return (
          <div key={idx} className="account-block">
            <p>
              <strong>{acc.name}</strong>
              {displayId && ` ${displayId}`}
              {acc.original_creditor && ` - ${acc.original_creditor}`}
            </p>
            <div className="issue-badges">
              <span className="badge">{primary ? formatIssueType(primary) : 'Unknown'}</span>
              {secondaryIssues.map((type, i) => (
                <span key={i} className="chip">
                  {formatIssueType(type)}
                </span>
              ))}
            </div>
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
            {debugEvidence && (
              <details className="evidence-toggle">
                <summary>View evidence</summary>
                <pre className="summary-box">
                  {JSON.stringify(
                    {
                      account_trace: acc.account_trace ?? {},
                      bureau_details: acc.bureau_details ?? {},
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            )}
          </div>
        );
      })}
      <button onClick={handleSubmit}>Generate Letters</button>
    </div>
  );
}
