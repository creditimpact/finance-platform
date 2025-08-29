import { useEffect, useRef, useState } from 'react';
import { uploadReport, pollResult, listAccounts, getAccount } from '../api';

export default function UploadPage() {
  const [email, setEmail] = useState('');
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | uploading | queued | processing | done | error
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [accounts, setAccounts] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [accountDetail, setAccountDetail] = useState(null);
  const abortRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please provide a PDF file.');
      return;
    }
    setStatus('uploading');
    setError('');
    try {
      const { session_id } = await uploadReport(email, file);
      setSessionId(session_id);
      setStatus('queued');

      // cancel any previous polling
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();

      // simple polling loop
      // we keep the interval externally; this call is one attempt
      const pollLoop = async () => {
        while (true) {
          const data = await pollResult(session_id, abortRef.current.signal);
          if (data?.ok && data.status === 'done') {
            setStatus('done');
            // Fetch compact accounts list from the new Accounts API
            const items = await listAccounts(session_id);
            setAccounts(items);
            setSelectedId(null);
            setAccountDetail(null);
            return;
          }
          if (data?.ok && (data.status === 'queued' || data.status === 'processing')) {
            setStatus(data.status);
            await new Promise((r) => setTimeout(r, 3000));
            continue;
          }
          // ok:false or unexpected shape
          throw new Error(data?.message || 'Processing error');
        }
      };
      await pollLoop();
    } catch (err) {
      console.error(err);
      setError(err?.message || 'Upload failed');
      setStatus('error');
    }
  };

  useEffect(() => () => abortRef.current?.abort(), []);

  return (
    <div className="container">
      <h2>Upload Credit Report</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="email">Email:</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="file">PDF File:</label>
          <input
            id="file"
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>
        {error && <p className="error">{error}</p>}
        <button
          type="submit"
          disabled={status === 'uploading' || status === 'queued' || status === 'processing'}
        >
          {status === 'uploading'
            ? 'Uploading...'
            : status === 'queued'
            ? 'Queued...'
            : status === 'processing'
            ? 'Processing...'
            : 'Start Processing'}
        </button>
      </form>
      {status === 'queued' && <p>Queued… waiting for worker.</p>}
      {status === 'processing' && <p>Processing…</p>}
      {status === 'done' && Array.isArray(accounts) && (
        <div style={{ marginTop: 16 }}>
          <h3>Problem Accounts</h3>
          {accounts.length > 0 ? (
            <ul>
              {accounts.map((acc) => {
                const name = acc.name || acc.normalized_name || 'Account';
                const singleNeg = (acc.negative_bureaus || []).length === 1 ? acc.negative_bureaus[0] : null;
                return (
                  <li key={acc.account_id} style={{ marginBottom: 12 }} onClick={async () => {
                    try {
                      setSelectedId(acc.account_id);
                      setAccountDetail(await getAccount(sessionId, acc.account_id));
                    } catch (e) { console.error(e); }
                  }}>
                    <div className="font-semibold capitalize">
                      {name}
                      {acc.primary_issue && (
                        <span className="ml-2 inline-block rounded px-2 py-0.5 text-xs" style={{ background: '#FFE8CC', color: '#9A4D00', border: '1px solid #FFD199' }}>{acc.primary_issue}</span>
                      )}
                    </div>
                    <div className="text-sm" style={{ opacity: 0.8 }}>
                      Account #: {acc.account_number_display ? <code>{acc.account_number_display}</code> : (acc.account_number_last4 ? <code>{'****' + acc.account_number_last4}</code> : <i>—</i>)}
                    </div>
                    {(acc.negative_bureaus || []).length > 0 && (
                      <div className="mt-1 text-sm">
                        {(acc.negative_bureaus || []).map((b) => (
                          <div key={b}><span className="font-medium">{b[0].toUpperCase() + b.slice(1)}:</span> flagged</div>
                        ))}
                        {singleNeg && (
                          <div className="mt-1">
                            <span className="inline-block rounded px-2 py-0.5 text-xs" style={{ background: '#FFF0E0', color: '#8A3C00', border: '1px solid #FFD8B0' }}>
                              Flagged by {singleNeg} only
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          ) : (
            <p>No problem accounts.</p>
          )}
          {selectedId && accountDetail && (
            <details style={{ marginTop: 12 }} open>
              <summary>Details for {selectedId}</summary>
              <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(accountDetail, null, 2)}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
