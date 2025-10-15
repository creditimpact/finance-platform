function getImportMetaEnv() {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch {
    return {};
  }
}

const metaEnv = getImportMetaEnv();

const API =
  metaEnv.VITE_API_URL ||
  metaEnv.VITE_API_BASE_URL ||
  (typeof process !== 'undefined'
    ? process.env?.VITE_API_URL || process.env?.VITE_API_BASE_URL
    : undefined) ||
  'http://localhost:5000';

function encodePathSegments(path = '') {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

function buildRunAssetUrl(sessionId, relativePath) {
  const base = `${API}/runs/${encodeURIComponent(sessionId)}`;
  if (!relativePath) {
    return base;
  }
  return `${base}/${encodePathSegments(relativePath)}`;
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  let data = null;
  let parseError = null;
  try {
    data = await response.json();
  } catch (error) {
    parseError = error;
  }

  if (!response.ok) {
    const detail = (data && (data.message || data.error)) || response.statusText;
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }

  if (data === null && parseError) {
    throw new Error('Failed to parse response');
  }

  return data;
}

export async function fetchRunFrontendIndex(sessionId, init) {
  return fetchJson(buildRunAssetUrl(sessionId, 'frontend/index.json'), init);
}

export async function fetchRunAccountPack(sessionId, packPath, init) {
  if (!packPath) {
    throw new Error('Missing pack path');
  }
  return fetchJson(buildRunAssetUrl(sessionId, packPath), init);
}

export async function startProcess(email, file) {
  const formData = new FormData();
  formData.append('email', email);
  formData.append('file', file);

  const response = await fetch(`${API}/api/start-process`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to start process');
  }

  return response.json();
}

// Async upload API (preferred)
export async function uploadReport(email, file) {
  const fd = new FormData();
  fd.append('email', email);
  fd.append('file', file);

  const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
  let data = {};
  try {
    data = await res.json();
  } catch {
    // swallow JSON parse errors to craft a useful message below
  }
  if (!res.ok || !data?.ok || !data?.session_id) {
    const msg = data?.message || `Upload failed (status ${res.status})`;
    throw new Error(msg);
  }
  return data; // { ok:true, status:"queued", session_id, task_id }
}

export async function pollResult(sessionId, abortSignal) {
  // One attempt of polling. Treat 404 as in-progress (session not yet materialized)
  try {
    const res = await fetch(
      `${API}/api/result?session_id=${encodeURIComponent(sessionId)}`,
      { signal: abortSignal }
    );
    let data = null;
    try {
      data = await res.json();
    } catch {
      // ignore JSON parse errors; handle via status code below
    }
    if (res.status === 404) {
      return { ok: true, status: 'processing' };
    }
    if (!res.ok) {
      throw new Error(data?.message || `Result request failed (${res.status})`);
    }
    return data;
  } catch {
    // Network/reset: surface as in-progress to keep UI tolerant
    return { ok: true, status: 'processing' };
  }
}

export async function listAccounts(sessionId) {
  const res = await fetch(`${API}/api/accounts/${encodeURIComponent(sessionId)}`);
  const data = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `List accounts failed (${res.status})`);
  }
  return data.accounts || [];
}

export async function getAccount(sessionId, accountId) {
  const res = await fetch(
    `${API}/api/accounts/${encodeURIComponent(sessionId)}/${encodeURIComponent(accountId)}`
  );
  const data = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `Get account failed (${res.status})`);
  }
  return data.account;
}

export async function submitExplanations(payload) {
  const response = await fetch(`${API}/api/submit-explanations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error('Failed to submit explanations');
  }
  return response.json();
}

export async function getSummaries(sessionId) {
  const response = await fetch(`${API}/api/summaries/${sessionId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch summaries');
  }
  return response.json();
}
