function getImportMetaEnv() {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch {
    return {};
  }
}

const metaEnv = getImportMetaEnv();

const apiBaseUrl =
  metaEnv.VITE_API_BASE_URL ||
  metaEnv.VITE_API_URL ||
  (typeof process !== 'undefined'
    ? process.env?.VITE_API_BASE_URL || process.env?.VITE_API_URL
    : undefined);

const API = apiBaseUrl || 'http://127.0.0.1:5000';

function encodePathSegments(path = '') {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

export function joinRunAsset(base, rel) {
  const b = base.replace(/\/+$/, '');
  const r = rel.replace(/\\/g, '/').replace(/^\/+/, '');
  return `${b}/${r}`;
}

function buildRunAssetUrl(sessionId, relativePath) {
  const base = `/runs/${encodeURIComponent(sessionId)}`;
  if (!relativePath) {
    return base;
  }
  const normalizedPath = (typeof relativePath === 'string' ? relativePath : '').replace(/\\/g, '/');
  const encodedPath = encodePathSegments(normalizedPath);
  return joinRunAsset(base, encodedPath);
}

function trimSlashes(input = '') {
  if (typeof input !== 'string') {
    return '';
  }
  let result = input.trim();
  while (result.startsWith('/')) {
    result = result.slice(1);
  }
  while (result.endsWith('/')) {
    result = result.slice(0, -1);
  }
  return result;
}

function ensureFrontendPath(candidate, fallback) {
  const trimmed = trimSlashes(typeof candidate === 'string' ? candidate : '');
  const base = trimmed || trimSlashes(typeof fallback === 'string' ? fallback : '');
  if (!base) {
    return 'frontend';
  }
  if (base.startsWith('frontend/')) {
    return base;
  }
  return `frontend/${base}`;
}

function stripFrontendPrefix(path) {
  const trimmed = trimSlashes(typeof path === 'string' ? path : '');
  if (trimmed.startsWith('frontend/')) {
    return trimmed.slice('frontend/'.length);
  }
  return trimmed;
}

function joinFrontendPath(base, child) {
  return [trimSlashes(base), trimSlashes(child)].filter(Boolean).join('/');
}

function buildFrontendReviewAccountUrl(sessionId, accountId) {
  return `${API}/api/runs/${encodeURIComponent(sessionId)}/frontend/review/accounts/${encodeURIComponent(accountId)}`;
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
    const detail = parseError instanceof Error ? parseError.message : String(parseError);
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Failed to parse JSON${suffix}`);
  }

  return data;
}

export async function fetchFrontendReviewManifest(sessionId, init) {
  const rootIndex = await fetchJson(buildRunAssetUrl(sessionId, 'frontend/index.json'), init);

  const stage = (rootIndex && rootIndex.review) || {};
  const indexPath = ensureFrontendPath(
    stage.index_rel || stage.index || 'review/index.json',
    'review/index.json'
  );
  const packsDirPath = ensureFrontendPath(
    stage.packs_dir_rel || stage.packs_dir || 'review/packs',
    'review/packs'
  );
  const responsesDirPath = ensureFrontendPath(
    stage.responses_dir_rel || stage.responses_dir || 'review/responses',
    'review/responses'
  );

  let manifestPayload = stage;
  if (!manifestPayload || !Array.isArray(manifestPayload.packs)) {
    manifestPayload = await fetchJson(buildRunAssetUrl(sessionId, indexPath));
  }

  const packs = Array.isArray(manifestPayload?.packs)
    ? manifestPayload.packs.map((entry) => {
        const pack = { ...entry };
        const rawPath =
          typeof entry.pack_path === 'string'
            ? entry.pack_path
            : typeof entry.path === 'string'
            ? entry.path
            : undefined;

        const normalizedPath = rawPath
          ? ensureFrontendPath(rawPath, joinFrontendPath(packsDirPath, `${entry.account_id}.json`))
          : joinFrontendPath(packsDirPath, `${entry.account_id}.json`);

        pack.pack_path = normalizedPath;
        pack.pack_path_rel = stripFrontendPrefix(normalizedPath);
        pack.path = normalizedPath;
        return pack;
      })
    : [];

  return {
    sid: manifestPayload?.sid || rootIndex?.sid,
    stage: manifestPayload?.stage || stage.stage || 'review',
    schema_version: manifestPayload?.schema_version || stage.schema_version,
    counts: manifestPayload?.counts || stage.counts,
    generated_at: manifestPayload?.generated_at || stage.generated_at,
    packs,
    index_rel: stripFrontendPrefix(indexPath),
    index_path: indexPath,
    packs_dir_rel: stripFrontendPrefix(packsDirPath),
    packs_dir_path: packsDirPath,
    responses_dir_rel: stripFrontendPrefix(responsesDirPath),
    responses_dir_path: responsesDirPath,
  };
}

export async function fetchFrontendReviewAccount(sessionId, accountId, init) {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  let packPath;
  let requestInit = init;

  if (init && typeof init === 'object' && Object.prototype.hasOwnProperty.call(init, 'packPath')) {
    const { packPath: candidate, ...rest } = init;
    requestInit = rest;
    if (typeof candidate === 'string' && candidate.trim()) {
      packPath = ensureFrontendPath(candidate, candidate);
    }
  }

  if (!packPath) {
    const manifest = await fetchFrontendReviewManifest(sessionId);
    const match = manifest.packs?.find((entry) => entry.account_id === accountId);
    if (match?.pack_path) {
      packPath = match.pack_path;
    } else if (match?.path) {
      packPath = ensureFrontendPath(match.path, match.path);
    } else {
      const packsDir = manifest.packs_dir_path || ensureFrontendPath('review/packs', 'review/packs');
      packPath = joinFrontendPath(packsDir, `${accountId}.json`);
    }
  }

  const normalizedPath = packPath ? ensureFrontendPath(packPath, packPath) : null;
  if (!normalizedPath) {
    throw new Error(`Unable to resolve pack path for account ${accountId}`);
  }

  return fetchJson(buildRunAssetUrl(sessionId, normalizedPath), requestInit);
}

export async function submitFrontendReviewAnswers(sessionId, accountId, answers, init) {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  const payload = {
    answers,
    client_ts: new Date().toISOString(),
  };
  return fetchJson(`${buildFrontendReviewAccountUrl(sessionId, accountId)}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    body: JSON.stringify(payload),
    ...init,
  });
}

export async function completeFrontendReview(sessionId, init) {
  const response = await fetch(`${API}/api/runs/${encodeURIComponent(sessionId)}/frontend/review/complete`, {
    method: 'POST',
    ...(init || {}),
  });

  if (!response.ok) {
    let detail;
    try {
      const data = await response.json();
      detail = data?.message || data?.error;
    } catch {
      // Ignore parse errors for non-JSON responses
    }
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }
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
