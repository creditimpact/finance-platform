function getImportMetaEnv(): Record<string, string | undefined> {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch (err) {
    return {};
  }
}

const metaEnv = getImportMetaEnv();

const API =
  metaEnv.VITE_API_URL ??
  metaEnv.VITE_API_BASE_URL ??
  (typeof process !== 'undefined'
    ? process.env?.VITE_API_URL ?? process.env?.VITE_API_BASE_URL
    : undefined) ??
  'http://localhost:5000';

function encodePathSegments(path: string): string {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

function buildRunAssetUrl(sessionId: string, relativePath: string): string {
  const base = `${API}/runs/${encodeURIComponent(sessionId)}`;
  if (!relativePath) {
    return base;
  }
  return `${base}/${encodePathSegments(relativePath)}`;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  let data: any = null;
  let parseError: unknown = null;
  try {
    data = await response.json();
  } catch (err) {
    parseError = err;
  }

  if (!response.ok) {
    const detail = data?.message ?? data?.error ?? response.statusText;
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }

  if (data === null && parseError) {
    throw new Error('Failed to parse response');
  }

  return data as T;
}

export interface FrontendAccountIndexEntry {
  account_id: string;
  holder_name?: string | null;
  primary_issue?: string | null;
  account_number?: unknown;
  account_type?: unknown;
  status?: unknown;
  balance_owed?: unknown;
  date_opened?: unknown;
  closed_date?: unknown;
  pack_path?: string;
}

export interface FrontendIndexResponse {
  schema_version?: string | number;
  packs_count?: number;
  accounts?: FrontendAccountIndexEntry[];
}

export async function fetchRunFrontendIndex(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendIndexResponse> {
  return fetchJson<FrontendIndexResponse>(
    buildRunAssetUrl(sessionId, 'frontend/index.json'),
    init
  );
}

export async function fetchRunAccountPack<T = unknown>(
  sessionId: string,
  packPath: string,
  init?: RequestInit
): Promise<T> {
  if (!packPath) {
    throw new Error('Missing pack path');
  }
  return fetchJson<T>(buildRunAssetUrl(sessionId, packPath), init);
}

export async function uploadReport(email: string, file: File) {
  const fd = new FormData();
  fd.append('email', email);
  fd.append('file', file);

  const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
  let data: any = {};
  try {
    data = await res.json();
  } catch (_) {}
  if (!res.ok || !data?.ok || !data?.session_id) {
    const msg = data?.message || `Upload failed (status ${res.status})`;
    throw new Error(msg);
  }
  return data as { ok: true; status: string; session_id: string; task_id?: string };
}

export async function pollResult(sessionId: string, abortSignal?: AbortSignal) {
  while (true) {
    try {
      const res = await fetch(`${API}/api/result?session_id=${encodeURIComponent(sessionId)}`, { signal: abortSignal });
      let data: any = null;
      try { data = await res.json(); } catch {}

      if (res.status === 404) {
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      if (!res.ok) {
        throw new Error(data?.message || `Result request failed (${res.status})`);
      }
      if (data?.ok && data?.status === 'done') {
        return data.result;
      }
      if (data?.ok && (data?.status === 'queued' || data?.status === 'processing')) {
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      throw new Error(data?.message || 'Processing error');
    } catch (_) {
      // Treat transient errors as in-progress and keep waiting
      await new Promise((r) => setTimeout(r, 3000));
      continue;
    }
  }
}
