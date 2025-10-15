function getImportMetaEnv(): Record<string, string | undefined> {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch (err) {
    return {};
  }
}

const metaEnv = getImportMetaEnv();

import type { AccountPack } from './components/AccountCard';

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

export interface FrontendReviewManifestPack {
  account_id: string;
  holder_name?: string | null;
  primary_issue?: string | null;
  display?: AccountPack['display'];
  account_number?: unknown;
  account_type?: unknown;
  status?: unknown;
  balance_owed?: unknown;
  date_opened?: unknown;
  closed_date?: unknown;
  path?: string;
  pack_path?: string;
  has_questions?: boolean;
}

export interface FrontendReviewManifest {
  sid?: string;
  stage?: string;
  schema_version?: string | number;
  counts?: { packs?: number; responses?: number };
  packs?: FrontendReviewManifestPack[];
}

function buildFrontendReviewIndexUrl(sessionId: string): string {
  return `${API}/api/runs/${encodeURIComponent(sessionId)}/frontend/review/index`;
}

function buildFrontendReviewAccountUrl(sessionId: string, accountId: string): string {
  return `${API}/api/runs/${encodeURIComponent(sessionId)}/frontend/review/accounts/${encodeURIComponent(accountId)}`;
}

export async function fetchFrontendReviewManifest(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendReviewManifest> {
  return fetchJson<FrontendReviewManifest>(buildFrontendReviewIndexUrl(sessionId), init);
}

export async function fetchFrontendReviewAccount<T = AccountPack>(
  sessionId: string,
  accountId: string,
  init?: RequestInit
): Promise<T> {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  return fetchJson<T>(buildFrontendReviewAccountUrl(sessionId, accountId), init);
}

export async function submitFrontendReviewAnswers(
  sessionId: string,
  accountId: string,
  answers: Record<string, string>,
  init?: RequestInit
): Promise<{ ok: true }> {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  const payload = {
    answers,
    client_ts: new Date().toISOString(),
  };
  return fetchJson<{ ok: true }>(
    `${buildFrontendReviewAccountUrl(sessionId, accountId)}/answer`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      ...init,
    }
  );
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
