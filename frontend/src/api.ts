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

function trimSlashes(input?: string | null): string {
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
function ensureFrontendPath(candidate: string | null | undefined, fallback: string): string {
  const trimmed = trimSlashes(candidate);
  const base = trimmed || trimSlashes(fallback);
  if (!base) {
    return 'frontend';
  }
  if (base.startsWith('frontend/')) {
    return base;
  }
  return `frontend/${base}`;
}

function stripFrontendPrefix(path: string | null | undefined): string {
  const trimmed = trimSlashes(path);
  if (trimmed.startsWith('frontend/')) {
    return trimmed.slice('frontend/'.length);
  }
  return trimmed;
}

function joinFrontendPath(base: string, child: string): string {
  return [trimSlashes(base), trimSlashes(child)].filter(Boolean).join('/');
}

function buildRunApiUrl(sessionId: string, path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${API}/api/runs/${encodeURIComponent(sessionId)}${normalized}`;
}

function buildFrontendReviewAccountUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/accounts')}/${encodeURIComponent(accountId)}`;
}

function buildFrontendReviewResponseUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/response')}/${encodeURIComponent(accountId)}`;
}

export function buildFrontendReviewStreamUrl(sessionId: string): string {
  return buildRunApiUrl(sessionId, '/frontend/review/stream');
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
  pack_path_rel?: string;
}

export interface FrontendReviewManifest {
  sid?: string;
  stage?: string;
  schema_version?: string | number;
  counts?: { packs?: number; responses?: number };
  packs?: FrontendReviewManifestPack[];
  generated_at?: string;
  index_rel?: string;
  index_path?: string;
  packs_dir_rel?: string;
  packs_dir_path?: string;
  responses_dir_rel?: string;
  responses_dir_path?: string;
}

interface FrontendStageDescriptor {
  stage?: string;
  schema_version?: string | number;
  generated_at?: string;
  index?: string;
  index_rel?: string;
  packs_dir?: string;
  packs_dir_rel?: string;
  responses_dir?: string;
  responses_dir_rel?: string;
  counts?: { packs?: number; responses?: number };
  packs?: FrontendReviewManifestPack[];
}

interface FrontendRootIndex {
  sid?: string;
  review?: FrontendStageDescriptor;
}

export interface RunFrontendManifestResponse {
  sid?: string;
  frontend?: {
    review?: FrontendStageDescriptor | null;
    [key: string]: unknown;
  } | null;
}

export async function fetchRunFrontendManifest(
  sessionId: string,
  init?: RequestInit
): Promise<RunFrontendManifestResponse> {
  const url = new URL(buildRunApiUrl(sessionId, '/frontend/manifest'));
  url.searchParams.set('section', 'frontend');
  const res = await fetchJson<RunFrontendManifestResponse>(url.toString(), init);

  const frontendSection = res?.frontend;
  const reviewCandidate =
    (frontendSection as { review?: FrontendStageDescriptor | null } | null | undefined)?.review ??
    (frontendSection as FrontendStageDescriptor | null | undefined) ??
    (res as { review?: FrontendStageDescriptor | null } | null | undefined)?.review ??
    null;

  const normalizedReview =
    reviewCandidate && typeof reviewCandidate === 'object' && !Array.isArray(reviewCandidate)
      ? (reviewCandidate as FrontendStageDescriptor)
      : null;

  let normalizedFrontend: RunFrontendManifestResponse['frontend'];

  if (frontendSection && typeof frontendSection === 'object' && !Array.isArray(frontendSection)) {
    normalizedFrontend = {
      ...(frontendSection as Record<string, unknown>),
      review: normalizedReview,
    } as RunFrontendManifestResponse['frontend'];
  } else if (frontendSection === null) {
    normalizedFrontend = null;
  } else if (normalizedReview) {
    normalizedFrontend = { review: normalizedReview };
  } else {
    normalizedFrontend = frontendSection as RunFrontendManifestResponse['frontend'];
  }

  return {
    ...res,
    frontend: normalizedFrontend,
  };
}

export async function fetchFrontendReviewManifest(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendReviewManifest> {
  const rootIndex = await fetchJson<FrontendRootIndex>(
    buildRunAssetUrl(sessionId, 'frontend/index.json'),
    init
  );

  const stage = (rootIndex?.review ?? {}) as FrontendStageDescriptor;
  const indexPath = ensureFrontendPath(
    stage.index_rel ?? stage.index ?? 'review/index.json',
    'review/index.json'
  );
  const packsDirPath = ensureFrontendPath(
    stage.packs_dir_rel ?? stage.packs_dir ?? 'review/packs',
    'review/packs'
  );
  const responsesDirPath = ensureFrontendPath(
    stage.responses_dir_rel ?? stage.responses_dir ?? 'review/responses',
    'review/responses'
  );

  let manifestPayload: FrontendStageDescriptor | FrontendReviewManifest | null = stage;
  if (!manifestPayload || !Array.isArray(manifestPayload.packs)) {
    manifestPayload = await fetchJson<FrontendReviewManifest>(
      buildRunAssetUrl(sessionId, indexPath)
    );
  }

  const packs = Array.isArray(manifestPayload?.packs)
    ? manifestPayload.packs.map((entry) => {
        const pack: FrontendReviewManifestPack = { ...entry };
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
    sid: manifestPayload?.sid ?? rootIndex?.sid,
    stage: manifestPayload?.stage ?? stage.stage ?? 'review',
    schema_version: manifestPayload?.schema_version ?? stage.schema_version,
    counts: manifestPayload?.counts ?? stage.counts,
    generated_at:
      (manifestPayload as FrontendStageDescriptor | FrontendReviewManifest | undefined)?.generated_at ??
      stage.generated_at,
    packs,
    index_rel: stripFrontendPrefix(indexPath),
    index_path: indexPath,
    packs_dir_rel: stripFrontendPrefix(packsDirPath),
    packs_dir_path: packsDirPath,
    responses_dir_rel: stripFrontendPrefix(responsesDirPath),
    responses_dir_path: responsesDirPath,
  };
}

export interface FrontendReviewPackListingItem {
  account_id: string;
  file?: string;
}

export interface FrontendReviewPackListingResponse {
  items: FrontendReviewPackListingItem[];
}

export async function fetchRunFrontendReviewIndex(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendStageDescriptor | Record<string, unknown>> {
  return fetchJson<FrontendStageDescriptor | Record<string, unknown>>(
    buildRunApiUrl(sessionId, '/frontend/index'),
    init
  );
}

export async function fetchRunReviewPackListing(
  sessionId: string,
  init?: RequestInit
): Promise<FrontendReviewPackListingResponse> {
  const response = await fetchJson<{ items?: FrontendReviewPackListingItem[] | null }>(
    buildRunApiUrl(sessionId, '/frontend/review/packs'),
    init
  );
  const items = Array.isArray(response.items) ? response.items : [];
  return { items };
}

export async function fetchFrontendReviewAccount<T = AccountPack>(
  sessionId: string,
  accountId: string,
  initOrOptions?: RequestInit | (RequestInit & { packPath?: string | null | undefined })
): Promise<T> {
  if (!accountId) {
    throw new Error('Missing account id');
  }

  let packPath: string | undefined;
  let init: RequestInit | undefined;

  if (initOrOptions && typeof initOrOptions === 'object' && 'packPath' in initOrOptions) {
    const { packPath: candidate, ...rest } = initOrOptions as RequestInit & {
      packPath?: string | null | undefined;
    };
    if (typeof candidate === 'string' && candidate.trim()) {
      packPath = ensureFrontendPath(candidate, candidate);
    }
    init = rest as RequestInit;
  } else {
    init = initOrOptions as RequestInit | undefined;
  }

  if (!packPath) {
    const manifest = await fetchFrontendReviewManifest(sessionId);
    const match = manifest.packs?.find((entry) => entry.account_id === accountId);
    if (match?.pack_path) {
      packPath = match.pack_path;
    } else if (match?.path) {
      packPath = ensureFrontendPath(match.path, match.path);
    } else {
      const packsDir = manifest.packs_dir_path ?? ensureFrontendPath('review/packs', 'review/packs');
      packPath = joinFrontendPath(packsDir, `${accountId}.json`);
    }
  }

  const normalizedPath = packPath ? ensureFrontendPath(packPath, packPath) : undefined;
  if (!normalizedPath) {
    throw new Error(`Unable to resolve pack path for account ${accountId}`);
  }

  return fetchJson<T>(buildRunAssetUrl(sessionId, normalizedPath), init);
}

export interface FrontendReviewResponseClientMeta {
  user_agent?: string | null;
  tz?: string | null;
  ts?: string | null;
  [key: string]: unknown;
}

export interface FrontendReviewResponse {
  account_id?: string | null;
  answers?: Record<string, string> | null;
  client_meta?: FrontendReviewResponseClientMeta | null;
  saved_at?: string | null;
  [key: string]: unknown;
}

function resolveUserAgent(): string | undefined {
  if (typeof navigator !== 'undefined' && typeof navigator.userAgent === 'string') {
    return navigator.userAgent;
  }
  return undefined;
}

function resolveTimeZone(): string | undefined {
  if (typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function') {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      if (typeof tz === 'string' && tz.trim() !== '') {
        return tz;
      }
    } catch (err) {
      console.warn('Unable to resolve timezone', err);
    }
  }
  return undefined;
}

export async function submitFrontendReviewAnswers(
  sessionId: string,
  accountId: string,
  answers: Record<string, string>,
  init?: RequestInit
): Promise<FrontendReviewResponse> {
  if (!accountId) {
    throw new Error('Missing account id');
  }
  const payload = {
    answers,
    client_meta: {
      user_agent: resolveUserAgent() ?? 'unknown',
      tz: resolveTimeZone() ?? 'UTC',
      ts: new Date().toISOString(),
    },
  };
  return fetchJson<FrontendReviewResponse>(
    buildFrontendReviewResponseUrl(sessionId, accountId),
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      body: JSON.stringify(payload),
      ...init,
    }
  );
}

export async function completeFrontendReview(sessionId: string, init?: RequestInit): Promise<void> {
  const response = await fetch(buildRunApiUrl(sessionId, '/frontend/review/complete'), {
    method: 'POST',
    ...(init ?? {}),
  });

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const data = await response.json();
      detail = data?.message ?? data?.error ?? undefined;
    } catch (err) {
      // Ignore parse errors for non-JSON responses
    }
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(`Request failed (${response.status})${suffix}`);
  }
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

export interface SubmitExplanationsPayload {
  [key: string]: unknown;
}

export async function submitExplanations(payload: SubmitExplanationsPayload): Promise<any> {
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

export async function getSummaries(sessionId: string): Promise<any> {
  const response = await fetch(`${API}/api/summaries/${encodeURIComponent(sessionId)}`);

  if (!response.ok) {
    throw new Error('Failed to fetch summaries');
  }

  return response.json();
}
