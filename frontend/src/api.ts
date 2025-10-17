function getImportMetaEnv(): Record<string, string | boolean | undefined> {
  try {
    return new Function('return (typeof import !== "undefined" && import.meta && import.meta.env) || {};')();
  } catch (err) {
    return {};
  }
}

const metaEnv = getImportMetaEnv();

import type { AccountPack } from './components/AccountCard';
import { REVIEW_DEBUG_ENABLED, reviewDebugLog } from './utils/reviewDebug';

const metaEnvConfiguredApiBaseRaw = metaEnv.VITE_API_BASE_URL;

const trimmedMetaEnvConfiguredApiBase =
  typeof metaEnvConfiguredApiBaseRaw === 'string'
    ? metaEnvConfiguredApiBaseRaw.trim()
    : typeof metaEnvConfiguredApiBaseRaw === 'number' ||
        typeof metaEnvConfiguredApiBaseRaw === 'boolean'
      ? String(metaEnvConfiguredApiBaseRaw).trim()
      : '';

const fallbackConfiguredApiBase =
  metaEnv.VITE_API_URL ??
  (typeof process !== 'undefined'
    ? process.env?.VITE_API_BASE_URL ?? process.env?.VITE_API_URL
    : undefined);

const rawConfiguredApiBase =
  trimmedMetaEnvConfiguredApiBase || fallbackConfiguredApiBase;

const trimmedConfiguredApiBase =
  typeof rawConfiguredApiBase === 'string' ? rawConfiguredApiBase.trim() : '';

const metaEnvDev = (metaEnv as Record<string, unknown>).DEV;
const isMetaEnvDev =
  typeof metaEnvDev === 'boolean'
    ? metaEnvDev
    : typeof metaEnvDev === 'string'
      ? metaEnvDev.toLowerCase() === 'true'
      : false;

const processEnv = typeof process !== 'undefined' ? process.env : undefined;
const nodeEnv = processEnv?.NODE_ENV;
const isProcessDev = typeof nodeEnv === 'string' ? nodeEnv !== 'production' : false;

const fallbackApiBase =
  !trimmedConfiguredApiBase && (isMetaEnvDev || isProcessDev) ? 'http://127.0.0.1:5000' : '';

const effectiveApiBaseInput = trimmedConfiguredApiBase || fallbackApiBase;

export const API_BASE_URL = effectiveApiBaseInput
  ? effectiveApiBaseInput.replace(/\/+$/, '')
  : '';

export const API_BASE_CONFIGURED = trimmedMetaEnvConfiguredApiBase.length > 0;
export const API_BASE_INFERRED = !API_BASE_CONFIGURED && API_BASE_URL.length > 0;

if (API_BASE_INFERRED && typeof console !== 'undefined') {
  console.warn(
    '[api] Falling back to default API base URL. Configure VITE_API_BASE_URL to point to your backend.'
  );
}

const API_BASE = API_BASE_URL;

export const apiUrl = (path: string) =>
  `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;

function encodePathSegments(path: string): string {
  return path
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

export function joinRunAsset(base: string, rel: string): string {
  const b = base.replace(/\/+$/, '');
  const r = rel.replace(/\\/g, '/').replace(/^\/+/, '');
  return `${b}/${r}`;
}

function buildRunAssetUrl(sessionId: string, relativePath: string): string {
  const basePath = `/runs/${encodeURIComponent(sessionId)}`;
  const baseUrl = apiUrl(basePath);
  if (!relativePath) {
    return baseUrl;
  }
  const normalizedPath = relativePath.replace(/\\/g, '/');
  const encodedPath = encodePathSegments(normalizedPath);
  return joinRunAsset(baseUrl, encodedPath);
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

function isAbsoluteUrl(candidate: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(candidate) || candidate.startsWith('//');
}

function buildRunApiUrl(sessionId: string, path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return apiUrl(`/api/runs/${encodeURIComponent(sessionId)}${normalized}`);
}

export function buildRunFrontendManifestUrl(sessionId: string): string {
  const baseUrl = buildRunApiUrl(sessionId, '/frontend/manifest');
  return baseUrl.includes('?') ? `${baseUrl}&section=frontend` : `${baseUrl}?section=frontend`;
}

function buildFrontendReviewAccountUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/accounts')}/${encodeURIComponent(accountId)}`;
}

function buildFrontendReviewResponseUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/response')}/${encodeURIComponent(accountId)}`;
}

function buildFrontendReviewPackUrl(sessionId: string, accountId: string): string {
  return `${buildRunApiUrl(sessionId, '/frontend/review/pack')}/${encodeURIComponent(accountId)}`;
}

export function buildFrontendReviewStreamUrl(sessionId: string): string {
  return buildRunApiUrl(sessionId, '/frontend/review/stream');
}

export interface FetchJsonErrorInfo {
  status?: number;
  statusText?: string;
  url?: string;
  responseText?: string;
}

export class FetchJsonError extends Error {
  status?: number;
  statusText?: string;
  url?: string;
  responseText?: string;

  constructor(message: string, info: FetchJsonErrorInfo = {}) {
    super(message);
    this.name = 'FetchJsonError';
    this.status = info.status;
    this.statusText = info.statusText;
    this.url = info.url;
    this.responseText = info.responseText;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:start', { url, init });
  }
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new FetchJsonError(`Request failed: ${detail}`, { url });
  }
  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:response', {
      url,
      status: response.status,
      ok: response.ok,
    });
  }

  const contentType = response.headers.get('Content-Type') ?? undefined;
  let rawBody: string | null = null;
  let data: unknown = null;
  let parseError: unknown = null;

  try {
    rawBody = await response.text();
  } catch (err) {
    parseError = err;
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:parse-error', { url, error: err, contentType });
    }
  }

  if (rawBody != null && parseError == null) {
    try {
      data = JSON.parse(rawBody);
    } catch (err) {
      parseError = err;
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetch:parse-error', {
          url,
          error: err,
          contentType,
          snippet: rawBody.slice(0, 200),
        });
      }
    }
  }

  if (!response.ok) {
    const bodyRecord = (data ?? null) && typeof data === 'object' && !Array.isArray(data)
      ? (data as Record<string, unknown>)
      : null;
    const detail =
      bodyRecord?.message ??
      bodyRecord?.error ??
      response.statusText ??
      (typeof rawBody === 'string' && rawBody ? rawBody.slice(0, 200) : undefined);
    const suffix = detail ? `: ${detail}` : '';
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:error', { url, status: response.status, detail });
    }
    throw new FetchJsonError(`Request failed (${response.status})${suffix}`, {
      status: response.status,
      statusText: response.statusText ?? undefined,
      url,
      responseText: typeof rawBody === 'string' ? rawBody : undefined,
    });
  }

  if (parseError) {
    const snippet = typeof rawBody === 'string' ? rawBody.slice(0, 200) : '';
    console.error(
      `[frontend-review] JSON parse failed (${url}) - content-type: ${contentType ?? 'unknown'}; snippet(first 200 bytes): ${snippet}`
    );
    const detail = parseError instanceof Error ? parseError.message : String(parseError);
    const suffix = detail ? `: ${detail}` : '';
    throw new FetchJsonError(`Failed to parse JSON${suffix}`, {
      url,
      responseText: typeof rawBody === 'string' ? rawBody : undefined,
    });
  }

  if (REVIEW_DEBUG_ENABLED) {
    reviewDebugLog('fetch:success', { url, body: data });
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
  const url = buildRunFrontendManifestUrl(sessionId);
  const res = await fetchJson<RunFrontendManifestResponse>(url, init);

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
  const normalizedItems = items.map((item) => {
    if (!item || typeof item !== 'object') {
      return item as FrontendReviewPackListingItem;
    }
    const file = typeof item.file === 'string' ? item.file.replace(/\\/g, '/') : item.file;
    if (file === item.file) {
      return item as FrontendReviewPackListingItem;
    }
    return { ...item, file } as FrontendReviewPackListingItem;
  });
  return { items: normalizedItems };
}

type FetchFrontendReviewAccountOptions = RequestInit & {
  staticPath?: string | null | undefined;
};

function normalizeStaticPackPath(path: string): string {
  const replaced = path.replace(/\\/g, '/').trim();
  if (!replaced) {
    return replaced;
  }
  if (isAbsoluteUrl(replaced)) {
    return replaced;
  }
  const trimmed = trimSlashes(replaced);
  if (!trimmed) {
    return trimmed;
  }
  return ensureFrontendPath(trimmed, trimmed);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

type ExtractedAccountPack = AccountPack & {
  account_id?: string | null;
  answers?: Record<string, string> | null;
  response?: FrontendReviewResponse | null;
};

function unwrapPackCandidate(source: unknown): Record<string, unknown> | null {
  if (!isRecord(source)) {
    return null;
  }

  const visited = new Set<Record<string, unknown>>();
  let current: Record<string, unknown> | null = source;

  while (current && !visited.has(current)) {
    visited.add(current);
    const nested = (current as { pack?: unknown }).pack;
    if (isRecord(nested)) {
      current = nested;
      continue;
    }
    break;
  }

  return current;
}

function extractPackPayload(
  candidate: unknown,
  fallbackAccountId?: string
): ExtractedAccountPack | null {
  const packCandidate = unwrapPackCandidate(candidate);
  if (!packCandidate) {
    return null;
  }

  const result = { ...packCandidate } as ExtractedAccountPack;
  const rootRecord = isRecord(candidate) ? (candidate as Record<string, unknown>) : null;

  const accountId = result.account_id;
  if (typeof accountId !== 'string' || accountId.trim() === '') {
    const normalizedFallback =
      typeof fallbackAccountId === 'string' ? fallbackAccountId.trim() : '';
    if (!normalizedFallback) {
      return null;
    }
    result.account_id = normalizedFallback;
  }

  if (result.answers == null && rootRecord) {
    const rootAnswers = rootRecord.answers;
    if (isRecord(rootAnswers)) {
      result.answers = rootAnswers as Record<string, string>;
    }
  }

  if (result.response == null && rootRecord) {
    const rootResponse = rootRecord.response;
    if (isRecord(rootResponse)) {
      result.response = rootResponse as FrontendReviewResponse;
    }
  }

  return result;
}

function normalizeAccountPackPayload(
  candidate: unknown,
  fallbackAccountId: string
): ExtractedAccountPack | null {
  const pack = extractPackPayload(candidate, fallbackAccountId);
  if (!pack) {
    return null;
  }

  const normalizedAccountId =
    typeof pack.account_id === 'string' ? pack.account_id.trim() : '';
  if (!normalizedAccountId) {
    return null;
  }

  pack.account_id = normalizedAccountId;
  return pack;
}

function hasDisplayData(
  pack: ExtractedAccountPack | null | undefined
): pack is ExtractedAccountPack & { display: NonNullable<AccountPack['display']> } {
  if (!pack) {
    return false;
  }

  const display = pack.display;
  if (!display || typeof display !== 'object' || Array.isArray(display)) {
    return false;
  }

  return true;
}

type FrontendReviewAccountAttemptKind = 'account' | 'pack' | 'static';

interface FrontendReviewAccountAttempt {
  kind: FrontendReviewAccountAttemptKind;
  url: string;
  label: string;
  error?: Error;
  status?: number;
  responseText?: string;
}

export interface FrontendReviewAccountAttemptResult {
  kind: FrontendReviewAccountAttemptKind;
  url: string;
  label: string;
  error?: Error;
  status?: number;
  responseText?: string;
}

export class FrontendReviewAccountError extends Error {
  readonly attempts: ReadonlyArray<FrontendReviewAccountAttemptResult>;

  constructor(message: string, attempts: FrontendReviewAccountAttempt[]) {
    super(message);
    this.name = 'FrontendReviewAccountError';
    this.attempts = attempts.map((attempt) => ({
      kind: attempt.kind,
      url: attempt.url,
      label: attempt.label,
      error: attempt.error,
      status: attempt.status,
      responseText: attempt.responseText,
    }));
  }
}

export async function fetchFrontendReviewAccount<T = AccountPack>(
  sessionId: string,
  accountId: string,
  initOrOptions?: RequestInit | FetchFrontendReviewAccountOptions
): Promise<T> {
  if (!accountId) {
    throw new Error('Missing account id');
  }

  let init: RequestInit | undefined;
  let staticPath: string | undefined;

  if (initOrOptions && typeof initOrOptions === 'object' && 'staticPath' in initOrOptions) {
    const { staticPath: providedStaticPath, ...rest } = initOrOptions as FetchFrontendReviewAccountOptions;
    if (typeof providedStaticPath === 'string' && providedStaticPath.trim() !== '') {
      const normalized = normalizeStaticPackPath(providedStaticPath);
      staticPath = normalized || undefined;
    }
    init = rest as RequestInit;
  } else {
    init = initOrOptions as RequestInit | undefined;
  }

  if (!staticPath) {
    const defaultStaticPath = normalizeStaticPackPath(
      joinFrontendPath('frontend/review/packs', `${accountId}.json`)
    );
    staticPath = defaultStaticPath || undefined;
  }

  const staticUrl = staticPath
    ? isAbsoluteUrl(staticPath)
      ? staticPath
      : buildRunAssetUrl(sessionId, staticPath)
    : null;
  const failureMessages: string[] = [];
  const attemptRecords: FrontendReviewAccountAttempt[] = [];
  if (staticUrl) {
    const staticLabel = stripFrontendPrefix(staticPath) || staticPath;
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetchFrontendReviewAccount:attempt', {
        accountId,
        label: `static(${staticLabel})`,
        url: staticUrl,
      });
    }

    try {
      const staticInit: RequestInit = { ...(init ?? {}), cache: 'no-store' };
      const payload = await fetchJson<unknown>(staticUrl, staticInit);
      const pack = normalizeAccountPackPayload(payload, accountId);
      if (!pack) {
        throw new Error('No pack payload in response');
      }
      console.info(
        `[frontend-review] Pack ${accountId}: loaded from static fallback (${staticUrl})`
      );
      return pack as T;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      const attemptRecord: FrontendReviewAccountAttempt = {
        kind: 'static',
        url: staticUrl,
        label: `static(${staticLabel})`,
        error,
      };
      if (err instanceof FetchJsonError) {
        attemptRecord.status = err.status;
        attemptRecord.responseText = err.responseText ?? undefined;
      }
      attemptRecords.push(attemptRecord);
      failureMessages.push(`static(${staticLabel}): ${error.message}`);
      console.error(
        `[frontend-review] Pack ${accountId}: static fallback failed (${staticUrl}) - ${error.message}`
      );
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetchFrontendReviewAccount:attempt-error', {
          accountId,
          label: `static(${staticLabel})`,
          url: staticUrl,
          error,
        });
      }
    }
  }

  const attempts: FrontendReviewAccountAttempt[] = [
    {
      kind: 'account',
      url: buildFrontendReviewAccountUrl(sessionId, accountId),
      label: 'accounts/:id',
    },
    {
      kind: 'pack',
      url: buildFrontendReviewPackUrl(sessionId, accountId),
      label: 'pack/:id',
    },
  ];
  attemptRecords.push(...attempts);

  let fallbackPack: ExtractedAccountPack | null = null;
  let fallbackSource: FrontendReviewAccountAttempt | null = null;

  for (const attempt of attempts) {
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetchFrontendReviewAccount:attempt', {
        accountId,
        label: attempt.label,
        url: attempt.url,
      });
    }

    try {
      const payload = await fetchJson<unknown>(attempt.url, init);
      const pack = normalizeAccountPackPayload(payload, accountId);

      if (pack) {
        if (hasDisplayData(pack)) {
          if (attempt.kind === 'static') {
            console.info(
              `[frontend-review] Pack ${accountId}: loaded from static fallback (${attempt.url})`
            );
          }
          return pack as T;
        }

        if (!fallbackPack) {
          fallbackPack = pack;
          fallbackSource = attempt;
        }

        continue;
      }

      const noPackError = new Error('No pack payload in response');
      attempt.error = noPackError;
      console.error(
        `[frontend-review] Pack ${accountId}: ${attempt.label} returned no pack payload (${attempt.url})`
      );
      failureMessages.push(`${attempt.label}: ${noPackError.message}`);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      attempt.error = error;
      if (err instanceof FetchJsonError) {
        attempt.status = err.status;
        attempt.responseText = err.responseText ?? undefined;
      }

      console.error(
        `[frontend-review] Pack ${accountId}: ${attempt.label} failed (${attempt.url}) - ${error.message}`
      );

      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetchFrontendReviewAccount:attempt-error', {
          accountId,
          label: attempt.label,
          url: attempt.url,
          error,
        });
      }
      failureMessages.push(`${attempt.label}: ${error.message}`);
    }
  }

  if (fallbackPack) {
    if (fallbackSource?.kind === 'static') {
      console.info(
        `[frontend-review] Pack ${accountId}: using static fallback without display (${fallbackSource.url})`
      );
    }
    return fallbackPack as T;
  }

  const detail = failureMessages.length > 0 ? failureMessages.join('; ') : 'No pack payload found';

  throw new FrontendReviewAccountError(`Pack ${accountId}: ${detail}`, attemptRecords);
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

  const res = await fetch(apiUrl('/api/upload'), { method: 'POST', body: fd });
  if (res.status === 404) {
    throw new Error(
      'Could not reach backend (404 from dev server). Did you configure VITE_API_BASE_URL or Vite proxy?'
    );
  }
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

export interface PollResultResponse {
  ok?: boolean;
  status?: string;
  result?: unknown;
  message?: string;
}

export async function pollResult(
  sessionId: string,
  abortSignal?: AbortSignal
): Promise<PollResultResponse> {
  const url = apiUrl(`/api/result?session_id=${encodeURIComponent(sessionId)}`);
  try {
    const res = await fetch(url, { signal: abortSignal });
    let data: PollResultResponse | null = null;
    try {
      data = (await res.json()) as PollResultResponse;
    } catch (err) {
      if (REVIEW_DEBUG_ENABLED) {
        reviewDebugLog('fetch:parse-error', {
          url,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (res.status === 404) {
      return { ok: true, status: 'processing' };
    }

    if (!res.ok) {
      throw new Error(data?.message || `Result request failed (${res.status})`);
    }

    return data ?? { ok: true, status: 'processing' };
  } catch (err) {
    if (REVIEW_DEBUG_ENABLED) {
      reviewDebugLog('fetch:error', {
        url,
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return { ok: true, status: 'processing' };
  }
}

export async function getAccount(sessionId: string, accountId: string): Promise<any> {
  const res = await fetch(
    apiUrl(`/api/accounts/${encodeURIComponent(sessionId)}/${encodeURIComponent(accountId)}`)
  );
  const data: any = await res.json();
  if (!res.ok || !data?.ok) {
    throw new Error(data?.message || `Get account failed (${res.status})`);
  }
  return data.account;
}

export interface SubmitExplanationsPayload {
  [key: string]: unknown;
}

export async function submitExplanations(payload: SubmitExplanationsPayload): Promise<any> {
  const response = await fetch(apiUrl('/api/submit-explanations'), {
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
  const response = await fetch(apiUrl(`/api/summaries/${encodeURIComponent(sessionId)}`));

  if (!response.ok) {
    throw new Error('Failed to fetch summaries');
  }

  return response.json();
}
