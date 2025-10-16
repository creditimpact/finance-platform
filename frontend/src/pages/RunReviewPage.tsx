import * as React from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  type AccountQuestionAnswers,
  type AccountQuestionKey,
} from '../components/AccountQuestions';
import ReviewCard, {
  type ReviewAccountPack,
  type ReviewCardStatus,
} from '../components/ReviewCard';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import {
  buildFrontendReviewStreamUrl,
  fetchFrontendReviewAccount,
  fetchRunFrontendManifest,
  fetchRunFrontendReviewIndex,
  fetchRunReviewPackListing,
  completeFrontendReview,
  submitFrontendReviewAnswers,
} from '../api.ts';
import type {
  FrontendReviewPackListingItem,
  FrontendReviewResponse,
  RunFrontendManifestResponse,
} from '../api.ts';
import { ReviewPackStoreProvider, useReviewPackStore } from '../stores/reviewPackStore';
import { useToast } from '../components/ToastProvider';
import { REVIEW_DEBUG_ENABLED, reviewDebugLog } from '../utils/reviewDebug';

const POLL_INTERVAL_MS = 2000;
const WORKER_HINT_DELAY_MS = 30_000;

type CardStatus = ReviewCardStatus;

interface CardState {
  status: CardStatus;
  pack: ReviewAccountPack | null;
  answers: AccountQuestionAnswers;
  error: string | null;
  success: boolean;
  response: FrontendReviewResponse | null;
}

type CardsState = Record<string, CardState>;

type Phase = 'loading' | 'waiting' | 'ready' | 'error';

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return null;
}

function extractPacksCount(source: unknown): number {
  if (!source || typeof source !== 'object') {
    return 0;
  }
  const record = source as Record<string, unknown>;
  const direct = toNumber(record.packs_count);
  if (direct !== null) {
    return direct;
  }
  const counts = record.counts;
  if (counts && typeof counts === 'object') {
    const value = toNumber((counts as Record<string, unknown>).packs);
    if (value !== null) {
      return value;
    }
  }
  return 0;
}

function cleanAnswers(answers: AccountQuestionAnswers): Record<string, string> {
  const result: Record<string, string> = {};
  (['ownership', 'recognize', 'explanation', 'identity_theft'] as AccountQuestionKey[]).forEach((key) => {
    const value = answers[key];
    if (typeof value === 'string' && value.trim() !== '') {
      result[key] = value;
    }
  });
  return result;
}

function normalizeExistingAnswers(source: unknown): AccountQuestionAnswers {
  if (!source || typeof source !== 'object') {
    return {};
  }
  const record = source as Record<string, unknown>;
  const normalized: AccountQuestionAnswers = {};
  (['ownership', 'recognize', 'explanation', 'identity_theft'] as AccountQuestionKey[]).forEach((key) => {
    const value = record[key];
    if (typeof value === 'string' && value.trim() !== '') {
      normalized[key] = value;
    }
  });
  return normalized;
}

function createInitialCardState(): CardState {
  return {
    status: 'idle',
    pack: null,
    answers: {},
    error: null,
    success: false,
    response: null,
  };
}

interface ReviewCardContainerProps {
  accountId: string;
  state: CardState;
  onChange: (answers: AccountQuestionAnswers) => void;
  onSubmit: () => void;
  onLoad: (accountId: string) => void;
  onRetry: (accountId: string) => void;
}

function ReviewCardContainer({ accountId, state, onChange, onSubmit, onLoad, onRetry }: ReviewCardContainerProps) {
  const cardRef = React.useRef<HTMLDivElement | null>(null);
  const hasRequestedRef = React.useRef(false);

  React.useEffect(() => {
    if (state.status !== 'waiting') {
      hasRequestedRef.current = false;
      return undefined;
    }

    if (hasRequestedRef.current) {
      return undefined;
    }

    const node = cardRef.current;
    if (!node) {
      return undefined;
    }

    if (typeof window !== 'undefined' && typeof window.IntersectionObserver === 'function') {
      const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !hasRequestedRef.current) {
            hasRequestedRef.current = true;
            onLoad(accountId);
          }
        }
      }, { rootMargin: '0px 0px 200px 0px' });

      observer.observe(node);
      return () => {
        observer.disconnect();
      };
    }

    hasRequestedRef.current = true;
    onLoad(accountId);
    return undefined;
  }, [accountId, onLoad, state.status]);

  if (state.status === 'waiting') {
    return (
      <Card ref={cardRef} className="w-full">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-slate-900">Account {accountId}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="h-5 w-1/2 animate-pulse rounded bg-slate-200" />
          <div className="h-40 animate-pulse rounded bg-slate-100" />
        </CardContent>
      </Card>
    );
  }

  if (!state.pack) {
    return (
      <Card ref={cardRef} className="w-full">
        <CardHeader className="border-b border-slate-100 pb-4">
          <CardTitle className="text-xl font-semibold text-slate-900">Account {accountId}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
            {state.error ?? 'Unable to load account details.'}
          </div>
          <button
            type="button"
            onClick={() => onRetry(accountId)}
            className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-400 hover:text-slate-900"
          >
            Retry
          </button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div ref={cardRef} className="w-full">
      <ReviewCard
        accountId={accountId}
        pack={state.pack}
        answers={state.answers}
        status={state.status}
        error={state.error}
        success={state.success}
        onAnswersChange={onChange}
        onSubmit={onSubmit}
      />
    </div>
  );
}

function RunReviewPageContent({ sid }: { sid: string | undefined }) {
  const { getPack, setPack, clear } = useReviewPackStore();
  const { showToast } = useToast();
  const navigate = useNavigate();
  const [phase, setPhase] = React.useState<Phase>('loading');
  const [phaseError, setPhaseError] = React.useState<string | null>(null);
  const [manifest, setManifest] = React.useState<RunFrontendManifestResponse | null>(null);
  const [cards, setCards] = React.useState<CardsState>({});
  const [order, setOrder] = React.useState<string[]>([]);
  const [submittedAccounts, setSubmittedAccounts] = React.useState<Set<string>>(() => new Set());
  const [isCompleting, setIsCompleting] = React.useState(false);
  const [showWorkerHint, setShowWorkerHint] = React.useState(false);
  const [liveUpdatesUnavailable, setLiveUpdatesUnavailable] = React.useState(false);
  const [frontendMissing, setFrontendMissing] = React.useState(false);

  const isMountedRef = React.useRef(false);
  const loadingRef = React.useRef(false);
  const loadedRef = React.useRef(false);
  const pollTimeoutRef = React.useRef<number | null>(null);
  const pollIterationRef = React.useRef(0);
  const eventSourceRef = React.useRef<EventSource | null>(null);
  const packListingRef = React.useRef<Record<string, FrontendReviewPackListingItem & { account_id: string }>>({});
  const loadingAccountsRef = React.useRef<Set<string>>(new Set());
  const workerHintTimeoutRef = React.useRef<number | null>(null);
  const workerWaitingRef = React.useRef(false);
  const retryAttemptsRef = React.useRef<Record<string, number>>({});
  const retryTimeoutsRef = React.useRef<Record<string, number | undefined>>({});
  const hasShownLiveUpdateToastRef = React.useRef(false);

  React.useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  React.useEffect(() => {
    loadedRef.current = false;
    loadingRef.current = false;
    packListingRef.current = {};
    loadingAccountsRef.current = new Set();
    setSubmittedAccounts(new Set());
    clear();
    workerWaitingRef.current = false;
    if (workerHintTimeoutRef.current !== null) {
      window.clearTimeout(workerHintTimeoutRef.current);
      workerHintTimeoutRef.current = null;
    }
    if (isMountedRef.current) {
      setShowWorkerHint(false);
    }
    for (const key of Object.keys(retryTimeoutsRef.current)) {
      const timeoutId = retryTimeoutsRef.current[key];
      if (typeof timeoutId === 'number') {
        window.clearTimeout(timeoutId);
      }
      delete retryTimeoutsRef.current[key];
    }
    retryAttemptsRef.current = {};
    setFrontendMissing(false);
    setLiveUpdatesUnavailable(false);
    hasShownLiveUpdateToastRef.current = false;
    if (isMountedRef.current) {
      setPhase('loading');
      setPhaseError(null);
    }
  }, [sid, clear]);

  const clearWorkerWait = React.useCallback(() => {
    workerWaitingRef.current = false;
    if (workerHintTimeoutRef.current !== null) {
      window.clearTimeout(workerHintTimeoutRef.current);
      workerHintTimeoutRef.current = null;
    }
    if (isMountedRef.current) {
      setShowWorkerHint(false);
    }
  }, []);

  const beginWorkerWait = React.useCallback(() => {
    workerWaitingRef.current = true;
    if (showWorkerHint) {
      return;
    }
    if (workerHintTimeoutRef.current !== null) {
      return;
    }
    workerHintTimeoutRef.current = window.setTimeout(() => {
      workerHintTimeoutRef.current = null;
      if (isMountedRef.current && workerWaitingRef.current) {
        setShowWorkerHint(true);
      }
    }, WORKER_HINT_DELAY_MS);
  }, [showWorkerHint]);

  const loadManifestInfo = React.useCallback(
    async (sessionId: string, options?: { isCancelled?: () => boolean }) => {
      const isCancelled = options?.isCancelled ?? (() => false);
      try {
        const manifestResponse = await fetchRunFrontendManifest(sessionId);
        if (isCancelled() || !isMountedRef.current) {
          return;
        }
        setManifest(manifestResponse);
        const frontendSection = manifestResponse.frontend;
        setFrontendMissing(!(frontendSection && typeof frontendSection === 'object'));
      } catch (err) {
        if (isCancelled()) {
          return;
        }
        console.warn('Unable to load run manifest', err);
      }
    },
    []
  );

  const clearRetryTimeout = React.useCallback((accountId: string) => {
    const timeoutId = retryTimeoutsRef.current[accountId];
    if (typeof timeoutId === 'number') {
      window.clearTimeout(timeoutId);
    }
    delete retryTimeoutsRef.current[accountId];
  }, []);

  const updateCard = React.useCallback((accountId: string, updater: (state: CardState) => CardState) => {
    setCards((previous) => {
      const current = previous[accountId] ?? createInitialCardState();
      const next = updater(current);
      if (next === current) {
        return previous;
      }
      return { ...previous, [accountId]: next };
    });
  }, []);

  const markSubmitted = React.useCallback((accountId: string) => {
    setSubmittedAccounts((previous) => {
      if (previous.has(accountId)) {
        return previous;
      }
      const next = new Set(previous);
      next.add(accountId);
      return next;
    });
  }, []);

  const markUnsubmitted = React.useCallback((accountId: string) => {
    setSubmittedAccounts((previous) => {
      if (!previous.has(accountId)) {
        return previous;
      }
      const next = new Set(previous);
      next.delete(accountId);
      return next;
    });
  }, []);

  const loadPackListing = React.useCallback(async () => {
    if (!sid || loadingRef.current || loadedRef.current) {
      return;
    }
    reviewDebugLog('loadPackListing:start', { sid });
    loadingRef.current = true;
    setPhaseError(null);
    if (isMountedRef.current) {
      setPhase((state) => (state === 'ready' ? state : 'waiting'));
    }

    try {
      const packsUrl = `/api/runs/${encodeURIComponent(sid)}/frontend/review/packs`;
      reviewDebugLog('loadPackListing:fetch', { url: packsUrl });
      const { items } = await fetchRunReviewPackListing(sid);
      if (!isMountedRef.current) {
        return;
      }

      const filteredItems = items.filter((item): item is FrontendReviewPackListingItem & { account_id: string } => {
        return typeof item.account_id === 'string' && item.account_id.trim() !== '';
      });

      reviewDebugLog('loadPackListing:received', {
        url: packsUrl,
        count: filteredItems.length,
      });

      const listingMap: Record<string, FrontendReviewPackListingItem & { account_id: string }> = {};
      for (const item of filteredItems) {
        listingMap[item.account_id] = item;
      }
      packListingRef.current = listingMap;

      setOrder(filteredItems.map((item) => item.account_id));
      const initialSubmitted = new Set<string>();
      setCards(() => {
        const initial: CardsState = {};
        for (const item of filteredItems) {
          const cached = getPack(item.account_id);
          if (cached) {
            const normalizedAnswers = normalizeExistingAnswers((cached as Record<string, unknown> | undefined)?.answers);
            const hasResponse = Boolean(cached.response);
            if (hasResponse) {
              initialSubmitted.add(item.account_id);
            }
            initial[item.account_id] = {
              status: 'ready',
              pack: cached,
              answers: normalizedAnswers,
              error: null,
              success: hasResponse,
              response: cached.response ?? null,
            };
          } else {
            initial[item.account_id] = {
              status: 'waiting',
              pack: null,
              answers: {},
              error: null,
              success: false,
              response: null,
            };
          }
        }
        return initial;
      });
      setSubmittedAccounts(initialSubmitted);
      clearWorkerWait();

      if (isMountedRef.current) {
        setPhase('ready');
        loadedRef.current = true;
      }
      reviewDebugLog('loadPackListing:ready', { count: filteredItems.length });
    } catch (err) {
      if (!isMountedRef.current) {
        return;
      }
      const message = err instanceof Error ? err.message : 'Unable to load review packs';
      clearWorkerWait();
      setPhase('error');
      setPhaseError(message);
    } finally {
      loadingRef.current = false;
    }
  }, [clearWorkerWait, getPack, sid]);

  const bootstrap = React.useCallback(
    async (sessionId: string, options?: { isCancelled?: () => boolean }) => {
      const isCancelled = options?.isCancelled ?? (() => false);
      if (!sessionId) {
        return;
      }
      const indexUrl = `/api/runs/${encodeURIComponent(sessionId)}/frontend/index`;
      reviewDebugLog('bootstrap:fetch', { url: indexUrl, sessionId });
      try {
        const payload = await fetchRunFrontendReviewIndex(sessionId);
        if (isCancelled() || !isMountedRef.current) {
          return;
        }
        const packsCount = extractPacksCount(payload);
        reviewDebugLog('bootstrap:packs-count', { url: indexUrl, packsCount });
        if (packsCount > 0) {
          reviewDebugLog('bootstrap:packs-ready', { url: indexUrl, packsCount });
          clearWorkerWait();
          await loadPackListing();
          return;
        }
        setPhase((state) => (state === 'ready' ? state : 'waiting'));
        beginWorkerWait();
      } catch (err) {
        if (isCancelled() || !isMountedRef.current) {
          return;
        }
        const message = err instanceof Error ? err.message : 'Unable to load review packs';
        clearWorkerWait();
        setPhase('error');
        setPhaseError(message);
        reviewDebugLog('bootstrap:error', { url: indexUrl, error: err });
        console.error(`[RunReviewPage] Failed to load ${indexUrl}`, err);
      }
    },
    [beginWorkerWait, clearWorkerWait, loadPackListing]
  );

  const loadAccountPack = React.useCallback(
    async (accountId: string, options?: { force?: boolean; resetAttempts?: boolean }) => {
      if (!sid) {
        return;
      }

      const cachedPack = getPack(accountId);
      if (cachedPack && !options?.force) {
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          pack: cachedPack,
          answers: state.answers && Object.keys(state.answers).length > 0
            ? state.answers
            : normalizeExistingAnswers((cachedPack as Record<string, unknown> | undefined)?.answers),
          error: null,
          success: Boolean(cachedPack.response),
          response: cachedPack.response ?? null,
        }));
        if (cachedPack.response) {
          markSubmitted(accountId);
        } else {
          markUnsubmitted(accountId);
        }
        return;
      }

      if (loadingAccountsRef.current.has(accountId)) {
        return;
      }

      loadingAccountsRef.current.add(accountId);
      if (options?.resetAttempts) {
        delete retryAttemptsRef.current[accountId];
      }
      clearRetryTimeout(accountId);
      updateCard(accountId, (state) => ({
        ...state,
        status: state.pack && !options?.force ? state.status : 'waiting',
        error: null,
      }));

      try {
        const listing = packListingRef.current[accountId];
        const pack = await fetchFrontendReviewAccount<ReviewAccountPack>(sid, accountId, {
          packPath: typeof listing?.file === 'string' ? listing.file : undefined,
        });
        if (!isMountedRef.current) {
          return;
        }
        setPack(accountId, pack);
        delete retryAttemptsRef.current[accountId];
        clearRetryTimeout(accountId);
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          pack,
          answers: normalizeExistingAnswers((pack as Record<string, unknown> | undefined)?.answers),
          error: null,
          success: Boolean(pack.response),
          response: pack.response ?? null,
        }));
        if (pack.response) {
          markSubmitted(accountId);
        } else {
          markUnsubmitted(accountId);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to load account details';
        if (isMountedRef.current) {
          updateCard(accountId, (state) => ({
            ...state,
            status: 'ready',
            error: message,
            pack: state.pack,
          }));
        }
        const nextAttempt = (retryAttemptsRef.current[accountId] ?? 0) + 1;
        retryAttemptsRef.current[accountId] = nextAttempt;
        const delay = Math.min(30_000, 1_000 * 2 ** (nextAttempt - 1));
        clearRetryTimeout(accountId);
        retryTimeoutsRef.current[accountId] = window.setTimeout(() => {
          delete retryTimeoutsRef.current[accountId];
          if (!isMountedRef.current) {
            return;
          }
          void loadAccountPack(accountId);
        }, delay);
      } finally {
        loadingAccountsRef.current.delete(accountId);
      }
    },
    [clearRetryTimeout, getPack, setPack, sid, updateCard, markSubmitted, markUnsubmitted]
  );

  const stopPolling = React.useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
      reviewDebugLog('poll:stop');
    }
  }, []);

  const stopStream = React.useCallback(() => {
    if (eventSourceRef.current) {
      reviewDebugLog('sse:close-request', {
        readyState: eventSourceRef.current.readyState,
      });
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      reviewDebugLog('sse:closed');
    }
  }, []);

  React.useEffect(() => {
    reviewDebugLog('RunReviewPage session', { sid, debug: REVIEW_DEBUG_ENABLED });
  }, [sid]);

  const schedulePoll = React.useCallback(
    (sessionId: string) => {
      stopPolling();
      reviewDebugLog('poll:schedule', { sessionId });
      pollIterationRef.current = 0;
      if (isMountedRef.current) {
        setPhase((state) => {
          if (state === 'ready' || state === 'error') {
            return state;
          }
          return 'waiting';
        });
      }
      beginWorkerWait();

      const poll = async () => {
        if (!isMountedRef.current) {
          return;
        }
        const iteration = pollIterationRef.current + 1;
        pollIterationRef.current = iteration;
        reviewDebugLog('poll:tick', { sessionId, iteration });
        try {
          const payload = await fetchRunFrontendReviewIndex(sessionId);
          if (!isMountedRef.current) {
            return;
          }
          const packsCount = extractPacksCount(payload);
          reviewDebugLog('poll:packs-count', { sessionId, iteration, packsCount });
          if (packsCount > 0) {
            reviewDebugLog('poll:packs-ready', { sessionId, iteration, packsCount });
            stopPolling();
            clearWorkerWait();
            await loadPackListing();
            return;
          }
          beginWorkerWait();
        } catch (err) {
          reviewDebugLog('poll:error', { sessionId, error: err });
          console.warn('Review poll failed', err);
        }

        pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
        reviewDebugLog('poll:scheduled-next', { sessionId, iteration, delay: POLL_INTERVAL_MS });
      };

      pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
      reviewDebugLog('poll:initial-timeout', { sessionId, delay: POLL_INTERVAL_MS });
    },
    [beginWorkerWait, clearWorkerWait, loadPackListing, stopPolling]
  );

  const startStream = React.useCallback(
    (sessionId: string) => {
      stopStream();
      try {
        const url = buildFrontendReviewStreamUrl(sessionId);
        reviewDebugLog('sse:connect', { url, sessionId });
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;
        if (isMountedRef.current) {
          setPhase((state) => {
            if (state === 'ready' || state === 'loading' || state === 'error') {
              return state;
            }
            return 'waiting';
          });
        }
        beginWorkerWait();

        eventSource.onopen = () => {
          reviewDebugLog('sse:open', { url, sessionId });
        };

        eventSource.addEventListener('packs_ready', async (event) => {
          reviewDebugLog('sse:event', { url, sessionId, type: 'packs_ready', data: event?.data });
          try {
            if (!isMountedRef.current) {
              return;
            }
            stopPolling();
            clearWorkerWait();
            await loadPackListing();
          } catch (err) {
            reviewDebugLog('sse:event-error', { url, sessionId, error: err });
            console.error('Failed to load packs after packs_ready', err);
          }
        });

        eventSource.onerror = () => {
          reviewDebugLog('sse:error', { url, sessionId });
          eventSource.close();
          eventSourceRef.current = null;
          reviewDebugLog('sse:error-closed', { url, sessionId });
          if (!isMountedRef.current) {
            return;
          }
          setLiveUpdatesUnavailable(true);
          if (!hasShownLiveUpdateToastRef.current) {
            hasShownLiveUpdateToastRef.current = true;
            showToast({
              variant: 'warning',
              title: 'Live updates unavailable',
              description: 'Live updates unavailable, falling back to polling…',
            });
          }
          reviewDebugLog('sse:fallback-poll', { sessionId });
          schedulePoll(sessionId);
        };
      } catch (err) {
        reviewDebugLog('sse:connect-error', { sessionId, error: err });
        console.warn('Unable to open review stream', err);
        setLiveUpdatesUnavailable(true);
        if (!hasShownLiveUpdateToastRef.current) {
          hasShownLiveUpdateToastRef.current = true;
          showToast({
            variant: 'warning',
            title: 'Live updates unavailable',
            description: 'Live updates unavailable, falling back to polling…',
          });
        }
        schedulePoll(sessionId);
      }
    },
    [beginWorkerWait, clearWorkerWait, loadPackListing, schedulePoll, showToast, stopStream]
  );

  React.useEffect(() => {
    if (!sid) {
      setPhase('error');
      setPhaseError('Missing run id.');
      return () => {
        stopPolling();
        stopStream();
        clearWorkerWait();
      };
    }

    setPhase('loading');
    setPhaseError(null);
    setManifest(null);
    setCards({});
    setOrder([]);

    let cancelled = false;
    const isCancelled = () => cancelled;

    startStream(sid);
    void loadManifestInfo(sid, { isCancelled });
    void bootstrap(sid, { isCancelled });

    return () => {
      cancelled = true;
      stopPolling();
      stopStream();
      clearWorkerWait();
    };
  }, [sid, bootstrap, loadManifestInfo, startStream, stopPolling, stopStream, clearWorkerWait]);

  const handleAnswerChange = React.useCallback(
    (accountId: string, answers: AccountQuestionAnswers) => {
      updateCard(accountId, (state) => ({
        ...state,
        answers,
        status: state.status === 'done' ? 'ready' : state.status,
        error: null,
        success: false,
      }));
      markUnsubmitted(accountId);
    },
    [markUnsubmitted, updateCard]
  );

  const handleSubmit = React.useCallback(
    async (accountId: string) => {
      const card = cards[accountId];
      if (!sid || !card || card.status === 'saving') {
        return;
      }
      const cleaned = cleanAnswers(card.answers);
      if (Object.keys(cleaned).length === 0) {
        updateCard(accountId, (state) => ({
          ...state,
          error: 'Please provide an answer before submitting.',
        }));
        return;
      }

      updateCard(accountId, (state) => ({
        ...state,
        status: 'saving',
        error: null,
        success: true,
      }));

      try {
        const response = await submitFrontendReviewAnswers(sid, accountId, cleaned);
        let updatedPack: ReviewAccountPack | null = null;
        updateCard(accountId, (state) => {
          const nextPack = state.pack
            ? {
                ...state.pack,
                answers: (response && response.answers) || cleaned,
                response,
              }
            : state.pack;
          updatedPack = nextPack ?? null;
          return {
            ...state,
            status: 'done',
            success: true,
            error: null,
            response: response ?? null,
            pack: nextPack,
          };
        });
        if (updatedPack) {
          setPack(accountId, updatedPack);
        }
        markSubmitted(accountId);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to submit answers';
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          error: message,
          success: false,
        }));
        showToast({
          variant: 'error',
          title: 'Save failed',
          description: message,
        });
      }
    },
    [cards, sid, updateCard, setPack, showToast, markSubmitted]
  );

  React.useEffect(() => {
    reviewDebugLog('cards:rendered', { count: order.length, accounts: order });
  }, [order]);

  const orderedCards = React.useMemo(() => order.map((accountId) => ({ accountId, state: cards[accountId] })), [order, cards]);

  const readyCount = React.useMemo(
    () => orderedCards.filter(({ state }) => state?.status === 'ready' || state?.status === 'done').length,
    [orderedCards]
  );
  const submittedCount = submittedAccounts.size;
  const totalCards = orderedCards.length;

  const handleCardLoad = React.useCallback(
    (accountId: string) => {
      void loadAccountPack(accountId);
    },
    [loadAccountPack]
  );

  const handleRetryLoad = React.useCallback(
    (accountId: string) => {
      void loadAccountPack(accountId, { force: true, resetAttempts: true });
    },
    [loadAccountPack]
  );

  const allDone = totalCards > 0 && submittedCount === totalCards;
  const isLoadingPhase = phase === 'loading' || phase === 'waiting';
  const loaderMessage = phase === 'waiting'
    ? showWorkerHint
      ? 'Waiting for worker…'
      : 'Waiting for review packs…'
    : 'Loading review packs…';
  const showNoCardsMessage = orderedCards.length === 0 && phase !== 'error' && phase !== 'loading';

  const handleFinishReview = React.useCallback(async () => {
    if (!sid) {
      return;
    }
    setIsCompleting(true);
    try {
      await completeFrontendReview(sid);
      navigate(`/runs/${encodeURIComponent(sid)}/review/complete`);
      return;
    } catch (err) {
      console.warn('Failed to complete review', err);
      showToast({
        variant: 'error',
        title: 'Unable to finish review',
        description: err instanceof Error ? err.message : 'Unknown error occurred.',
      });
    } finally {
      setIsCompleting(false);
    }
  }, [sid, navigate, showToast]);

  return (
    <div className={`mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8 ${allDone ? 'pb-32' : ''}`}>
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900">Run review</h1>
        {sid ? <p className="text-sm text-slate-600">Run {sid}</p> : null}
        {manifest?.frontend?.review && readyCount > 0 ? (
          <p className="text-sm text-slate-600">
            {readyCount} {readyCount === 1 ? 'card' : 'cards'} ready for review.
          </p>
        ) : null}
        {liveUpdatesUnavailable ? (
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
            Live updates unavailable, falling back to polling…
          </div>
        ) : null}
        {isLoadingPhase ? (
          <div className="flex items-center gap-2 text-sm text-slate-600" role="status" aria-live="polite">
            <span
              aria-hidden="true"
              className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-transparent"
            />
            <span>{loaderMessage}</span>
          </div>
        ) : null}
        {phaseError && phase === 'error' ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">{phaseError}</div>
        ) : null}
      </header>

      {frontendMissing && sid ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          Frontend manifest block missing for run <span className="font-mono">{sid}</span>. Waiting for worker to publish review
          metadata.
        </div>
      ) : null}

      {showNoCardsMessage ? (
        <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">No review cards yet</div>
      ) : null}

      <div className="space-y-6">
        {orderedCards.map(({ accountId, state }) => (
          <ReviewCardContainer
            key={accountId}
            accountId={accountId}
            state={state ?? createInitialCardState()}
            onChange={(answers) => handleAnswerChange(accountId, answers)}
            onSubmit={() => handleSubmit(accountId)}
            onLoad={handleCardLoad}
            onRetry={handleRetryLoad}
          />
        ))}
      </div>

      {allDone ? (
        <div className="flex flex-col items-start gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-semibold">All set!</p>
            <p className="text-emerald-800">You answered every review card.</p>
          </div>
          {sid ? (
            <Link
              to={`/runs/${encodeURIComponent(sid)}/accounts`}
              className="inline-flex items-center justify-center rounded-md border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-500"
            >
              View accounts
            </Link>
          ) : null}
        </div>
      ) : null}

      {orderedCards.length > 0 ? (
        <p className="text-xs text-slate-500">
          {submittedCount} of {orderedCards.length} cards submitted.
        </p>
      ) : null}

      {allDone ? (
        <div className="fixed bottom-0 left-0 right-0 border-t border-slate-200 bg-white shadow-lg">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
            <div>
              <p className="text-sm font-semibold text-slate-900">All cards submitted</p>
              <p className="text-sm text-slate-600">You can finish the review now.</p>
            </div>
            <button
              type="button"
              onClick={handleFinishReview}
              className="inline-flex items-center justify-center rounded-md border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:border-emerald-300 disabled:bg-emerald-300"
              disabled={isCompleting}
            >
              {isCompleting ? 'Finishing…' : 'Finish review'}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function RunReviewPage() {
  const { sid } = useParams();

  return (
    <ReviewPackStoreProvider>
      <RunReviewPageContent sid={sid} />
    </ReviewPackStoreProvider>
  );
}
