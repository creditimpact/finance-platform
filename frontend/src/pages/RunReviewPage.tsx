import * as React from 'react';
import { Link, useParams } from 'react-router-dom';
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
  submitFrontendReviewAnswers,
  type FrontendReviewPackListingItem,
  type FrontendReviewResponse,
  type RunFrontendManifestResponse,
} from '../api';
import { ReviewPackStoreProvider, useReviewPackStore } from '../stores/reviewPackStore';
import { useToast } from '../components/ToastProvider';

const POLL_INTERVAL_MS = 2000;

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

type Phase = 'idle' | 'loading_manifest' | 'waiting' | 'ready' | 'error';

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
}

function ReviewCardContainer({ accountId, state, onChange, onSubmit, onLoad }: ReviewCardContainerProps) {
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
  const [phase, setPhase] = React.useState<Phase>('idle');
  const [phaseError, setPhaseError] = React.useState<string | null>(null);
  const [manifest, setManifest] = React.useState<RunFrontendManifestResponse | null>(null);
  const [cards, setCards] = React.useState<CardsState>({});
  const [order, setOrder] = React.useState<string[]>([]);

  const isMountedRef = React.useRef(false);
  const loadingRef = React.useRef(false);
  const loadedRef = React.useRef(false);
  const pollTimeoutRef = React.useRef<number | null>(null);
  const eventSourceRef = React.useRef<EventSource | null>(null);
  const packListingRef = React.useRef<Record<string, FrontendReviewPackListingItem & { account_id: string }>>({});
  const loadingAccountsRef = React.useRef<Set<string>>(new Set());

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
    clear();
  }, [sid, clear]);

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

  const loadPackListing = React.useCallback(async () => {
    if (!sid || loadingRef.current || loadedRef.current) {
      return;
    }
    loadingRef.current = true;
    setPhaseError(null);
    if (isMountedRef.current) {
      setPhase((state) => (state === 'ready' ? state : 'waiting'));
    }

    try {
      const { items } = await fetchRunReviewPackListing(sid);
      if (!isMountedRef.current) {
        return;
      }

      const filteredItems = items.filter((item): item is FrontendReviewPackListingItem & { account_id: string } => {
        return typeof item.account_id === 'string' && item.account_id.trim() !== '';
      });

      const listingMap: Record<string, FrontendReviewPackListingItem & { account_id: string }> = {};
      for (const item of filteredItems) {
        listingMap[item.account_id] = item;
      }
      packListingRef.current = listingMap;

      setOrder(filteredItems.map((item) => item.account_id));
      setCards(() => {
        const initial: CardsState = {};
        for (const item of filteredItems) {
          const cached = getPack(item.account_id);
          if (cached) {
            initial[item.account_id] = {
              status: 'ready',
              pack: cached,
              answers: normalizeExistingAnswers((cached as Record<string, unknown> | undefined)?.answers),
              error: null,
              success: Boolean(cached.response),
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

      if (isMountedRef.current) {
        setPhase('ready');
        loadedRef.current = true;
      }
    } catch (err) {
      if (!isMountedRef.current) {
        return;
      }
      const message = err instanceof Error ? err.message : 'Unable to load review packs';
      setPhase('error');
      setPhaseError(message);
    } finally {
      loadingRef.current = false;
    }
  }, [getPack, sid]);

  const loadAccountPack = React.useCallback(
    async (accountId: string) => {
      if (!sid) {
        return;
      }

      const cachedPack = getPack(accountId);
      if (cachedPack) {
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
        return;
      }

      if (loadingAccountsRef.current.has(accountId)) {
        return;
      }

      loadingAccountsRef.current.add(accountId);
      updateCard(accountId, (state) => ({
        ...state,
        status: 'waiting',
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
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          pack,
          answers: normalizeExistingAnswers((pack as Record<string, unknown> | undefined)?.answers),
          error: null,
          success: Boolean(pack.response),
          response: pack.response ?? null,
        }));
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
      } finally {
        loadingAccountsRef.current.delete(accountId);
      }
    },
    [getPack, setPack, sid, updateCard]
  );

  const stopPolling = React.useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  const stopStream = React.useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const schedulePoll = React.useCallback(
    (sessionId: string) => {
      stopPolling();

      const poll = async () => {
        if (!isMountedRef.current) {
          return;
        }
        try {
          const payload = await fetchRunFrontendReviewIndex(sessionId);
          if (!isMountedRef.current) {
            return;
          }
          const packsCount = extractPacksCount(payload);
          if (packsCount > 0) {
            stopPolling();
            await loadPackListing();
            return;
          }
        } catch (err) {
          console.warn('Review poll failed', err);
        }

        pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
      };

      pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
    },
    [loadPackListing, stopPolling]
  );

  const startStream = React.useCallback(
    (sessionId: string) => {
      stopStream();
      try {
        const url = buildFrontendReviewStreamUrl(sessionId);
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;
        if (isMountedRef.current) {
          setPhase((state) => (state === 'ready' ? state : 'waiting'));
        }

        eventSource.addEventListener('packs_ready', async (event) => {
          try {
            if (!isMountedRef.current) {
              return;
            }
            stopPolling();
            await loadPackListing();
          } catch (err) {
            console.error('Failed to load packs after packs_ready', err);
          }
        });

        eventSource.onerror = () => {
          eventSource.close();
          eventSourceRef.current = null;
          if (!isMountedRef.current) {
            return;
          }
          schedulePoll(sessionId);
        };
      } catch (err) {
        console.warn('Unable to open review stream', err);
        schedulePoll(sessionId);
      }
    },
    [loadPackListing, schedulePoll, stopStream]
  );

  React.useEffect(() => {
    if (!sid) {
      setPhase('error');
      setPhaseError('Missing run id.');
      return undefined;
    }

    setPhase('loading_manifest');
    setPhaseError(null);
    setManifest(null);
    setCards({});
    setOrder([]);

    let cancelled = false;

    const init = async () => {
      try {
        const manifestResponse = await fetchRunFrontendManifest(sid);
        if (cancelled || !isMountedRef.current) {
          return;
        }
        setManifest(manifestResponse);

        const reviewStage = manifestResponse.frontend?.review;
        const packsCount = extractPacksCount(reviewStage);
        if (packsCount > 0) {
          await loadPackListing();
        } else {
          startStream(sid);
        }
      } catch (err) {
        if (!isMountedRef.current || cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : 'Unable to load run manifest';
        setPhase('error');
        setPhaseError(message);
        startStream(sid);
      }
    };

    init();

    return () => {
      cancelled = true;
      stopPolling();
      stopStream();
    };
  }, [sid, loadPackListing, schedulePoll, startStream, stopPolling]);

  const handleAnswerChange = React.useCallback(
    (accountId: string, answers: AccountQuestionAnswers) => {
      updateCard(accountId, (state) => ({
        ...state,
        answers,
        status: state.status === 'done' ? 'ready' : state.status,
        error: null,
        success: false,
      }));
    },
    [updateCard]
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
    [cards, sid, updateCard, setPack, showToast]
  );

  const orderedCards = React.useMemo(() => order.map((accountId) => ({ accountId, state: cards[accountId] })), [order, cards]);

  const readyCount = React.useMemo(
    () => orderedCards.filter(({ state }) => state?.status === 'ready' || state?.status === 'done').length,
    [orderedCards]
  );
  const doneCount = React.useMemo(
    () => orderedCards.filter(({ state }) => state?.status === 'done').length,
    [orderedCards]
  );

  const handleCardLoad = React.useCallback(
    (accountId: string) => {
      void loadAccountPack(accountId);
    },
    [loadAccountPack]
  );

  const allDone = orderedCards.length > 0 && doneCount === orderedCards.length;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900">Run review</h1>
        {sid ? <p className="text-sm text-slate-600">Run {sid}</p> : null}
        {manifest?.frontend?.review && readyCount > 0 ? (
          <p className="text-sm text-slate-600">
            {readyCount} {readyCount === 1 ? 'card' : 'cards'} ready for review.
          </p>
        ) : null}
        {phase === 'waiting' ? (
          <p className="text-sm text-slate-500">Waiting for review packs…</p>
        ) : null}
        {phaseError && phase === 'error' ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">{phaseError}</div>
        ) : null}
      </header>

      {orderedCards.length === 0 && phase === 'ready' ? (
        <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
          No review cards available for this run.
        </div>
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
          {doneCount} of {orderedCards.length} cards submitted.
        </p>
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
