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
  type RunFrontendManifestResponse,
} from '../api';

const POLL_INTERVAL_MS = 2000;

type CardStatus = ReviewCardStatus;

interface CardState {
  status: CardStatus;
  pack: ReviewAccountPack | null;
  answers: AccountQuestionAnswers;
  error: string | null;
  success: boolean;
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
  };
}

interface ReviewCardContainerProps {
  accountId: string;
  state: CardState;
  onChange: (answers: AccountQuestionAnswers) => void;
  onSubmit: () => void;
}

function ReviewCardContainer({ accountId, state, onChange, onSubmit }: ReviewCardContainerProps) {
  if (state.status === 'waiting') {
    return (
      <Card className="w-full">
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
      <Card className="w-full">
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
  );
}

export default function RunReviewPage() {
  const { sid } = useParams();
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

  React.useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  React.useEffect(() => {
    loadedRef.current = false;
    loadingRef.current = false;
  }, [sid]);

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

  const loadPacks = React.useCallback(async () => {
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

      setOrder(filteredItems.map((item) => item.account_id));
      setCards(() => {
        const initial: CardsState = {};
        for (const item of filteredItems) {
          initial[item.account_id] = {
            status: 'waiting',
            pack: null,
            answers: {},
            error: null,
            success: false,
          };
        }
        return initial;
      });

      await Promise.all(
        filteredItems.map(async (item) => {
          try {
            const pack = await fetchFrontendReviewAccount<ReviewAccountPack>(sid, item.account_id, {
              packPath: typeof item.file === 'string' ? item.file : undefined,
            });

            if (!isMountedRef.current) {
              return;
            }

            updateCard(item.account_id, (state) => ({
              ...state,
              status: 'ready',
              pack,
              answers: normalizeExistingAnswers((pack as Record<string, unknown> | undefined)?.answers),
              error: null,
              success: false,
            }));
          } catch (err) {
            if (!isMountedRef.current) {
              return;
            }
            const message = err instanceof Error ? err.message : 'Unable to load account details';
            updateCard(item.account_id, (state) => ({
              ...state,
              status: 'ready',
              error: message,
              pack: state.pack,
            }));
          }
        })
      );

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
  }, [sid, updateCard]);

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
            await loadPacks();
            return;
          }
        } catch (err) {
          console.warn('Review poll failed', err);
        }

        pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
      };

      pollTimeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
    },
    [loadPacks, stopPolling]
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
            await loadPacks();
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
    [loadPacks, schedulePoll, stopStream]
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
          await loadPacks();
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
  }, [sid, loadPacks, schedulePoll, startStream, stopPolling]);

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

      updateCard(accountId, (state) => ({ ...state, status: 'saving', error: null, success: false }));

      try {
        await submitFrontendReviewAnswers(sid, accountId, cleaned);
        updateCard(accountId, (state) => ({ ...state, status: 'done', success: true }));
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to submit answers';
        updateCard(accountId, (state) => ({
          ...state,
          status: 'ready',
          error: message,
          success: false,
        }));
      }
    },
    [cards, sid, updateCard]
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
