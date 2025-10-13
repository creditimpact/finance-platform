import * as React from 'react';
import { useParams } from 'react-router-dom';
import AccountCard, { type AccountPack } from '../components/AccountCard';
import AccountCardSkeleton from '../components/AccountCardSkeleton';
import {
  fetchRunAccountPack,
  fetchRunFrontendIndex,
  type FrontendAccountIndexEntry,
} from '../api';

const DEFAULT_SKELETON_COUNT = 3;

type LoadedAccount = {
  accountId: string;
  pack: AccountPack;
};

function normalizeSearchTerm(term: string): string {
  return term.trim().toLowerCase();
}

function matchesSearch(pack: AccountPack, normalizedQuery: string): boolean {
  if (!normalizedQuery) {
    return true;
  }

  const holder = (pack.holder_name ?? '').toLowerCase();
  const issueRaw = pack.primary_issue ?? '';
  const issue = issueRaw.toLowerCase();
  const issueReadable = issueRaw.replace(/_/g, ' ').toLowerCase();

  return (
    holder.includes(normalizedQuery) ||
    issue.includes(normalizedQuery) ||
    issueReadable.includes(normalizedQuery)
  );
}

function hasPackPath(entry: FrontendAccountIndexEntry): entry is FrontendAccountIndexEntry & {
  pack_path: string;
} {
  return typeof entry.pack_path === 'string' && entry.pack_path.length > 0;
}

export default function AccountsPage() {
  const { sid } = useParams();
  const [accounts, setAccounts] = React.useState<LoadedAccount[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [searchTerm, setSearchTerm] = React.useState('');
  const [skeletonCount, setSkeletonCount] = React.useState(DEFAULT_SKELETON_COUNT);
  const [reloadToken, setReloadToken] = React.useState(0);

  React.useEffect(() => {
    if (!sid) {
      setError('Missing session id');
      setAccounts([]);
      setLoading(false);
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);
    setAccounts([]);
    setSkeletonCount(DEFAULT_SKELETON_COUNT);

    (async () => {
      try {
        const indexPayload = await fetchRunFrontendIndex(sid);
        if (!active) {
          return;
        }

        const entries = (indexPayload.accounts ?? []).filter(hasPackPath);
        setSkeletonCount(entries.length > 0 ? entries.length : DEFAULT_SKELETON_COUNT);

        const packs = await Promise.all(
          entries.map(async (entry) => {
            try {
              return await fetchRunAccountPack<AccountPack>(sid, entry.pack_path);
            } catch (err) {
              const detail = err instanceof Error ? err.message : String(err);
              throw new Error(`Failed to load account ${entry.account_id}: ${detail}`);
            }
          })
        );

        if (!active) {
          return;
        }

        const combined = entries.map<LoadedAccount>((entry, index) => ({
          accountId: entry.account_id,
          pack: packs[index],
        }));

        setAccounts(combined);
        setLoading(false);
      } catch (err) {
        if (!active) {
          return;
        }
        console.error('Failed to load account packs', err);
        setError(err instanceof Error ? err.message : 'Unable to load accounts');
        setLoading(false);
      }
    })();

    return () => {
      active = false;
    };
  }, [sid, reloadToken]);

  const normalizedSearch = React.useMemo(() => normalizeSearchTerm(searchTerm), [searchTerm]);

  const filteredAccounts = React.useMemo(() => {
    if (!normalizedSearch) {
      return accounts;
    }
    return accounts.filter((account) => matchesSearch(account.pack, normalizedSearch));
  }, [accounts, normalizedSearch]);

  const totalCount = accounts.length;
  const visibleCount = filteredAccounts.length;
  const showEmptyState = !loading && !error && totalCount === 0;
  const showFilteredEmpty = !loading && !error && totalCount > 0 && visibleCount === 0;

  const handleRetry = () => setReloadToken((token) => token + 1);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold text-slate-900">Accounts</h1>
          {sid ? (
            <p className="text-sm text-slate-600">Run {sid}</p>
          ) : (
            <p className="text-sm text-slate-600">No session selected</p>
          )}
          <p className="text-sm text-slate-600">
            Compare bureau data at a glance and expand to see per-bureau details.
          </p>
        </div>
        <div className="flex w-full max-w-sm flex-col gap-2">
          <label htmlFor="account-search" className="text-sm font-medium text-slate-700">
            Search accounts
          </label>
          <input
            id="account-search"
            type="search"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search by holder or issue"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-200"
          />
          {totalCount > 0 && !loading ? (
            <span className="text-xs text-slate-500">
              Showing {visibleCount} of {totalCount} accounts
            </span>
          ) : null}
        </div>
      </header>

      {error ? (
        <div
          role="alert"
          className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900"
        >
          <p className="font-semibold">Unable to load accounts</p>
          <p className="mt-1 text-rose-800">{error}</p>
          <button
            type="button"
            onClick={handleRetry}
            className="mt-3 inline-flex items-center justify-center rounded-md border border-rose-200 bg-white px-3 py-1.5 text-sm font-semibold text-rose-900 shadow-sm transition hover:border-rose-300 hover:bg-rose-100"
          >
            Try again
          </button>
        </div>
      ) : null}

      {loading ? (
        <div className="space-y-4">
          {Array.from({ length: Math.max(skeletonCount, DEFAULT_SKELETON_COUNT) }).map((_, index) => (
            <AccountCardSkeleton key={index} />
          ))}
        </div>
      ) : null}

      {!loading && !error ? (
        <div className="space-y-6">
          {filteredAccounts.map((account) => (
            <AccountCard key={account.accountId} pack={account.pack} />
          ))}
        </div>
      ) : null}

      {showEmptyState ? (
        <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
          No accounts found for this run.
        </div>
      ) : null}

      {showFilteredEmpty ? (
        <div className="rounded-lg border border-slate-200 bg-white p-6 text-sm text-slate-600">
          No accounts match your search.
        </div>
      ) : null}
    </div>
  );
}
