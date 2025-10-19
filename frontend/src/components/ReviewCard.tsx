import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { cn } from '../lib/utils';
import {
  BUREAUS,
  BUREAU_LABELS,
  MISSING_VALUE,
  type BureauKey,
} from './accountFieldTypes';
import { summarizeField, type BureauTriple } from '../utils/bureauSummary';
import { type AccountQuestionAnswers, type ClaimDocuments } from './AccountQuestions';
import type { AccountPack } from './AccountCard';
import { type FrontendReviewResponse, uploadReviewDoc } from '../api.ts';
import { CLAIMS, type ClaimKey } from '../constants/claims';
import { normalizeClaims } from '../utils/reviewClaims';
import ClaimPicker from './ClaimPicker';

export type ReviewCardStatus = 'idle' | 'waiting' | 'ready' | 'saving' | 'done';

type QuestionDescriptor = {
  id?: string | null;
  required?: boolean | string | null;
  prompt?: string | null;
};

export type ReviewAccountPack = AccountPack & {
  account_id?: string;
  questions?: QuestionDescriptor[] | null;
  answers?: Record<string, unknown> | null;
  response?: FrontendReviewResponse | null;
};

type SummaryFieldKey = 'account_number' | 'account_type' | 'status';

type DetailFieldKey =
  | 'account_number'
  | 'account_type'
  | 'status'
  | 'balance_owed'
  | 'date_opened'
  | 'closed_date';

interface BureauFieldConfig<K extends DetailFieldKey> {
  key: K;
  label: string;
  kind?: 'account_number';
}

const SUMMARY_FIELDS: BureauFieldConfig<SummaryFieldKey>[] = [
  { key: 'account_number', label: 'Account number', kind: 'account_number' },
  { key: 'account_type', label: 'Account type' },
  { key: 'status', label: 'Status' },
];

const DETAIL_FIELDS: BureauFieldConfig<DetailFieldKey>[] = [
  { key: 'account_number', label: 'Account number', kind: 'account_number' },
  { key: 'account_type', label: 'Account type' },
  { key: 'status', label: 'Status' },
  { key: 'balance_owed', label: 'Balance owed' },
  { key: 'date_opened', label: 'Date opened' },
  { key: 'closed_date', label: 'Closed date' },
];

const REVIEW_CLAIMS_ENABLED = (import.meta as { env?: Record<string, string | undefined> }).env?.
  VITE_REVIEW_CLAIMS === '1';

type UploadedDocMap = Record<ClaimKey, Record<string, string[]>>;

type PerBureauSource =
  | {
      per_bureau?: Partial<Record<BureauKey, string | null | undefined>>;
    }
  | Partial<Record<BureauKey, string | null | undefined>>
  | null
  | undefined;

function toBureauTriple(source: PerBureauSource): BureauTriple {
  const triple: BureauTriple = {};
  if (!source) {
    return triple;
  }

  const perBureau = (source as { per_bureau?: Partial<Record<BureauKey, string | null | undefined>> }).per_bureau;
  const data = perBureau && typeof perBureau === 'object' ? perBureau : source;

  for (const bureau of BUREAUS) {
    const rawValue = data?.[bureau];
    if (rawValue == null) {
      continue;
    }
    const text = typeof rawValue === 'string' ? rawValue : String(rawValue);
    triple[bureau] = text;
  }

  return triple;
}

function extractPerBureauValues(source: PerBureauSource): Partial<Record<BureauKey, string>> {
  const result: Partial<Record<BureauKey, string>> = {};
  if (!source) {
    return result;
  }

  const perBureau = (source as { per_bureau?: Partial<Record<BureauKey, string | null | undefined>> }).per_bureau;
  const data = perBureau && typeof perBureau === 'object' ? perBureau : source;

  for (const bureau of BUREAUS) {
    const value = data?.[bureau];
    if (value == null) {
      continue;
    }
    result[bureau] = typeof value === 'string' ? value : String(value);
  }

  return result;
}

const MISSING_TOKENS = new Set(['', '--', '—']);

function normalizeDisplayValue(value?: string | null): { text: string; isMissing: boolean } {
  if (value == null) {
    return { text: MISSING_VALUE, isMissing: true };
  }
  const trimmed = value.trim();
  if (MISSING_TOKENS.has(trimmed)) {
    return { text: MISSING_VALUE, isMissing: true };
  }
  return { text: trimmed, isMissing: false };
}

function buildUploadedMapFromDocuments(claimDocuments?: ClaimDocuments): UploadedDocMap {
  const map: UploadedDocMap = {};
  if (!claimDocuments) {
    return map;
  }
  for (const [claimKey, docMap] of Object.entries(claimDocuments)) {
    if (!CLAIMS[claimKey as ClaimKey] || !docMap || typeof docMap !== 'object') {
      continue;
    }
    const perClaimEntries: Record<string, string[]> = {};
    for (const [docKey, docIds] of Object.entries(docMap)) {
      if (!Array.isArray(docIds)) {
        continue;
      }
      const filtered = docIds
        .map((id) => (typeof id === 'string' ? id.trim() : ''))
        .filter((id): id is string => Boolean(id));
      if (filtered.length > 0) {
        perClaimEntries[docKey] = filtered;
      }
    }
    map[claimKey as ClaimKey] = perClaimEntries;
  }
  return map;
}

function toClaimDocuments(map: UploadedDocMap, claims: ClaimKey[]): ClaimDocuments {
  const documents: ClaimDocuments = {};
  for (const claim of claims) {
    const docEntries = map[claim];
    if (!docEntries) {
      continue;
    }
    const cleaned: Record<string, string[]> = {};
    for (const [docKey, docIds] of Object.entries(docEntries)) {
      const filtered = docIds
        .map((id) => (typeof id === 'string' ? id.trim() : ''))
        .filter((id): id is string => Boolean(id));
      if (filtered.length > 0) {
        cleaned[docKey] = filtered;
      }
    }
    if (Object.keys(cleaned).length > 0) {
      documents[claim] = cleaned;
    }
  }
  return documents;
}

function areClaimListsEqual(a: ClaimKey[], b: ClaimKey[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) {
      return false;
    }
  }
  return true;
}

function areUploadedMapsEqual(a: UploadedDocMap, b: UploadedDocMap): boolean {
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) {
    return false;
  }
  for (const claimKey of aKeys) {
    const docMapA = a[claimKey as ClaimKey] ?? {};
    const docMapB = b[claimKey as ClaimKey] ?? {};
    const docKeysA = Object.keys(docMapA);
    const docKeysB = Object.keys(docMapB);
    if (docKeysA.length !== docKeysB.length) {
      return false;
    }
    for (const docKey of docKeysA) {
      const listA = docMapA[docKey] ?? [];
      const listB = docMapB[docKey] ?? [];
      if (listA.length !== listB.length) {
        return false;
      }
      for (let index = 0; index < listA.length; index += 1) {
        if (listA[index] !== listB[index]) {
          return false;
        }
      }
    }
  }
  return true;
}

function formatPrimaryIssue(issue?: string | null): string | null {
  if (!issue) {
    return null;
  }
  const text = issue.replace(/_/g, ' ').trim();
  if (!text) {
    return null;
  }
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatHolderName(holder?: string | null, fallback?: string | null): string {
  const value = holder ?? fallback;
  if (!value) {
    return 'Account holder';
  }
  return value;
}

export interface ReviewCardProps {
  pack: ReviewAccountPack;
  accountId?: string;
  sessionId?: string;
  answers: AccountQuestionAnswers;
  status: ReviewCardStatus;
  error?: string | null;
  success?: boolean;
  onAnswersChange?: (answers: AccountQuestionAnswers) => void;
  onSubmit: () => void;
}

const CheckCircleIcon = ({ className }: { className?: string }) => (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    className={cn('h-4 w-4', className)}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M9 12l2 2 4-4" />
    <circle cx="12" cy="12" r="9" />
  </svg>
);

export function ReviewCard({
  pack,
  accountId,
  sessionId,
  answers,
  status,
  error,
  success,
  onAnswersChange,
  onSubmit,
}: ReviewCardProps) {
  const display = (pack.display ?? {}) as Partial<NonNullable<AccountPack['display']>>;

  const holderName = formatHolderName(pack.holder_name, display.holder_name ?? null);
  const primaryIssue = formatPrimaryIssue(pack.primary_issue ?? display.primary_issue ?? null);
  const explanationId = React.useId();

  const summaryFields = React.useMemo(() => {
    return SUMMARY_FIELDS.map((field) => {
      const triple = toBureauTriple(display[field.key]);
      const summary = summarizeField(
        triple,
        field.kind === 'account_number' ? { kind: 'account_number' } : undefined
      );
      const normalized = normalizeDisplayValue(summary.summary);
      return {
        key: field.key,
        label: field.label,
        value: normalized.text,
        isMissing: normalized.isMissing,
      };
    });
  }, [display]);

  const detailFields = React.useMemo(() => {
    return DETAIL_FIELDS.map((field) => ({
      key: field.key,
      label: field.label,
      values: extractPerBureauValues(display[field.key]),
    }));
  }, [display]);

  const claimsEnabled = Boolean(REVIEW_CLAIMS_ENABLED);
  const effectiveAccountId = accountId ?? pack.account_id ?? '';
  const [selectedClaims, setSelectedClaims] = React.useState<ClaimKey[]>(() =>
    claimsEnabled ? normalizeClaims(answers.claims) : []
  );
  const [uploadedMap, setUploadedMap] = React.useState<UploadedDocMap>(() =>
    claimsEnabled ? buildUploadedMapFromDocuments(answers.claimDocuments) : {}
  );

  const syncAnswers = React.useCallback(
    (claims: ClaimKey[], map: UploadedDocMap) => {
      if (!onAnswersChange) {
        return;
      }
      const nextAnswers: AccountQuestionAnswers = { ...answers };
      if (claims.length > 0) {
        nextAnswers.claims = claims;
      } else {
        delete nextAnswers.claims;
      }
      const documents = toClaimDocuments(map, claims);
      if (Object.keys(documents).length > 0) {
        nextAnswers.claimDocuments = documents;
      } else {
        delete nextAnswers.claimDocuments;
      }
      onAnswersChange(nextAnswers);
    },
    [answers, onAnswersChange]
  );

  React.useEffect(() => {
    if (!claimsEnabled) {
      setSelectedClaims([]);
      setUploadedMap({});
      return;
    }
    const normalized = normalizeClaims(answers.claims);
    setSelectedClaims((previous) =>
      areClaimListsEqual(previous, normalized) ? previous : normalized
    );
  }, [answers.claims, claimsEnabled]);

  React.useEffect(() => {
    if (!claimsEnabled) {
      setUploadedMap({});
      return;
    }
    const nextMap = buildUploadedMapFromDocuments(answers.claimDocuments);
    setUploadedMap((previous) => (areUploadedMapsEqual(previous, nextMap) ? previous : nextMap));
  }, [answers.claimDocuments, claimsEnabled]);

  const handleClaimsChange = React.useCallback(
    (nextClaims: ClaimKey[]) => {
      if (!claimsEnabled) {
        return;
      }
      setSelectedClaims((previous) =>
        areClaimListsEqual(previous, nextClaims) ? previous : [...nextClaims]
      );
      setUploadedMap((previous) => {
        const nextMap: UploadedDocMap = {};
        for (const claim of nextClaims) {
          nextMap[claim] = { ...(previous[claim] ?? {}) };
        }
        syncAnswers(nextClaims, nextMap);
        return nextMap;
      });
    },
    [claimsEnabled, syncAnswers]
  );

  const queueUpload = React.useCallback(
    async (claim: ClaimKey, docKey: string, files: File[]) => {
      if (!claimsEnabled) {
        return;
      }
      const filteredFiles = files.filter((file): file is File => Boolean(file));
      if (filteredFiles.length === 0) {
        return;
      }
      if (!sessionId || !effectiveAccountId) {
        console.error('Missing session or account id for upload.');
        return;
      }
      try {
        const response = await uploadReviewDoc(
          sessionId,
          effectiveAccountId,
          claim,
          docKey,
          filteredFiles
        );
        const docIds = Array.isArray(response?.doc_ids)
          ? response.doc_ids
              .map((id) => (typeof id === 'string' ? id.trim() : ''))
              .filter((id): id is string => Boolean(id))
          : [];
        if (docIds.length === 0) {
          return;
        }
        const baseClaims = selectedClaims.includes(claim)
          ? selectedClaims
          : [...selectedClaims, claim];
        setSelectedClaims((previous) =>
          areClaimListsEqual(previous, baseClaims) ? previous : [...baseClaims]
        );
        setUploadedMap((previous) => {
          const prevClaimDocs = previous[claim] ?? {};
          const existingDocIds = prevClaimDocs[docKey] ?? [];
          const mergedDocIds = Array.from(new Set([...existingDocIds, ...docIds]));
          const nextClaimDocs = { ...prevClaimDocs, [docKey]: mergedDocIds };
          const nextMap: UploadedDocMap = { ...previous, [claim]: nextClaimDocs };
          for (const claimKey of baseClaims) {
            if (!nextMap[claimKey]) {
              nextMap[claimKey] = {};
            }
          }
          syncAnswers(baseClaims, nextMap);
          return nextMap;
        });
      } catch (uploadError) {
        console.error('Failed to upload review document', uploadError);
      }
    },
    [
      claimsEnabled,
      effectiveAccountId,
      selectedClaims,
      sessionId,
      syncAnswers,
    ]
  );

  const requiresDocsSatisfied = React.useMemo(() => {
    if (!claimsEnabled) {
      return true;
    }
    return selectedClaims.every((claimKey) => {
      const definition = CLAIMS[claimKey];
      if (!definition?.requiresDocs) {
        return true;
      }
      const entries = uploadedMap[claimKey] ?? {};
      return definition.requiredDocs.every((docKey) => {
        const docIds = entries?.[docKey];
        return Array.isArray(docIds) && docIds.length > 0;
      });
    });
  }, [claimsEnabled, selectedClaims, uploadedMap]);

  const [detailsOpen, setDetailsOpen] = React.useState(false);

  const explanationValue = answers.explanation ?? '';
  const trimmedExplanation =
    typeof explanationValue === 'string' ? explanationValue.trim() : '';
  const hasExplanation = trimmedExplanation.length > 0;
  const hasSelectedClaims = selectedClaims.length > 0;
  const canSubmitContent = hasExplanation || hasSelectedClaims;
  const disableBecauseOfStatus = status === 'saving' || status === 'waiting';
  const submitDisabled =
    disableBecauseOfStatus || !!error || !canSubmitContent || !requiresDocsSatisfied;

  const handleExplanationChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextValue = event.target.value;
      if (onAnswersChange) {
        onAnswersChange({
          ...answers,
          explanation: nextValue,
        });
      }
    },
    [answers, onAnswersChange]
  );

  const handleSubmit = React.useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      if (submitDisabled) {
        return;
      }
      onSubmit();
    },
    [onSubmit, submitDisabled]
  );

  const buttonLabel = success ? 'Saved' : status === 'saving' ? 'Saving…' : 'Submit';

  return (
    <Card className="w-full">
      <CardHeader className="border-b border-slate-100 pb-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl font-semibold text-slate-900">{holderName}</CardTitle>
            {primaryIssue ? (
              <p className="text-sm uppercase tracking-wide text-slate-500">{primaryIssue}</p>
            ) : null}
          </div>
          <div className="flex flex-col items-end gap-2">
            {success ? (
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
                <CheckCircleIcon className="text-emerald-600" /> Saved
              </span>
            ) : null}
            {accountId ? (
              <span className="text-xs font-medium uppercase tracking-wide text-slate-400">Account {accountId}</span>
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6 pt-6">
        <div className="grid gap-4 sm:grid-cols-3">
          {summaryFields.map((field) => (
            <div key={field.key} className="rounded-md border border-slate-200 p-4">
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{field.label}</p>
              <p
                className={cn(
                  'mt-2 text-sm font-semibold',
                  field.isMissing ? 'text-slate-400' : 'text-slate-900'
                )}
              >
                {field.value}
              </p>
            </div>
          ))}
        </div>

        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setDetailsOpen((previous) => !previous)}
            className="text-sm font-medium text-slate-700 transition hover:text-slate-900"
            aria-expanded={detailsOpen}
          >
            {detailsOpen ? 'Hide bureau details' : 'Show bureau details'}
          </button>
          {detailsOpen ? (
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th scope="col" className="px-4 py-3 font-medium text-slate-500">
                      Field
                    </th>
                    {BUREAUS.map((bureau) => (
                      <th key={bureau} scope="col" className="px-4 py-3 text-center font-medium text-slate-500">
                        {BUREAU_LABELS[bureau]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {detailFields.map((field) => (
                    <tr key={field.key} className="bg-white">
                      <th scope="row" className="px-4 py-3 text-sm font-medium text-slate-700">
                        {field.label}
                      </th>
                      {BUREAUS.map((bureau) => {
                        const { text, isMissing } = normalizeDisplayValue(field.values[bureau]);
                        return (
                          <td
                            key={bureau}
                            className={cn(
                              'px-4 py-3 text-center text-sm',
                              isMissing ? 'text-slate-400' : 'text-slate-900'
                            )}
                          >
                            {text}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>

        {claimsEnabled ? (
          <ClaimPicker
            selected={selectedClaims}
            onChange={handleClaimsChange}
            onFilesSelected={queueUpload}
            uploadedMap={uploadedMap}
          />
        ) : null}

        <div className="space-y-3">
          <div className="space-y-1">
            <label htmlFor={explanationId} className="block text-base font-semibold text-slate-900">
              Explain
            </label>
            <p className="text-sm text-slate-600">
              Share a brief explanation to help us understand this account.
            </p>
          </div>
          <textarea
            id={explanationId}
            name="explanation"
            value={explanationValue}
            onChange={handleExplanationChange}
            disabled={disableBecauseOfStatus}
            required
            rows={5}
            className="block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500"
          />
        </div>

        {error ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">{error}</div>
        ) : null}

        {success && status !== 'ready' ? (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
            Explanation saved successfully.
          </div>
        ) : null}

        <div>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitDisabled}
            className={cn(
              'inline-flex w-full items-center justify-center rounded-md px-4 py-2 text-sm font-semibold shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 sm:w-auto',
              submitDisabled
                ? 'cursor-not-allowed border border-slate-200 bg-slate-100 text-slate-400'
                : 'border border-slate-900 bg-slate-900 text-white hover:bg-slate-800'
            )}
          >
            {buttonLabel}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

export default ReviewCard;
