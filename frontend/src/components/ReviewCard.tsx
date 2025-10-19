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
import {
  uploadFrontendReviewEvidence,
  type FrontendReviewResponse,
  type FrontendReviewUploadDocInfo,
} from '../api.ts';
import { BASIC_ALWAYS_INCLUDED, CLAIMS, type ClaimKey } from '../constants/claims';
import { shouldEnableReviewClaims } from '../config/featureFlags';
import { getMissingRequiredDocs, hasMissingRequiredDocs, formatDocKey } from '../utils/reviewClaims';

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

const REVIEW_CLAIMS_ENABLED = shouldEnableReviewClaims();

type ClaimDocMetadata = Partial<Record<ClaimKey, Partial<Record<string, FrontendReviewUploadDocInfo[]>>>>;

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

function createMetadataFromDocuments(claimDocuments?: ClaimDocuments): ClaimDocMetadata {
  const metadata: ClaimDocMetadata = {};
  if (!claimDocuments) {
    return metadata;
  }
  for (const [claimKey, docMap] of Object.entries(claimDocuments)) {
    if (!CLAIMS[claimKey as ClaimKey] || !docMap) {
      continue;
    }
    const perClaim: Partial<Record<string, FrontendReviewUploadDocInfo[]>> = {};
    for (const [docKey, docIds] of Object.entries(docMap)) {
      if (!Array.isArray(docIds)) {
        continue;
      }
      const entries: FrontendReviewUploadDocInfo[] = [];
      for (const id of docIds) {
        if (typeof id !== 'string' || id.trim() === '') {
          continue;
        }
        entries.push({ id, claim: claimKey, doc_key: docKey });
      }
      if (entries.length > 0) {
        perClaim[docKey] = entries;
      }
    }
    if (Object.keys(perClaim).length > 0) {
      metadata[claimKey as ClaimKey] = perClaim;
    }
  }
  return metadata;
}

function dedupeDocInfos(values: FrontendReviewUploadDocInfo[]): FrontendReviewUploadDocInfo[] {
  const seen = new Set<string>();
  const result: FrontendReviewUploadDocInfo[] = [];
  for (const entry of values) {
    const id = typeof entry.id === 'string' ? entry.id : '';
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    result.push(entry);
  }
  return result;
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

  const claimsEnabled = REVIEW_CLAIMS_ENABLED;
  const effectiveAccountId = accountId ?? pack.account_id ?? '';
  const selectedClaims = React.useMemo<ClaimKey[]>(() => {
    if (!claimsEnabled || !Array.isArray(answers.claims)) {
      return [];
    }
    return answers.claims.filter((claim): claim is ClaimKey => Boolean(claim && CLAIMS[claim as ClaimKey]));
  }, [answers.claims, claimsEnabled]);
  const claimDocuments = answers.claimDocuments;
  const missingDocsMap = React.useMemo(() => {
    if (!claimsEnabled) {
      return {} as Partial<Record<ClaimKey, string[]>>;
    }
    return getMissingRequiredDocs(selectedClaims, claimDocuments);
  }, [claimsEnabled, selectedClaims, claimDocuments]);
  const missingDocs = claimsEnabled && hasMissingRequiredDocs(selectedClaims, claimDocuments);
  const canUploadEvidence = claimsEnabled && Boolean(sessionId && effectiveAccountId);
  const [docMetadata, setDocMetadata] = React.useState<ClaimDocMetadata>(() =>
    claimsEnabled ? createMetadataFromDocuments(claimDocuments) : {}
  );
  React.useEffect(() => {
    if (!claimsEnabled) {
      setDocMetadata({});
      return;
    }
    setDocMetadata((previous) => {
      const merged: ClaimDocMetadata = {};
      const base = createMetadataFromDocuments(claimDocuments);
      for (const [claimKey, docMap] of Object.entries(base)) {
        const prevClaim = previous[claimKey as ClaimKey] ?? {};
        const mergedDocs: Partial<Record<string, FrontendReviewUploadDocInfo[]>> = {};
        for (const [docKey, docEntries] of Object.entries(docMap ?? {})) {
          const prevEntries = prevClaim?.[docKey] ?? [];
          mergedDocs[docKey] = dedupeDocInfos([...(prevEntries ?? []), ...(docEntries ?? [])]);
        }
        if (Object.keys(mergedDocs).length > 0) {
          merged[claimKey as ClaimKey] = mergedDocs;
        }
      }
      return merged;
    });
  }, [claimsEnabled, claimDocuments]);
  const [uploadingDocs, setUploadingDocs] = React.useState<Record<string, boolean>>({});
  const [uploadErrors, setUploadErrors] = React.useState<Record<string, string | null>>({});
  const alwaysIncludedDocsLabel = React.useMemo(
    () => BASIC_ALWAYS_INCLUDED.map((doc) => formatDocKey(doc)).join(', '),
    []
  );

  const makeUploadKey = React.useCallback((claimKey: ClaimKey, docKey: string) => `${claimKey}:${docKey}`, []);

  const updateClaimDocuments = React.useCallback(
    (claimKey: ClaimKey, docKey: string, docIds: string[]) => {
      if (!onAnswersChange) {
        return;
      }
      const nextDocs: ClaimDocuments = { ...(answers.claimDocuments ?? {}) };
      const claimEntry: Partial<Record<string, string[]>> = { ...(nextDocs[claimKey] ?? {}) };
      if (docIds.length > 0) {
        claimEntry[docKey] = docIds;
        nextDocs[claimKey] = claimEntry;
      } else {
        delete claimEntry[docKey];
        if (Object.keys(claimEntry).length > 0) {
          nextDocs[claimKey] = claimEntry;
        } else {
          delete nextDocs[claimKey];
        }
      }
      const nextAnswers: AccountQuestionAnswers = { ...answers };
      if (Object.keys(nextDocs).length > 0) {
        nextAnswers.claimDocuments = nextDocs;
      } else {
        delete nextAnswers.claimDocuments;
      }
      onAnswersChange(nextAnswers);
    },
    [answers, onAnswersChange]
  );

  const handleClaimToggle = React.useCallback(
    (claimKey: ClaimKey) => {
      if (!onAnswersChange) {
        return;
      }
      const currentClaims = Array.isArray(answers.claims)
        ? answers.claims.filter((claim): claim is ClaimKey => Boolean(claim && CLAIMS[claim as ClaimKey]))
        : [];
      const isSelected = currentClaims.includes(claimKey);
      const nextClaims = isSelected
        ? currentClaims.filter((key) => key !== claimKey)
        : [...currentClaims, claimKey];
      const nextAnswers: AccountQuestionAnswers = { ...answers };
      if (nextClaims.length > 0) {
        nextAnswers.claims = nextClaims;
      } else {
        delete nextAnswers.claims;
      }
      if (isSelected) {
        const nextDocs: ClaimDocuments = { ...(answers.claimDocuments ?? {}) };
        delete nextDocs[claimKey];
        if (Object.keys(nextDocs).length > 0) {
          nextAnswers.claimDocuments = nextDocs;
        } else {
          delete nextAnswers.claimDocuments;
        }
        setDocMetadata((previous) => {
          const nextMeta = { ...previous };
          delete nextMeta[claimKey];
          return nextMeta;
        });
        setUploadErrors((previous) => {
          const next = { ...previous };
          const prefix = `${claimKey}:`;
          for (const key of Object.keys(next)) {
            if (key.startsWith(prefix)) {
              delete next[key];
            }
          }
          return next;
        });
        setUploadingDocs((previous) => {
          const next = { ...previous };
          const prefix = `${claimKey}:`;
          for (const key of Object.keys(next)) {
            if (key.startsWith(prefix)) {
              delete next[key];
            }
          }
          return next;
        });
      }
      onAnswersChange(nextAnswers);
    },
    [answers, onAnswersChange]
  );

  const handleFileInputChange = React.useCallback(
    (claimKey: ClaimKey, docKey: string) =>
      async (event: React.ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files ? Array.from(event.target.files) : [];
        event.target.value = '';
        if (files.length === 0) {
          return;
        }
        const key = makeUploadKey(claimKey, docKey);
        if (!canUploadEvidence || !sessionId || !effectiveAccountId) {
          setUploadErrors((previous) => ({ ...previous, [key]: 'Uploads are unavailable for this account.' }));
          return;
        }
        let docIds = Array.isArray(claimDocuments?.[claimKey]?.[docKey])
          ? [...((claimDocuments?.[claimKey]?.[docKey] as string[]) ?? [])]
          : [];
        setUploadErrors((previous) => ({ ...previous, [key]: null }));
        setUploadingDocs((previous) => ({ ...previous, [key]: true }));
        try {
          for (const file of files) {
            const response = await uploadFrontendReviewEvidence(
              sessionId,
              effectiveAccountId,
              claimKey,
              docKey,
              file
            );
            const docInfo = response?.doc;
            const docId = docInfo?.id;
            if (!docId) {
              throw new Error('Upload failed: missing document id in response.');
            }
            if (!docIds.includes(docId)) {
              docIds = [...docIds, docId];
            }
            setDocMetadata((previous) => {
              const next = { ...previous };
              const claimEntry = { ...(next[claimKey] ?? {}) };
              const currentMeta = claimEntry[docKey] ?? [];
              const mergedMeta = dedupeDocInfos([
                ...currentMeta,
                { ...(docInfo ?? { id: docId, claim: claimKey, doc_key: docKey }) },
              ]);
              claimEntry[docKey] = mergedMeta;
              next[claimKey] = claimEntry;
              return next;
            });
          }
          updateClaimDocuments(claimKey, docKey, docIds);
        } catch (uploadError) {
          const message =
            uploadError instanceof Error ? uploadError.message : 'Unable to upload document.';
          setUploadErrors((previous) => ({ ...previous, [key]: message }));
        } finally {
          setUploadingDocs((previous) => ({ ...previous, [key]: false }));
        }
      },
    [
      canUploadEvidence,
      claimDocuments,
      effectiveAccountId,
      makeUploadKey,
      sessionId,
      updateClaimDocuments,
    ]
  );

  const handleRemoveDoc = React.useCallback(
    (claimKey: ClaimKey, docKey: string, docId: string) => {
      const current = Array.isArray(claimDocuments?.[claimKey]?.[docKey])
        ? [...((claimDocuments?.[claimKey]?.[docKey] as string[]) ?? [])]
        : [];
      const next = current.filter((entry) => entry !== docId);
      updateClaimDocuments(claimKey, docKey, next);
      setDocMetadata((previous) => {
        const nextMeta = { ...previous };
        const claimEntry = { ...(nextMeta[claimKey] ?? {}) };
        const currentMeta = claimEntry[docKey] ?? [];
        const filtered = currentMeta.filter((info) => info.id !== docId);
        if (filtered.length > 0) {
          claimEntry[docKey] = filtered;
          nextMeta[claimKey] = claimEntry;
        } else {
          delete claimEntry[docKey];
          if (Object.keys(claimEntry).length > 0) {
            nextMeta[claimKey] = claimEntry;
          } else {
            delete nextMeta[claimKey];
          }
        }
        return nextMeta;
      });
      setUploadErrors((previous) => {
        if (!previous) {
          return previous;
        }
        const nextErrors = { ...previous };
        delete nextErrors[makeUploadKey(claimKey, docKey)];
        return nextErrors;
      });
    },
    [claimDocuments, makeUploadKey, updateClaimDocuments]
  );

  const [detailsOpen, setDetailsOpen] = React.useState(false);

  const explanationValue = answers.explanation ?? '';
  const hasExplanation = typeof explanationValue === 'string' && explanationValue.trim() !== '';
  const disableBecauseOfStatus = status === 'saving' || status === 'waiting';
  const submitDisabled =
    disableBecauseOfStatus || !!error || !hasExplanation || (claimsEnabled && missingDocs);

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
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-base font-semibold text-slate-900">Claims</h3>
              <p className="text-sm text-slate-600">
                Select the statements that apply to this account. We’ll automatically attach {alwaysIncludedDocsLabel}.
              </p>
            </div>
            <div className="space-y-2">
              {Object.entries(CLAIMS).map(([key, definition]) => {
                const claimKey = key as ClaimKey;
                const checked = selectedClaims.includes(claimKey);
                return (
                  <label
                    key={claimKey}
                    className={cn(
                      'flex cursor-pointer flex-col gap-2 rounded-md border border-slate-200 bg-white p-3 transition',
                      checked ? 'border-slate-400' : 'hover:border-slate-300'
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        className="mt-1 h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500"
                        checked={checked}
                        onChange={() => handleClaimToggle(claimKey)}
                      />
                      <div className="space-y-1">
                        <p className="text-sm font-medium text-slate-900">{definition.label}</p>
                        {definition.hint ? (
                          <p className="text-xs text-slate-600">{definition.hint}</p>
                        ) : null}
                        {definition.requiresDocs ? (
                          <p className="text-xs text-slate-500">
                            Requires {definition.requiredDocs.length}{' '}
                            {definition.requiredDocs.length === 1 ? 'document' : 'documents'}.
                          </p>
                        ) : (
                          <p className="text-xs text-slate-500">No extra documents required.</p>
                        )}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
            {selectedClaims.some((claimKey) => CLAIMS[claimKey]?.requiresDocs) ? (
              <div className="space-y-4">
                {selectedClaims.map((claimKey) => {
                  const definition = CLAIMS[claimKey];
                  if (!definition?.requiresDocs) {
                    return null;
                  }
                  const claimMeta = docMetadata[claimKey] ?? {};
                  return (
                    <div
                      key={claimKey}
                      className="space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-4"
                    >
                      <div className="space-y-1">
                        <h4 className="text-sm font-semibold text-slate-900">
                          Upload documents for {definition.label}
                        </h4>
                        {definition.hint ? (
                          <p className="text-xs text-slate-600">{definition.hint}</p>
                        ) : null}
                      </div>
                      <div className="space-y-3">
                        {definition.requiredDocs.map((docKey) => {
                          const uploadKey = makeUploadKey(claimKey, docKey);
                          const docError = uploadErrors[uploadKey];
                          const isUploading = Boolean(uploadingDocs[uploadKey]);
                          const isMissing = Boolean(missingDocsMap[claimKey]?.includes(docKey)) && !disableBecauseOfStatus;
                          const entries = claimMeta?.[docKey] ?? [];
                          const docIds = (claimDocuments?.[claimKey]?.[docKey] as string[]) ?? [];
                          const renderedEntries = entries.length > 0
                            ? entries
                            : docIds.map((id) => ({ id, claim: claimKey, doc_key: docKey }));
                          return (
                            <div
                              key={docKey}
                              className={cn(
                                'rounded-md border p-3',
                                isMissing ? 'border-rose-300 bg-rose-50' : 'border-slate-200 bg-white'
                              )}
                            >
                              <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
                                <div>
                                  <p className="text-sm font-semibold text-slate-900">
                                    {formatDocKey(docKey)}
                                  </p>
                                  {isMissing ? (
                                    <p className="text-xs text-rose-700">
                                      At least one file is required for this document.
                                    </p>
                                  ) : null}
                                </div>
                                {isMissing ? (
                                  <span className="text-xs font-semibold uppercase text-rose-600">
                                    Required
                                  </span>
                                ) : null}
                              </div>
                              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                <input
                                  type="file"
                                  multiple
                                  onChange={handleFileInputChange(claimKey, docKey)}
                                  disabled={
                                    isUploading || !canUploadEvidence || disableBecauseOfStatus
                                  }
                                  className="max-w-xs text-sm text-slate-700 file:mr-3 file:rounded-md file:border file:border-slate-300 file:bg-white file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-50 disabled:cursor-not-allowed disabled:file:cursor-not-allowed"
                                />
                                <div className="flex items-center gap-3 text-xs text-slate-500">
                                  {isUploading ? <span>Uploading…</span> : null}
                                  {!canUploadEvidence ? (
                                    <span>Uploads unavailable for this account.</span>
                                  ) : null}
                                </div>
                              </div>
                              {docError ? (
                                <p className="mt-2 text-xs text-rose-700">{docError}</p>
                              ) : null}
                              {docIds.length > 0 ? (
                                <ul className="mt-3 space-y-2 text-xs text-slate-700">
                                  {renderedEntries.map((doc) => (
                                    <li
                                      key={doc.id}
                                      className="flex items-center justify-between gap-2 rounded border border-slate-200 bg-slate-50 px-2 py-1"
                                    >
                                      <span className="truncate">{doc.filename ?? doc.id}</span>
                                      <button
                                        type="button"
                                        onClick={() => handleRemoveDoc(claimKey, docKey, doc.id)}
                                        className="text-xs font-medium text-slate-500 transition hover:text-rose-600"
                                      >
                                        Remove
                                      </button>
                                    </li>
                                  ))}
                                </ul>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : null}
          </div>
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
