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
import AccountQuestions, { type AccountQuestionAnswers } from './AccountQuestions';
import { type AccountQuestionKey } from './questionCopy';
import type { AccountPack } from './AccountCard';
import type { FrontendReviewResponse } from '../api';

export type ReviewCardStatus = 'idle' | 'waiting' | 'ready' | 'saving' | 'done';

type QuestionDescriptor = {
  id?: string | null;
  required?: boolean | string | null;
  prompt?: string | null;
};

export type ReviewAccountPack = AccountPack & {
  account_id?: string;
  questions?: QuestionDescriptor[] | null;
  answers?: Record<string, string> | null;
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

const QUESTION_ORDER: AccountQuestionKey[] = [
  'ownership',
  'recognize',
  'identity_theft',
  'explanation',
];

function isAccountQuestionKey(value: unknown): value is AccountQuestionKey {
  return typeof value === 'string' && (QUESTION_ORDER as string[]).includes(value);
}

function defaultRequired(key: AccountQuestionKey): boolean {
  return key !== 'explanation';
}

function parseQuestionRequired(value: unknown, key: AccountQuestionKey): boolean {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim().toLowerCase();
    if (['true', 'required', 'yes', '1'].includes(trimmed)) {
      return true;
    }
    if (['false', 'optional', 'no', '0'].includes(trimmed)) {
      return false;
    }
  }

  return defaultRequired(key);
}

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

function useQuestionConfigs(pack: ReviewAccountPack) {
  return React.useMemo(() => {
    const configs: Array<{ key: AccountQuestionKey; required: boolean }> = [];
    const seen = new Set<AccountQuestionKey>();

    const pushConfig = (key: AccountQuestionKey, required?: boolean) => {
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      const requirement = required ?? defaultRequired(key);
      configs.push({ key, required: requirement });
    };

    const fromPack = Array.isArray(pack.questions) ? pack.questions : [];
    for (const descriptor of fromPack) {
      const key = descriptor?.id;
      if (!isAccountQuestionKey(key)) {
        continue;
      }
      pushConfig(key, parseQuestionRequired(descriptor?.required, key));
    }

    if (configs.length === 0) {
      const displayQuestions = pack.display?.questions;
      if (displayQuestions && typeof displayQuestions === 'object') {
        for (const key of Object.keys(displayQuestions)) {
          if (isAccountQuestionKey(key)) {
            pushConfig(key as AccountQuestionKey);
          }
        }
      }
    }

    if (configs.length === 0) {
      for (const key of QUESTION_ORDER) {
        pushConfig(key);
      }
    }

    configs.sort(
      (a, b) => QUESTION_ORDER.indexOf(a.key) - QUESTION_ORDER.indexOf(b.key)
    );

    return configs;
  }, [pack.questions, pack.display?.questions]);
}

export interface ReviewCardProps {
  pack: ReviewAccountPack;
  accountId?: string;
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

  const [detailsOpen, setDetailsOpen] = React.useState(false);

  const questionConfigs = useQuestionConfigs(pack);
  const visibleQuestions = React.useMemo(
    () => questionConfigs.map((config) => config.key),
    [questionConfigs]
  );
  const isAnswerProvided = React.useCallback((value?: string) => {
    return typeof value === 'string' && value.trim() !== '';
  }, []);

  const hasAnyAnswer = React.useMemo(
    () => questionConfigs.some((config) => isAnswerProvided(answers[config.key])),
    [answers, isAnswerProvided, questionConfigs]
  );

  const missingRequired = React.useMemo(
    () =>
      questionConfigs.some((config) => {
        if (!config.required) {
          return false;
        }
        return !isAnswerProvided(answers[config.key]);
      }),
    [answers, isAnswerProvided, questionConfigs]
  );

  const disableBecauseOfStatus = status === 'saving' || status === 'waiting';
  const requireAnswers = questionConfigs.length > 0;
  const submitDisabled =
    disableBecauseOfStatus ||
    !!error ||
    (requireAnswers && (!hasAnyAnswer || missingRequired));

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

  const buttonLabel = success ? 'Saved' : status === 'saving' ? 'Saving…' : 'Save answers';

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

        {questionConfigs.length > 0 ? (
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-semibold text-slate-900">Questions</h3>
              <p className="mt-1 text-sm text-slate-600">
                Your answers help us tailor dispute language and keep records of your responses.
              </p>
            </div>
            <AccountQuestions
              onChange={onAnswersChange}
              initialAnswers={answers}
              visibleQuestions={visibleQuestions}
            />
          </div>
        ) : null}

        {error ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">{error}</div>
        ) : null}

        {success && status !== 'ready' ? (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
            Answers saved successfully.
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
          {missingRequired && !disableBecauseOfStatus ? (
            <p className="mt-2 text-xs text-rose-600">Answer the required questions before submitting.</p>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

export default ReviewCard;
