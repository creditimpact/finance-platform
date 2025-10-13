import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';

type BureauKey = 'transunion' | 'experian' | 'equifax';

type PerBureauBlock = {
  per_bureau?: Partial<Record<BureauKey, string | null | undefined>>;
};

type DateBlock = Partial<Record<BureauKey, string | null | undefined>>;

type QuestionBlock = Partial<Record<'ownership' | 'recognize' | 'explanation' | 'identity_theft', string | null | undefined>>;

type AccountDisplay = {
  account_number?: PerBureauBlock;
  account_type?: PerBureauBlock;
  status?: PerBureauBlock;
  balance_owed?: PerBureauBlock;
  date_opened?: DateBlock;
  closed_date?: DateBlock;
  questions?: QuestionBlock;
};

type AccountPack = {
  holder_name?: string | null;
  primary_issue?: string | null;
  display?: AccountDisplay | null;
};

type AgreementLevel = 'all' | 'two' | 'mixed' | 'none';

type SummaryFieldConfig = {
  key: keyof AccountDisplay;
  label: string;
};

type FieldSummary = SummaryFieldConfig & {
  summaryValue: string;
  agreement: AgreementLevel;
  values: Record<BureauKey, string>;
};

const BUREAUS: BureauKey[] = ['transunion', 'experian', 'equifax'];
const BUREAU_LABELS: Record<BureauKey, string> = {
  transunion: 'TU',
  experian: 'EX',
  equifax: 'EF'
};

const SUMMARY_FIELDS: SummaryFieldConfig[] = [
  { key: 'account_type', label: 'Account type' },
  { key: 'status', label: 'Status' },
  { key: 'balance_owed', label: 'Balance owed' },
  { key: 'date_opened', label: 'Date opened' },
  { key: 'closed_date', label: 'Closed date' }
];

const AGREEMENT_LABELS: Record<AgreementLevel, string> = {
  all: 'All agree',
  two: '2 of 3',
  mixed: 'Mixed',
  none: '—'
};

const AGREEMENT_BADGE_STYLES: Record<AgreementLevel, string> = {
  all: 'bg-emerald-100 text-emerald-800',
  two: 'bg-amber-100 text-amber-900',
  mixed: 'bg-rose-100 text-rose-900',
  none: 'bg-slate-100 text-slate-600'
};

const QUESTION_COPY: Record<keyof NonNullable<QuestionBlock>, { title: string; helper: string }> = {
  ownership: {
    title: 'Do you own this account?',
    helper: 'Tell us if the account belongs to you or a shared account.'
  },
  recognize: {
    title: 'Do you recognize this account?',
    helper: 'Let us know if the account looks familiar or if it is unexpected.'
  },
  explanation: {
    title: 'Anything else we should know?',
    helper: 'Add a quick note that might help us understand the situation.'
  },
  identity_theft: {
    title: 'Could this be identity theft?',
    helper: 'Share if you suspect this account is the result of identity theft.'
  }
};

function normalizeValue(value: string | null | undefined): string {
  if (!value || value.trim() === '' || value === '--') {
    return '--';
  }
  return value;
}

function extractPerBureauValues(field: PerBureauBlock | DateBlock | undefined): Record<BureauKey, string> {
  const defaults: Record<BureauKey, string> = {
    transunion: '--',
    experian: '--',
    equifax: '--'
  };

  if (!field) {
    return defaults;
  }

  if ('per_bureau' in field) {
    const perBureau = field.per_bureau ?? {};
    return {
      transunion: normalizeValue(perBureau.transunion ?? '--'),
      experian: normalizeValue(perBureau.experian ?? '--'),
      equifax: normalizeValue(perBureau.equifax ?? '--')
    };
  }

  const dateField = field as DateBlock;
  return {
    transunion: normalizeValue(dateField.transunion ?? '--'),
    experian: normalizeValue(dateField.experian ?? '--'),
    equifax: normalizeValue(dateField.equifax ?? '--')
  };
}

function computeAgreement(values: Record<BureauKey, string>): { value: string; agreement: AgreementLevel } {
  const entries = BUREAUS.map((bureau) => normalizeValue(values[bureau]));
  const nonEmpty = entries.filter((value) => value !== '--');

  if (nonEmpty.length === 0) {
    return { value: '--', agreement: 'none' };
  }

  const counts = new Map<string, number>();
  nonEmpty.forEach((value) => {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  });

  const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const [topValue, topCount] = sorted[0];

  if (counts.size === 1) {
    if (nonEmpty.length === BUREAUS.length) {
      return { value: topValue, agreement: 'all' };
    }
    if (nonEmpty.length >= 2) {
      return { value: topValue, agreement: 'two' };
    }
    return { value: topValue, agreement: 'none' };
  }

  if (topCount >= 2) {
    return { value: topValue, agreement: 'two' };
  }

  return { value: topValue, agreement: 'mixed' };
}

function summarizeField(field: PerBureauBlock | DateBlock | undefined, config: SummaryFieldConfig): FieldSummary {
  const values = extractPerBureauValues(field);
  const { value, agreement } = computeAgreement(values);

  return {
    ...config,
    summaryValue: value,
    agreement,
    values
  };
}

function formatPrimaryIssue(issue?: string | null) {
  if (!issue) {
    return null;
  }
  return issue.replace(/_/g, ' ');
}

const ChevronDownIcon = ({ className }: { className?: string }) => (
  <svg
    aria-hidden="true"
    viewBox="0 0 24 24"
    className={cn('h-4 w-4 text-slate-600', className)}
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

export interface AccountCardProps {
  pack: AccountPack;
}

export function AccountCard({ pack }: AccountCardProps) {
  const display = pack.display ?? ({} as AccountDisplay);

  const fieldSummaries = React.useMemo<FieldSummary[]>(
    () =>
      SUMMARY_FIELDS.map((field) =>
        summarizeField(display[field.key] as PerBureauBlock | DateBlock | undefined, field)
      ),
    [display]
  );

  const hasDisagreement = fieldSummaries.some((field) => field.agreement === 'two' || field.agreement === 'mixed');

  const [expanded, setExpanded] = React.useState(hasDisagreement);

  React.useEffect(() => {
    setExpanded(hasDisagreement);
  }, [hasDisagreement]);

  const accountNumberSummary = React.useMemo(() => {
    const accountNumberValues = extractPerBureauValues(display.account_number);
    return computeAgreement(accountNumberValues).value;
  }, [display.account_number]);

  const questions: QuestionBlock = display.questions ?? {};

  return (
    <Card className="w-full">
      <CardHeader className="gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-xl font-semibold text-slate-900">
            {pack.holder_name ?? 'Unknown account holder'}
          </CardTitle>
          <p className="text-sm text-slate-600">Account number: {accountNumberSummary ?? '--'}</p>
        </div>
        {pack.primary_issue ? (
          <Badge variant="outline" className="whitespace-nowrap text-xs font-semibold capitalize text-slate-700">
            {formatPrimaryIssue(pack.primary_issue)}
          </Badge>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-wrap gap-4">
          {fieldSummaries.map((field) => (
            <div
              key={field.key}
              className="flex min-w-[160px] flex-1 flex-col gap-2 rounded-lg border border-slate-200 bg-slate-50 p-4"
            >
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">{field.label}</span>
              <span className="text-sm font-semibold text-slate-900">{field.summaryValue}</span>
              <Badge
                variant="subtle"
                className={cn('w-fit', AGREEMENT_BADGE_STYLES[field.agreement])}
              >
                {AGREEMENT_LABELS[field.agreement]}
              </Badge>
            </div>
          ))}
        </div>

        <div className="space-y-3">
          <button
            type="button"
            className="flex items-center gap-2 text-sm font-semibold text-slate-700 transition hover:text-slate-900"
            onClick={() => setExpanded((state) => !state)}
          >
            <span>{expanded ? 'Hide details' : 'See details'}</span>
            <ChevronDownIcon className={cn('transition-transform', expanded ? 'rotate-180' : 'rotate-0')} />
            {hasDisagreement ? (
              <Badge className="bg-amber-100 text-amber-900">Disagreement</Badge>
            ) : null}
          </button>

          {expanded ? (
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <div className="grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))] bg-slate-50 text-sm font-medium text-slate-600">
                <div className="px-4 py-3 text-left">Field</div>
                {BUREAUS.map((bureau) => (
                  <div key={bureau} className="px-4 py-3 text-center">
                    {BUREAU_LABELS[bureau]}
                  </div>
                ))}
              </div>
              <div className="divide-y divide-slate-200 text-sm">
                {fieldSummaries.map((field) => {
                  const rowHighlight = field.agreement === 'two' || field.agreement === 'mixed';
                  return (
                    <div
                      key={field.key}
                      className={cn('grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))]', rowHighlight && 'bg-amber-50')}
                    >
                      <div className="px-4 py-3 font-medium text-slate-700">{field.label}</div>
                      {BUREAUS.map((bureau) => (
                        <div key={bureau} className="px-4 py-3 text-center text-slate-800">
                          {field.values[bureau]}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          <h4 className="text-base font-semibold text-slate-900">Tell us about this account</h4>
          <div className="grid gap-3 md:grid-cols-2">
            {Object.entries(QUESTION_COPY).map(([key, copy]) => {
              const value = questions?.[key as keyof QuestionBlock];
              return (
                <div
                  key={key}
                  className="flex flex-col gap-2 rounded-lg border border-slate-200 p-4"
                >
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{copy.title}</p>
                    <p className="text-xs text-slate-600">{copy.helper}</p>
                  </div>
                  <Badge variant="subtle" className="w-fit bg-slate-100 text-slate-600">
                    {value ? value : 'No response yet'}
                  </Badge>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default AccountCard;
