import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { cn } from '../lib/utils';
import FieldSummary from './FieldSummary';
import BureauGrid from './BureauGrid';
import { AgreementLevel, BUREAUS, BureauKey, MISSING_VALUE } from './accountFieldTypes';
import { summarizeField as summarizeBureauField, type BureauTriple } from '../utils/bureauSummary';
import { QUESTION_COPY, type AccountQuestionKey } from './questionCopy';

type PerBureauBlock = {
  per_bureau?: Partial<Record<BureauKey, string | null | undefined>>;
};

type DateBlock = Partial<Record<BureauKey, string | null | undefined>>;

type QuestionBlock = Partial<Record<AccountQuestionKey, string | null | undefined>>;

type AccountDisplay = {
  account_number?: PerBureauBlock;
  account_type?: PerBureauBlock;
  status?: PerBureauBlock;
  balance_owed?: PerBureauBlock;
  date_opened?: DateBlock;
  closed_date?: DateBlock;
  questions?: QuestionBlock;
};

export type AccountPack = {
  holder_name?: string | null;
  primary_issue?: string | null;
  display?: AccountDisplay | null;
};

type SummaryFieldConfig = {
  key: keyof AccountDisplay;
  label: string;
};

type FieldSummaryEntry = SummaryFieldConfig & {
  summaryValue: string;
  agreement: AgreementLevel;
  values: Partial<Record<BureauKey, string>>;
};

const SUMMARY_FIELDS: SummaryFieldConfig[] = [
  { key: 'account_type', label: 'Account type' },
  { key: 'status', label: 'Status' },
  { key: 'balance_owed', label: 'Balance owed' },
  { key: 'date_opened', label: 'Date opened' },
  { key: 'closed_date', label: 'Closed date' }
];

function toBureauTriple(field: PerBureauBlock | DateBlock | undefined): BureauTriple {
  if (!field) {
    return {};
  }

  const triple: BureauTriple = {};

  if ('per_bureau' in field) {
    const perBureau = field.per_bureau ?? {};
    for (const bureau of BUREAUS) {
      const value = perBureau[bureau];
      if (value != null) {
        triple[bureau] = value;
      }
    }
    return triple;
  }

  for (const bureau of BUREAUS) {
    const value = (field as DateBlock)[bureau];
    if (value != null) {
      triple[bureau] = value;
    }
  }

  return triple;
}

function buildFieldSummary(
  field: PerBureauBlock | DateBlock | undefined,
  config: SummaryFieldConfig
): FieldSummaryEntry {
  const summary = summarizeBureauField(toBureauTriple(field));

  return {
    ...config,
    summaryValue: summary.summary,
    agreement: summary.agreement as AgreementLevel,
    values: summary.values
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

  const fieldSummaries = React.useMemo<FieldSummaryEntry[]>(() => {
    return SUMMARY_FIELDS.map((field) =>
      buildFieldSummary(display[field.key] as PerBureauBlock | DateBlock | undefined, field)
    );
  }, [display]);

  const hasDisagreement = fieldSummaries.some(
    (field) => field.agreement === 'majority' || field.agreement === 'mixed'
  );

  const [expanded, setExpanded] = React.useState(hasDisagreement);

  React.useEffect(() => {
    setExpanded(hasDisagreement);
  }, [hasDisagreement]);

  const accountNumberSummary = React.useMemo(() => {
    const result = summarizeBureauField(toBureauTriple(display.account_number), {
      kind: 'account_number'
    });
    return result.summary;
  }, [display.account_number]);

  const questions: QuestionBlock = display.questions ?? {};

  return (
    <Card className="w-full">
      <CardHeader className="gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-xl font-semibold text-slate-900">
            {pack.holder_name ?? 'Unknown account holder'}
          </CardTitle>
          <p className="text-sm text-slate-600">Account number: {accountNumberSummary ?? MISSING_VALUE}</p>
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
            <FieldSummary
              key={field.key}
              label={field.label}
              value={field.summaryValue}
              agreement={field.agreement}
            />
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
            <BureauGrid
              fields={fieldSummaries.map((field) => ({
                fieldKey: field.key,
                label: field.label,
                values: field.values,
                agreement: field.agreement
              }))}
            />
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
