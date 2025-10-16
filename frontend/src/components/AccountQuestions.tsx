import * as React from 'react';
import { QUESTION_COPY, type AccountQuestionKey } from './questionCopy';

export type AccountQuestionAnswers = Partial<Record<AccountQuestionKey, string>>;

export interface AccountQuestionsProps {
  onChange?: (answers: AccountQuestionAnswers) => void;
  initialAnswers?: AccountQuestionAnswers;
  visibleQuestions?: AccountQuestionKey[];
}

const MAX_EXPLANATION_LENGTH = 1500;

const QUESTION_ORDER: AccountQuestionKey[] = [
  'ownership',
  'recognize',
  'identity_theft',
  'explanation'
];

const OWNERSHIP_OPTIONS = [
  { value: '', label: 'Select an option' },
  { value: 'yes', label: 'Yes' },
  { value: 'no', label: 'No' },
  { value: 'not_sure', label: 'Not sure' }
];

const RECOGNIZE_OPTIONS = OWNERSHIP_OPTIONS;

const IDENTITY_THEFT_OPTIONS = [
  { value: '', label: 'Select an option' },
  { value: 'yes', label: 'Yes' },
  { value: 'no', label: 'No' }
];

function createInitialState(initialAnswers?: AccountQuestionAnswers) {
  return {
    ownership: initialAnswers?.ownership ?? '',
    recognize: initialAnswers?.recognize ?? '',
    explanation: initialAnswers?.explanation ?? '',
    identity_theft: initialAnswers?.identity_theft ?? ''
  } satisfies Record<AccountQuestionKey, string>;
}

export function AccountQuestions({ onChange, initialAnswers, visibleQuestions }: AccountQuestionsProps) {
  const normalizedVisibleQuestions = React.useMemo(() => {
    if (!visibleQuestions || visibleQuestions.length === 0) {
      return QUESTION_ORDER;
    }

    const seen = new Set<AccountQuestionKey>();
    const filtered: AccountQuestionKey[] = [];

    for (const key of visibleQuestions) {
      if (QUESTION_ORDER.includes(key) && !seen.has(key)) {
        filtered.push(key);
        seen.add(key);
      }
    }

    if (filtered.length === 0) {
      return QUESTION_ORDER;
    }

    return filtered;
  }, [visibleQuestions]);

  const [answers, setAnswers] = React.useState(() => createInitialState(initialAnswers));

  React.useEffect(() => {
    setAnswers(createInitialState(initialAnswers));
  }, [initialAnswers?.ownership, initialAnswers?.recognize, initialAnswers?.explanation, initialAnswers?.identity_theft]);

  const updateAnswers = React.useCallback(
    (field: AccountQuestionKey, value: string) => {
      setAnswers((previous) => {
        if (previous[field] === value) {
          return previous;
        }

        const next = { ...previous, [field]: value };
        onChange?.(normalizeAnswers(next));
        return next;
      });
    },
    [onChange]
  );

  const handleSelectChange = React.useCallback(
    (field: Exclude<AccountQuestionKey, 'explanation'>) =>
      (event: React.ChangeEvent<HTMLSelectElement>) => {
        updateAnswers(field, event.target.value);
      },
    [updateAnswers]
  );

  const handleExplanationChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextValue = event.target.value.slice(0, MAX_EXPLANATION_LENGTH);
      updateAnswers('explanation', nextValue);
    },
    [updateAnswers]
  );

  const explanationLength = answers.explanation.length;

  const showOwnership = normalizedVisibleQuestions.includes('ownership');
  const showRecognize = normalizedVisibleQuestions.includes('recognize');
  const showIdentityTheft = normalizedVisibleQuestions.includes('identity_theft');
  const showExplanation = normalizedVisibleQuestions.includes('explanation');

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        {showOwnership ? (
          <QuestionSelect
            id="account-question-ownership"
            label={QUESTION_COPY.ownership.title}
            helper={QUESTION_COPY.ownership.helper}
            options={OWNERSHIP_OPTIONS}
            value={answers.ownership}
            onChange={handleSelectChange('ownership')}
          />
        ) : null}
        {showRecognize ? (
          <QuestionSelect
            id="account-question-recognize"
            label={QUESTION_COPY.recognize.title}
            helper={QUESTION_COPY.recognize.helper}
            options={RECOGNIZE_OPTIONS}
            value={answers.recognize}
            onChange={handleSelectChange('recognize')}
          />
        ) : null}
        {showIdentityTheft ? (
          <QuestionSelect
            id="account-question-identity-theft"
            label={QUESTION_COPY.identity_theft.title}
            helper={QUESTION_COPY.identity_theft.helper}
            options={IDENTITY_THEFT_OPTIONS}
            value={answers.identity_theft}
            onChange={handleSelectChange('identity_theft')}
          />
        ) : null}
        {showExplanation ? (
          <ExplanationField
            id="account-question-explanation"
            label={QUESTION_COPY.explanation.title}
            helper={QUESTION_COPY.explanation.helper}
            value={answers.explanation}
            onChange={handleExplanationChange}
            maxLength={MAX_EXPLANATION_LENGTH}
            length={explanationLength}
          />
        ) : null}
      </div>
    </div>
  );
}

function normalizeAnswers(values: Record<AccountQuestionKey, string>): AccountQuestionAnswers {
  return Object.fromEntries(
    (Object.entries(values) as [AccountQuestionKey, string][]).map(([key, value]) => [
      key,
      value === '' ? undefined : value
    ])
  ) as AccountQuestionAnswers;
}

interface QuestionSelectProps {
  id: string;
  label: string;
  helper: string;
  options: Array<{ value: string; label: string }>;
  value: string;
  onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
}

function QuestionSelect({ id, label, helper, options, value, onChange }: QuestionSelectProps) {
  const helperId = `${id}-helper`;

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={id} className="text-sm font-semibold text-slate-900">
        {label}
      </label>
      <p id={helperId} className="text-xs text-slate-600">
        {helper}
      </p>
      <select
        id={id}
        name={id}
        value={value}
        onChange={onChange}
        aria-describedby={helperId}
        className="block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500"
      >
        {options.map((option) => (
          <option key={option.value || 'empty'} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface ExplanationFieldProps {
  id: string;
  label: string;
  helper: string;
  value: string;
  onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  maxLength: number;
  length: number;
}

function ExplanationField({ id, label, helper, value, onChange, maxLength, length }: ExplanationFieldProps) {
  const helperId = `${id}-helper`;
  const countId = `${id}-description`;
  const describedBy = [helperId, countId].filter(Boolean).join(' ');

  return (
    <div className="flex flex-col gap-2 md:col-span-2">
      <label htmlFor={id} className="text-sm font-semibold text-slate-900">
        {label}
      </label>
      <p id={helperId} className="text-xs text-slate-600">
        {helper}
      </p>
      <textarea
        id={id}
        name={id}
        value={value}
        onChange={onChange}
        maxLength={maxLength}
        aria-describedby={describedBy}
        rows={4}
        className="block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500"
      />
      <p id={countId} className="text-xs text-slate-500">
        {length} / {maxLength} characters
      </p>
    </div>
  );
}

export default AccountQuestions;
