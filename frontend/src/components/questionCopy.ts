export type AccountQuestionKey =
  | 'ownership'
  | 'recognize'
  | 'explanation'
  | 'identity_theft';

export type QuestionCopy = {
  title: string;
  helper: string;
};

export const QUESTION_COPY: Record<AccountQuestionKey, QuestionCopy> = {
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
