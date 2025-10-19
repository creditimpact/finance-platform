import { prepareAnswersPayload } from '../reviewClaims';
import type { AccountQuestionAnswers } from '../../components/AccountQuestions';

describe('prepareAnswersPayload', () => {
  it('trims explanations and omits empty values by default', () => {
    const answers: AccountQuestionAnswers = {
      explanation: '  This needs attention.  ',
    };

    expect(prepareAnswersPayload(answers)).toEqual({
      answers: { explanation: 'This needs attention.' },
    });
  });

  it('includes claims and evidence when requested', () => {
    const answers: AccountQuestionAnswers = {
      explanation: ' Paid and misreported ',
      claims: ['paid_in_full', 'wrong_dofd', 'paid_in_full' as any],
      claimDocuments: {
        paid_in_full: {
          pay_proof: [' runs/documents/pay.pdf ', '', 'runs/documents/pay.pdf'],
          payoff_letter: ['runs/documents/letter.pdf'],
        },
        wrong_dofd: {
          original_chargeoff_letter_or_old_statements: [
            ' runs/documents/dofd.pdf ',
          ],
        },
        // @ts-expect-error - unknown claim keys should be ignored by the helper
        invalid_claim: {
          doc_key: ['runs/ignored.pdf'],
        },
      },
    };

    expect(prepareAnswersPayload(answers, { includeClaims: true })).toEqual({
      answers: { explanation: 'Paid and misreported' },
      claims: ['paid_in_full', 'wrong_dofd'],
      evidence: [
        {
          claim: 'paid_in_full',
          docs: [
            {
              doc_key: 'pay_proof',
              doc_ids: ['runs/documents/pay.pdf'],
            },
            {
              doc_key: 'payoff_letter',
              doc_ids: ['runs/documents/letter.pdf'],
            },
          ],
        },
        {
          claim: 'wrong_dofd',
          docs: [
            {
              doc_key: 'original_chargeoff_letter_or_old_statements',
              doc_ids: ['runs/documents/dofd.pdf'],
            },
          ],
        },
      ],
    });
  });

  it('omits claims metadata when includeClaims is disabled', () => {
    const answers: AccountQuestionAnswers = {
      explanation: '  explain  ',
      claims: ['paid_in_full'],
      claimDocuments: {
        paid_in_full: {
          pay_proof: ['runs/documents/pay.pdf'],
        },
      },
    };

    expect(prepareAnswersPayload(answers, { includeClaims: false })).toEqual({
      answers: { explanation: 'explain' },
    });
  });
});
