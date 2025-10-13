import { fireEvent, render, screen } from '@testing-library/react';
import React from 'react';
import AccountQuestions, { type AccountQuestionAnswers } from '../components/AccountQuestions';

describe('AccountQuestions', () => {
  it('renders initial answers when provided', () => {
    const initial: AccountQuestionAnswers = {
      ownership: 'yes',
      recognize: 'no',
      explanation: 'Initial note',
      identity_theft: 'no'
    };

    render(<AccountQuestions initialAnswers={initial} />);

    expect(screen.getByLabelText(/Do you own this account/i)).toHaveValue('yes');
    expect(screen.getByLabelText(/Do you recognize this account/i)).toHaveValue('no');
    expect(screen.getByLabelText(/Could this be identity theft/i)).toHaveValue('no');
    expect(screen.getByLabelText(/Anything else we should know/i)).toHaveValue('Initial note');
  });

  it('bubbles answer changes to parent', () => {
    const handleChange = jest.fn();
    render(<AccountQuestions onChange={handleChange} />);

    fireEvent.change(screen.getByLabelText(/Do you own this account/i), {
      target: { value: 'yes' }
    });
    expect(handleChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        ownership: 'yes'
      })
    );

    fireEvent.change(screen.getByLabelText(/Anything else we should know/i), {
      target: { value: 'Some explanation' }
    });
    expect(handleChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        explanation: 'Some explanation'
      })
    );
  });

  it('limits the explanation to 1500 characters', () => {
    const handleChange = jest.fn();
    render(<AccountQuestions onChange={handleChange} />);

    const longText = 'a'.repeat(1600);
    const textarea = screen.getByLabelText(/Anything else we should know/i) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: longText } });

    expect(textarea.value).toHaveLength(1500);
    expect(handleChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        explanation: 'a'.repeat(1500)
      })
    );
  });
});
