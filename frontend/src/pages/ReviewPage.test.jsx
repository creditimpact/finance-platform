import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ReviewPage from './ReviewPage';

jest.mock('../api', () => ({
  submitExplanations: jest.fn(),
  getSummaries: jest.fn().mockResolvedValue({
    summaries: {
      acc1: { facts_summary: 'bank error' }
    }
  })
}));

const baseUploadData = {
  session_id: 'sess1',
  filename: 'file.pdf',
  email: 'test@example.com'
};

const account = {
  account_id: 'acc1',
  name: 'Account 1',
  account_number: '1234',
  issue_types: ['late_payment']
};

describe.each([
  'negative_accounts',
  'disputes',
  'open_accounts_with_issues',
  'goodwill'
])('ReviewPage with %s', (key) => {
  const uploadData = { ...baseUploadData, accounts: { [key]: [account] } };

  test('renders helper text', async () => {
    render(
      <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
        <ReviewPage />
      </MemoryRouter>
    );
    expect(
      await screen.findByText(/We’ll use your explanation as context/i)
    ).toBeInTheDocument();
  });

  test('shows summary box when toggle active', async () => {
    render(
      <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
        <ReviewPage />
      </MemoryRouter>
    );
    const toggle = await screen.findByLabelText(/Show how the system understood/i);
    fireEvent.click(toggle);
    expect(await screen.findByText(/bank error/i)).toBeInTheDocument();
  });
});

test('filters out accounts without issue_types', async () => {
  const uploadData = {
    ...baseUploadData,
    accounts: {
      negative_accounts: [
        account,
        { account_id: 'acc2', name: 'Account 2', account_number: '5678' }
      ]
    }
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  expect(await screen.findByText('Account 1')).toBeInTheDocument();
  expect(screen.queryByText('Account 2')).not.toBeInTheDocument();
});
