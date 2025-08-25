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
  normalized_name: 'account 1',
  account_number_last4: '1234',
  original_creditor: 'Creditor 1',
  primary_issue: 'late_payment',
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
        { account_id: 'acc2', name: 'Account 2', account_number_last4: '5678' }
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

test('renders primary badge from primary_issue and secondary chips with identifiers', async () => {
  const acc = {
    account_id: 'acc3',
    name: 'Account 3',
    normalized_name: 'account 3',
    account_number_last4: '7890',
    original_creditor: 'Bank A',
    primary_issue: 'charge_off',
    issue_types: ['collection', 'charge_off', 'late_payment'],
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}>
      <ReviewPage />
    </MemoryRouter>
  );
  const header = await screen.findByText('Account 3');
  expect(header.parentElement).toHaveTextContent('Account 3 ••••7890 - Bank A');
  expect(screen.getByText('Charge-Off')).toHaveClass('badge');
  expect(screen.getByText('Collection')).toHaveClass('chip');
  expect(screen.getByText('Late Payment')).toHaveClass('chip');
});

test('renders account_fingerprint when last4 missing', async () => {
  const acc = {
    account_id: 'acc4',
    name: 'Account 4',
    normalized_name: 'account 4',
    account_fingerprint: 'deadbeef',
    original_creditor: 'Creditor 4',
    primary_issue: 'late_payment',
    issue_types: ['late_payment'],
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const header = await screen.findByText('Account 4');
  expect(header.parentElement).toHaveTextContent('Account 4 deadbeef - Creditor 4');
});

test('prefers last4 over fingerprint when both provided', async () => {
  const acc = {
    account_id: 'acc5',
    name: 'Account 5',
    normalized_name: 'account 5',
    account_number_last4: '4321',
    account_fingerprint: 'cafebabe',
    original_creditor: 'Creditor 5',
    primary_issue: 'late_payment',
    issue_types: ['late_payment'],
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const header = await screen.findByText('Account 5');
  expect(header.parentElement).toHaveTextContent('Account 5 ••••4321 - Creditor 5');
  expect(header.parentElement).not.toHaveTextContent('cafebabe');
});

test('dedup uses last4 before fingerprint', async () => {
  const acc1 = {
    account_id: 'acc6',
    name: 'Account 6',
    normalized_name: 'account 6',
    account_number_last4: '9999',
    account_fingerprint: 'aaaa',
    primary_issue: 'late_payment',
    issue_types: ['late_payment'],
  };
  const acc2 = {
    ...acc1,
    account_id: 'acc7',
    account_fingerprint: 'bbbb',
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc1, acc2] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  // Only one card should render despite differing fingerprints
  const headers = await screen.findAllByText('Account 6');
  expect(headers).toHaveLength(1);
});

test('handles missing payment maps and late payments with identifier fallback', async () => {
  const acc1 = {
    account_id: 'acc8',
    name: 'Account 8',
    normalized_name: 'account 8',
    account_number_last4: '1234',
    issue_types: ['late_payment'],
  };
  const acc2 = {
    account_id: 'acc9',
    name: 'Account 9',
    normalized_name: 'account 9',
    account_fingerprint: 'ff99',
    issue_types: ['late_payment'],
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc1, acc2] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const h1 = await screen.findByText('Account 8');
  const h2 = await screen.findByText('Account 9');
  expect(h1.parentElement).toHaveTextContent('Account 8 ••••1234');
  expect(h2.parentElement).toHaveTextContent('Account 9 ff99');
});

test('renders evidence drawer when debug flag enabled', async () => {
  const originalEnv = process.env.VITE_DEBUG_EVIDENCE;
  process.env.VITE_DEBUG_EVIDENCE = '1';
  const acc = {
    ...account,
    account_trace: { payment_status: 'late' }
  };
  const uploadData = {
    ...baseUploadData,
    accounts: { negative_accounts: [acc] },
  };
  render(
    <MemoryRouter initialEntries={[{ pathname: '/review', state: { uploadData } }]}> 
      <ReviewPage />
    </MemoryRouter>
  );
  const expander = await screen.findByText('View evidence');
  const content = screen.getByText(/payment_status/i);
  expect(content).not.toBeVisible();
  fireEvent.click(expander);
  expect(content).toBeVisible();
  process.env.VITE_DEBUG_EVIDENCE = originalEnv;
});
