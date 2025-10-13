import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import AccountsPage from '../pages/Accounts';
import samplePack from '../__fixtures__/sampleAccountPack.json';

const originalFetch = global.fetch;

type FetchResponseInit = {
  ok?: boolean;
  status?: number;
  statusText?: string;
};

function createFetchResponse(data: unknown, init: FetchResponseInit = {}) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    json: jest.fn().mockResolvedValue(data),
  } as unknown as Response;
}

function renderWithRouter() {
  return render(
    <MemoryRouter initialEntries={['/runs/S123/accounts']}>
      <Routes>
        <Route path="/runs/:sid/accounts" element={<AccountsPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe('AccountsPage', () => {
  beforeEach(() => {
    (global as typeof globalThis & { fetch: jest.Mock }).fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  afterAll(() => {
    if (originalFetch) {
      global.fetch = originalFetch;
    } else {
      delete (global as typeof globalThis).fetch;
    }
  });

  it('loads account packs and renders cards', async () => {
    const mockIndex = {
      accounts: [
        { account_id: 'acct-1', pack_path: 'frontend/accounts/acct-1/pack.json' },
      ],
    };

    const mockPack = {
      ...samplePack,
      holder_name: 'John Doe',
      primary_issue: 'wrong_account',
    };

    (global.fetch as jest.Mock).mockResolvedValueOnce(createFetchResponse(mockIndex));
    (global.fetch as jest.Mock).mockResolvedValueOnce(createFetchResponse(mockPack));

    renderWithRouter();

    expect(screen.getByText('Accounts')).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'John Doe' })).toBeInTheDocument()
    );
    expect(screen.getByText('Run S123')).toBeInTheDocument();
  });

  it('filters accounts using the search box', async () => {
    const mockIndex = {
      accounts: [
        { account_id: 'acct-1', pack_path: 'frontend/accounts/acct-1/pack.json' },
        { account_id: 'acct-2', pack_path: 'frontend/accounts/acct-2/pack.json' },
      ],
    };

    const firstPack = { ...samplePack, holder_name: 'First Bank', primary_issue: 'wrong_account' };
    const secondPack = { ...samplePack, holder_name: 'Second Bank', primary_issue: 'identity_theft' };

    (global.fetch as jest.Mock).mockResolvedValueOnce(createFetchResponse(mockIndex));
    (global.fetch as jest.Mock).mockResolvedValueOnce(createFetchResponse(firstPack));
    (global.fetch as jest.Mock).mockResolvedValueOnce(createFetchResponse(secondPack));

    renderWithRouter();

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'First Bank' })).toBeInTheDocument()
    );
    expect(screen.getByRole('heading', { name: 'Second Bank' })).toBeInTheDocument();

    const input = screen.getByLabelText('Search accounts');

    fireEvent.change(input, { target: { value: 'identity' } });

    await waitFor(() =>
      expect(screen.queryByRole('heading', { name: 'First Bank' })).not.toBeInTheDocument()
    );
    expect(screen.getByRole('heading', { name: 'Second Bank' })).toBeInTheDocument();

    fireEvent.change(input, { target: { value: 'zzz' } });

    await waitFor(() =>
      expect(screen.getByText('No accounts match your search.')).toBeInTheDocument()
    );
  });

  it('shows an error message when fetching fails', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce(
      createFetchResponse({ message: 'boom' }, { ok: false, status: 500, statusText: 'Server error' })
    );

    renderWithRouter();

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByText(/Unable to load accounts/i)).toBeInTheDocument();
    expect(screen.getByText(/boom/i)).toBeInTheDocument();
  });
});
