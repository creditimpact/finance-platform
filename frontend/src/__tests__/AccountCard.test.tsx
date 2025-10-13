import { render, screen, fireEvent } from '@testing-library/react';
import AccountCard from '../components/AccountCard';
import samplePack from '../__fixtures__/sampleAccountPack.json';

describe('AccountCard', () => {
  it('renders the summary view and bureau grid', () => {
    render(<AccountCard pack={samplePack} />);

    expect(screen.getByRole('heading', { name: 'John Doe' })).toBeInTheDocument();

    const [accountTypeLabel] = screen.getAllByText('Account type');
    const accountTypeCell = accountTypeLabel.closest('div');
    expect(accountTypeCell).toHaveTextContent('Credit Card');
    expect(accountTypeCell).toHaveTextContent('2 of 3');

    const [statusLabel] = screen.getAllByText('Status');
    const statusCell = statusLabel.closest('div');
    expect(statusCell).toHaveTextContent('Closed');
    expect(statusCell).toHaveTextContent('2 of 3');

    expect(screen.getByText('2023-01-02')).toBeInTheDocument();

    const toggle = screen.getByRole('button', { name: /details/i });
    fireEvent.click(toggle);

    expect(screen.queryByText('2023-01-02')).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.getByText('2023-01-02')).toBeInTheDocument();
  });

  it('shows placeholders for unanswered questions', () => {
    render(<AccountCard pack={samplePack} />);

    expect(screen.getAllByText('No response yet')).toHaveLength(4);
  });
});
