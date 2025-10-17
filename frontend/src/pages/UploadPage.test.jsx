import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import UploadPage from './UploadPage';

jest.mock('../api.ts', () => ({
  uploadReport: jest.fn(),
  pollResult: jest.fn(),
  getAccount: jest.fn(),
}));

describe('UploadPage', () => {
  test('only renders email and PDF file inputs', () => {
    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>
    );
    expect(screen.getByLabelText(/Email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/PDF File/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/story/i)).toBeNull();
    expect(screen.queryByLabelText(/note/i)).toBeNull();
  });
});
