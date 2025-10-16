import { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error?: Error;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: undefined };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in ErrorBoundary:', error, errorInfo);
  }

  render() {
    const { error } = this.state;

    if (error) {
      return (
        <div
          role="alert"
          style={{
            backgroundColor: '#fee2e2',
            border: '1px solid #ef4444',
            borderRadius: '0.5rem',
            color: '#991b1b',
            padding: '1.5rem',
            margin: '1.5rem',
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          }}
        >
          <h1 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>UI crashed</h1>
          <p style={{ margin: 0 }}>
            UI crashed: <strong>{error.message}</strong>
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
