import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import StatusPage from './pages/StatusPage';
import ReviewPage from './pages/ReviewPage';
import RunReviewPage from './pages/RunReviewPage';
import AccountsPage from './pages/Accounts';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Home</Link>
        <Link to="/upload">Upload</Link>
      </nav>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/status" element={<StatusPage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/runs/:sid/review" element={<RunReviewPage />} />
        <Route path="/runs/:sid/accounts" element={<AccountsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
