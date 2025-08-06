import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { startProcess } from '../api';

export default function UploadPage() {
  const [email, setEmail] = useState('');
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !file) {
      setError('Please provide an email and PDF file.');
      return;
    }
    setUploading(true);
    setError('');
    try {
      const data = await startProcess(email, file);
      navigate('/review', { state: { uploadData: { ...data, email } } });
    } catch (err) {
      console.error(err);
      setError('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container">
      <h2>Upload Credit Report</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div>
          <label>PDF File:</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>
        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading...' : 'Start Processing'}
        </button>
      </form>
    </div>
  );
}
