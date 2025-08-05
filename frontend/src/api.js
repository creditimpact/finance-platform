const API_BASE_URL = 'http://localhost:5000';

export async function startProcess(email, file) {
  const formData = new FormData();
  formData.append('email', email);
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/start-process`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to start process');
  }

  return response.json();
}

export async function checkStatus(taskId) {
  const response = await fetch(`${API_BASE_URL}/api/status/${taskId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch status');
  }
  return response.json();
}
