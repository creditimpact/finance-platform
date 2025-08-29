const API = (import.meta as any).env.VITE_API_URL ?? 'http://localhost:5000';

export async function uploadReport(email: string, file: File) {
  const fd = new FormData();
  fd.append('email', email);
  fd.append('file', file);

  const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
  let data: any = {};
  try {
    data = await res.json();
  } catch (_) {}
  if (!res.ok || !data?.ok || !data?.session_id) {
    const msg = data?.message || `Upload failed (status ${res.status})`;
    throw new Error(msg);
  }
  return data as { ok: true; status: string; session_id: string; task_id?: string };
}

export async function pollResult(sessionId: string, abortSignal?: AbortSignal) {
  while (true) {
    try {
      const res = await fetch(`${API}/api/result?session_id=${encodeURIComponent(sessionId)}`, { signal: abortSignal });
      let data: any = null;
      try { data = await res.json(); } catch {}

      if (res.status === 404) {
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      if (!res.ok) {
        throw new Error(data?.message || `Result request failed (${res.status})`);
      }
      if (data?.ok && data?.status === 'done') {
        return data.result;
      }
      if (data?.ok && (data?.status === 'queued' || data?.status === 'processing')) {
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      throw new Error(data?.message || 'Processing error');
    } catch (_) {
      // Treat transient errors as in-progress and keep waiting
      await new Promise((r) => setTimeout(r, 3000));
      continue;
    }
  }
}
