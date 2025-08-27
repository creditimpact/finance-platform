export const emitUiEvent = (event, payload) => {
  const body = { event, ...payload };
  console.debug('telemetry', body);
  try {
    if (typeof navigator !== 'undefined' && navigator.sendBeacon) {
      navigator.sendBeacon('/api/ui-event', JSON.stringify(body));
    } else if (typeof fetch === 'function') {
      fetch('/api/ui-event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // ignore telemetry errors
  }
};
