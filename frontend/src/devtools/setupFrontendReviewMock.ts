function getMetaEnv(): Record<string, any> {
  try {
    return (typeof import !== 'undefined' && import.meta && import.meta.env) || {};
  } catch (err) {
    return {};
  }
}

function getProcessEnv(): Record<string, any> {
  if (typeof process !== 'undefined' && process.env) {
    return process.env;
  }
  return {};
}

const metaEnv = getMetaEnv();
const processEnv = getProcessEnv();

const mockFlag =
  metaEnv.VITE_ENABLE_FRONTEND_REVIEW_MOCK ??
  metaEnv.VITE_FRONTEND_REVIEW_MOCK ??
  processEnv.VITE_ENABLE_FRONTEND_REVIEW_MOCK ??
  processEnv.VITE_FRONTEND_REVIEW_MOCK;

const shouldEnableMock = mockFlag === '1' || (mockFlag === undefined && Boolean(metaEnv.DEV));

if (shouldEnableMock && typeof window !== 'undefined') {
  import('../mocks/frontendReviewMock')
    .then((mod) => {
      if (mod && typeof mod.enableFrontendReviewMock === 'function') {
        mod.enableFrontendReviewMock();
      }
    })
    .catch((err) => {
      if (metaEnv.DEV) {
        console.warn('Failed to enable frontend review mock', err);
      }
    });
}

export {};
