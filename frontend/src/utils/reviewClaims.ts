import { CLAIMS, type ClaimKey } from '../constants/claims';
import type { AccountQuestionAnswers, ClaimDocuments } from '../components/AccountQuestions';

function isClaimKey(value: string): value is ClaimKey {
  return Object.prototype.hasOwnProperty.call(CLAIMS, value);
}

export function normalizeClaims(value: unknown): ClaimKey[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const normalized: ClaimKey[] = [];
  for (const entry of value) {
    if (typeof entry !== 'string') {
      continue;
    }
    const trimmed = entry.trim();
    if (!isClaimKey(trimmed) || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    normalized.push(trimmed);
  }
  return normalized;
}

export function normalizeClaimDocuments(value: unknown): ClaimDocuments {
  if (!value || typeof value !== 'object') {
    return {};
  }
  const record = value as Record<string, unknown>;
  const normalized: ClaimDocuments = {};
  for (const [claimKey, claimValue] of Object.entries(record)) {
    if (!isClaimKey(claimKey) || !claimValue || typeof claimValue !== 'object') {
      continue;
    }
    const docEntries = claimValue as Record<string, unknown>;
    const docMap: Partial<Record<string, string[]>> = {};
    for (const [docKey, docValue] of Object.entries(docEntries)) {
      const collected: string[] = [];
      const maybeArray = Array.isArray(docValue) ? docValue : [docValue];
      for (const entry of maybeArray) {
        if (typeof entry !== 'string') {
          continue;
        }
        const trimmed = entry.trim();
        if (!trimmed || collected.includes(trimmed)) {
          continue;
        }
        collected.push(trimmed);
      }
      if (collected.length > 0) {
        docMap[docKey] = collected;
      }
    }
    if (Object.keys(docMap).length > 0) {
      normalized[claimKey as ClaimKey] = docMap;
    }
  }
  return normalized;
}

export function prepareClaimDocuments(claimDocuments?: ClaimDocuments): Record<string, Record<string, string[]>> | undefined {
  if (!claimDocuments) {
    return undefined;
  }
  const prepared: Record<string, Record<string, string[]>> = {};
  for (const [claimKey, docMap] of Object.entries(claimDocuments)) {
    if (!isClaimKey(claimKey) || !docMap || typeof docMap !== 'object') {
      continue;
    }
    const normalizedDocs: Record<string, string[]> = {};
    for (const [docKey, docIds] of Object.entries(docMap)) {
      if (!docIds) {
        continue;
      }
      const array = Array.isArray(docIds) ? docIds : [docIds];
      const filtered: string[] = [];
      for (const value of array) {
        if (typeof value !== 'string') {
          continue;
        }
        const trimmed = value.trim();
        if (!trimmed || filtered.includes(trimmed)) {
          continue;
        }
        filtered.push(trimmed);
      }
      if (filtered.length > 0) {
        normalizedDocs[docKey] = filtered;
      }
    }
    if (Object.keys(normalizedDocs).length > 0) {
      prepared[claimKey] = normalizedDocs;
    }
  }
  return Object.keys(prepared).length > 0 ? prepared : undefined;
}

export function getMissingRequiredDocs(
  selectedClaims: ClaimKey[] | undefined,
  claimDocuments: ClaimDocuments | undefined
): Partial<Record<ClaimKey, string[]>> {
  const missing: Partial<Record<ClaimKey, string[]>> = {};
  if (!selectedClaims || selectedClaims.length === 0) {
    return missing;
  }
  for (const claimKey of selectedClaims) {
    const definition = CLAIMS[claimKey];
    if (!definition?.requiresDocs) {
      continue;
    }
    const docMap = claimDocuments?.[claimKey] ?? {};
    const missingDocs: string[] = [];
    for (const docKey of definition.requiredDocs) {
      const entries = docMap?.[docKey];
      if (!Array.isArray(entries) || entries.length === 0) {
        missingDocs.push(docKey);
      }
    }
    if (missingDocs.length > 0) {
      missing[claimKey] = missingDocs;
    }
  }
  return missing;
}

export function hasMissingRequiredDocs(
  selectedClaims: ClaimKey[] | undefined,
  claimDocuments: ClaimDocuments | undefined
): boolean {
  const missing = getMissingRequiredDocs(selectedClaims, claimDocuments);
  return Object.keys(missing).length > 0;
}

export function formatDocKey(docKey: string): string {
  if (!docKey) {
    return 'Document';
  }
  return docKey
    .split('_')
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ');
}

export function normalizeExistingAnswers(source: unknown): AccountQuestionAnswers {
  if (!source || typeof source !== 'object') {
    return {};
  }
  const record = source as Record<string, unknown>;
  const explanation = typeof record.explanation === 'string' ? record.explanation : undefined;
  const claims = normalizeClaims(record.claims);
  const claimDocuments = normalizeClaimDocuments(record.claim_documents ?? record.claimDocuments);
  const answers: AccountQuestionAnswers = {};
  if (explanation && explanation.trim() !== '') {
    answers.explanation = explanation;
  }
  if (claims.length > 0) {
    answers.claims = claims;
  }
  if (Object.keys(claimDocuments).length > 0) {
    answers.claimDocuments = claimDocuments;
  }
  return answers;
}

export function prepareAnswersPayload(
  answers: AccountQuestionAnswers,
  options?: { includeClaims?: boolean }
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  const explanation = answers.explanation;
  if (typeof explanation === 'string' && explanation.trim() !== '') {
    payload.explanation = explanation.trim();
  }

  if (options?.includeClaims) {
    const claims = normalizeClaims(answers.claims);
    if (claims.length > 0) {
      payload.claims = claims;
    }
    const claimDocuments = prepareClaimDocuments(answers.claimDocuments);
    if (claimDocuments) {
      payload.claim_documents = claimDocuments;
    }
  }

  return payload;
}
