import { CLAIMS, type ClaimKey } from '../constants/claims';
import type { AccountQuestionAnswers, ClaimDocuments } from '../components/AccountQuestions';
import type { ReviewClaimEvidence, SubmitReviewPayload } from '../api';

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

export function prepareClaimEvidence(
  claimDocuments?: ClaimDocuments
): ReviewClaimEvidence[] | undefined {
  if (!claimDocuments) {
    return undefined;
  }

  const evidence: ReviewClaimEvidence[] = [];

  for (const [claimKey, docMap] of Object.entries(claimDocuments)) {
    if (!isClaimKey(claimKey) || !docMap || typeof docMap !== 'object') {
      continue;
    }

    const docs: ReviewClaimEvidence['docs'] = [];

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
        docs.push({ doc_key: docKey, doc_ids: filtered });
      }
    }

    if (docs.length > 0) {
      evidence.push({ claim: claimKey, docs });
    }
  }

  return evidence.length > 0 ? evidence : undefined;
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

function convertEvidenceToClaimDocuments(evidence: unknown): ClaimDocuments {
  if (!Array.isArray(evidence)) {
    return {};
  }

  const documents: ClaimDocuments = {};

  for (const entry of evidence) {
    if (!entry || typeof entry !== 'object') {
      continue;
    }

    const record = entry as { claim?: unknown; docs?: unknown };
    const claimKey = typeof record.claim === 'string' ? record.claim.trim() : undefined;
    if (!claimKey || !isClaimKey(claimKey)) {
      continue;
    }

    const docEntries = Array.isArray(record.docs) ? record.docs : [];
    const docMap: Partial<Record<string, string[]>> = documents[claimKey] ?? {};

    for (const docEntry of docEntries) {
      if (!docEntry || typeof docEntry !== 'object') {
        continue;
      }

      const docRecord = docEntry as { doc_key?: unknown; doc_ids?: unknown };
      const docKey = typeof docRecord.doc_key === 'string' ? docRecord.doc_key : undefined;
      if (!docKey) {
        continue;
      }

      const array = Array.isArray(docRecord.doc_ids) ? docRecord.doc_ids : [docRecord.doc_ids];
      const collected: string[] = docMap[docKey] ? [...docMap[docKey]!] : [];

      for (const value of array) {
        if (typeof value !== 'string') {
          continue;
        }
        const trimmed = value.trim();
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
      documents[claimKey] = docMap;
    }
  }

  return documents;
}

function mergeClaimDocuments(
  primary: ClaimDocuments | undefined,
  secondary: ClaimDocuments | undefined
): ClaimDocuments {
  if (!primary && !secondary) {
    return {};
  }

  const merged: ClaimDocuments = {};

  const apply = (source?: ClaimDocuments) => {
    if (!source) {
      return;
    }
    for (const [claimKey, docMap] of Object.entries(source)) {
      if (!isClaimKey(claimKey) || !docMap) {
        continue;
      }
      const target: Partial<Record<string, string[]>> = merged[claimKey] ?? {};
      for (const [docKey, docIds] of Object.entries(docMap)) {
        if (!Array.isArray(docIds) || docIds.length === 0) {
          continue;
        }
        const collected = target[docKey] ? new Set(target[docKey]) : new Set<string>();
        for (const value of docIds) {
          if (typeof value !== 'string') {
            continue;
          }
          const trimmed = value.trim();
          if (!trimmed) {
            continue;
          }
          collected.add(trimmed);
        }
        if (collected.size > 0) {
          target[docKey] = Array.from(collected);
        }
      }
      if (Object.keys(target).length > 0) {
        merged[claimKey] = target;
      }
    }
  };

  apply(secondary);
  apply(primary);

  return merged;
}

export function normalizeExistingAnswers(source: unknown): AccountQuestionAnswers {
  if (!source || typeof source !== 'object') {
    return {};
  }

  const record = source as Record<string, unknown>;
  const answersSection =
    record.answers && typeof record.answers === 'object' && !Array.isArray(record.answers)
      ? (record.answers as Record<string, unknown>)
      : undefined;

  const explanationFromRoot = typeof record.explanation === 'string' ? record.explanation : undefined;
  const explanationFromSection =
    answersSection && typeof answersSection.explanation === 'string'
      ? (answersSection.explanation as string)
      : undefined;

  const claims = normalizeClaims(
    record.claims ?? (answersSection?.claims as unknown)
  );

  const legacyClaimDocuments = normalizeClaimDocuments(
    record.claim_documents ??
      record.claimDocuments ??
      answersSection?.claim_documents ??
      answersSection?.claimDocuments
  );

  const evidenceDocuments = convertEvidenceToClaimDocuments(record.evidence ?? answersSection?.evidence);

  const answers: AccountQuestionAnswers = {};

  const explanation = explanationFromRoot ?? explanationFromSection;
  if (explanation && explanation.trim() !== '') {
    answers.explanation = explanation;
  }

  if (claims.length > 0) {
    answers.claims = claims;
  }

  const mergedDocuments = mergeClaimDocuments(legacyClaimDocuments, evidenceDocuments);
  if (Object.keys(mergedDocuments).length > 0) {
    answers.claimDocuments = mergedDocuments;
  }

  return answers;
}

export function prepareAnswersPayload(
  answers: AccountQuestionAnswers,
  options?: { includeClaims?: boolean }
): SubmitReviewPayload {
  const explanation = answers.explanation;
  const answersPayload: SubmitReviewPayload['answers'] = {};
  if (typeof explanation === 'string' && explanation.trim() !== '') {
    answersPayload.explanation = explanation.trim();
  }

  const payload: SubmitReviewPayload = { answers: answersPayload };

  if (options?.includeClaims) {
    const claims = normalizeClaims(answers.claims);
    if (claims.length > 0) {
      payload.claims = claims;
    }
    const evidence = prepareClaimEvidence(answers.claimDocuments);
    if (evidence) {
      payload.evidence = evidence;
    }
  }

  return payload;
}
