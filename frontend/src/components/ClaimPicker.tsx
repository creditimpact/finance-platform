import * as React from 'react';
import { CLAIMS, type ClaimKey } from '../constants/claims';
import { formatDocKey } from '../utils/reviewClaims';
import { cn } from '../lib/utils';

type Props = {
  selected: ClaimKey[];
  onChange: (next: ClaimKey[]) => void;
  onFilesSelected: (claim: ClaimKey, docKey: string, files: File[]) => void;
  uploadedMap: Record<ClaimKey, Record<string, string[]>>;
};

function ClaimPicker({ selected, onChange, onFilesSelected, uploadedMap }: Props) {
  const handleToggle = React.useCallback(
    (claim: ClaimKey) => {
      const isSelected = selected.includes(claim);
      const nextClaims = isSelected
        ? selected.filter((entry) => entry !== claim)
        : [...selected, claim];
      onChange(nextClaims);
    },
    [onChange, selected]
  );

  const handleFileChange = React.useCallback(
    (claim: ClaimKey, docKey: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = event.target.files ? Array.from(event.target.files) : [];
      event.target.value = '';
      if (fileList.length === 0) {
        return;
      }
      onFilesSelected(claim, docKey, fileList);
    },
    [onFilesSelected]
  );

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        {Object.entries(CLAIMS).map(([key, definition]) => {
          const claimKey = key as ClaimKey;
          const checked = selected.includes(claimKey);
          return (
            <label
              key={claimKey}
              className={cn(
                'flex cursor-pointer flex-col gap-2 rounded-md border border-slate-200 bg-white p-3 transition',
                checked ? 'border-slate-400 ring-1 ring-slate-400' : 'hover:border-slate-300'
              )}
            >
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500"
                  checked={checked}
                  onChange={() => handleToggle(claimKey)}
                />
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-900">{definition.label}</p>
                  {definition.hint ? (
                    <p className="text-xs text-slate-600">{definition.hint}</p>
                  ) : null}
                  <p className="text-xs text-slate-500">
                    {definition.requiresDocs
                      ? `Requires ${definition.requiredDocs.length} ${
                          definition.requiredDocs.length === 1 ? 'document' : 'documents'
                        }.`
                      : 'No extra documents required.'}
                  </p>
                </div>
              </div>
            </label>
          );
        })}
      </div>
      <p className="text-xs text-slate-500">
        Your basic ID &amp; Proof of Address are automatically attached to all disputes.
      </p>
      <div className="space-y-4">
        {selected.map((claimKey) => {
          const definition = CLAIMS[claimKey];
          if (!definition?.requiresDocs) {
            return null;
          }
          const docsForClaim = uploadedMap[claimKey] ?? {};
          return (
            <details
              key={claimKey}
              className="rounded-lg border border-slate-200 bg-slate-50 p-4"
              open
            >
              <summary className="cursor-pointer text-sm font-semibold text-slate-900">
                Upload documents for {definition.label}
              </summary>
              <div className="mt-3 space-y-3">
                {definition.requiredDocs.map((docKey) => {
                  const uploaded = docsForClaim[docKey] ?? [];
                  const hasDocs = Array.isArray(uploaded) && uploaded.length > 0;
                  return (
                    <div key={docKey} className="rounded-md border border-slate-200 bg-white p-3">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-slate-900">{formatDocKey(docKey)}</p>
                        {hasDocs ? (
                          <span role="img" aria-label="Document uploaded" className="text-base">
                            ✅
                          </span>
                        ) : null}
                      </div>
                      <input
                        type="file"
                        multiple
                        onChange={handleFileChange(claimKey, docKey)}
                        className="mt-2 block text-sm text-slate-700 file:mr-3 file:rounded-md file:border file:border-slate-300 file:bg-white file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-50"
                      />
                    </div>
                  );
                })}
              </div>
            </details>
          );
        })}
      </div>
    </div>
  );
}

export default ClaimPicker;
