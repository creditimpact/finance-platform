import * as React from 'react';
import { cn } from '../lib/utils';
import { AgreementLevel, BUREAUS, BUREAU_LABELS, BureauKey, MISSING_VALUE } from './accountFieldTypes';

export interface BureauGridField {
  fieldKey: string;
  label: string;
  values: Partial<Record<BureauKey, string>>;
  agreement: AgreementLevel;
}

export interface BureauGridProps {
  fields: BureauGridField[];
  className?: string;
}

export function BureauGrid({ fields, className }: BureauGridProps) {
  return (
    <div className={cn('overflow-hidden rounded-lg border border-slate-200', className)}>
      <div className="grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))] bg-slate-50 text-sm font-medium text-slate-600">
        <div className="px-4 py-3 text-left">Field</div>
        {BUREAUS.map((bureau) => (
          <div key={bureau} className="px-4 py-3 text-center">
            {BUREAU_LABELS[bureau]}
          </div>
        ))}
      </div>
      <div className="divide-y divide-slate-200 text-sm">
        {fields.map((field) => {
          const highlight = field.agreement === 'majority' || field.agreement === 'mixed';

          return (
            <div
              key={field.fieldKey}
              className={cn(
                'grid grid-cols-[minmax(150px,1.2fr)_repeat(3,minmax(0,1fr))]',
                highlight && 'bg-amber-50'
              )}
            >
              <div className="px-4 py-3 font-medium text-slate-700">{field.label}</div>
              {BUREAUS.map((bureau) => {
                const value = field.values[bureau] ?? MISSING_VALUE;
                const isMissing = value === MISSING_VALUE;

                return (
                  <div
                    key={bureau}
                    className={cn('px-4 py-3 text-center', isMissing ? 'text-slate-500' : 'text-slate-800')}
                  >
                    {isMissing ? MISSING_VALUE : value}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default BureauGrid;
