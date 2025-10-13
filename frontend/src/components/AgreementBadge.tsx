import * as React from 'react';
import { Badge } from './ui/badge';
import { AgreementLevel } from './accountFieldTypes';
import { cn } from '../lib/utils';

const LABELS: Record<AgreementLevel, string> = {
  all: 'All agree',
  two: '2 of 3',
  mixed: 'Mixed',
  none: '—'
};

const TONE_STYLES: Record<AgreementLevel, string> = {
  all: 'border-transparent bg-slate-200 text-slate-700',
  two: 'border-transparent bg-sky-100 text-sky-800',
  mixed: 'border-transparent bg-amber-100 text-amber-900',
  none: 'border-transparent bg-slate-100 text-slate-500'
};

export interface AgreementBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  agreement: AgreementLevel;
}

export function AgreementBadge({ agreement, className, ...props }: AgreementBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn('px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide', TONE_STYLES[agreement], className)}
      {...props}
    >
      {LABELS[agreement]}
    </Badge>
  );
}

export default AgreementBadge;
