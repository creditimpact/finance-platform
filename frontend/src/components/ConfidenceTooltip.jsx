import React from 'react';

export default function ConfidenceTooltip({ decisionSource, confidence, fieldsUsed, showFieldsUsed = false, decimals = 2 }) {
  let lines = [];
  if (decisionSource === 'ai') {
    const value = Number(confidence);
    if (!Number.isNaN(value)) {
      lines.push(`AI confidence: ${value.toFixed(decimals)}`);
    } else {
      lines.push('AI confidence: N/A');
    }
  } else {
    lines.push('No AI used (rules-only)');
  }
  if (showFieldsUsed && Array.isArray(fieldsUsed) && fieldsUsed.length > 0) {
    lines.push(`Fields used: ${fieldsUsed.join(', ')}`);
  }
  return (
    <span className="info-icon" title={lines.join('\n')}>
      ℹ️
    </span>
  );
}
