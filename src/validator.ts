export interface ValidationResult {
  valid: boolean;
  reason?: string;
}

const INJECTION_PATTERNS: RegExp[] = [
  /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)/i,
  /forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?|prompts?)/i,
  /disregard\s+(all\s+)?(previous|prior|above)/i,
  /you\s+are\s+now\s+(a|an)\s+\w+/i,
  /new\s+(system\s+)?instructions?\s*:/i,
  /<\|im_start\|>/,
  /<\|im_end\|>/,
  /\[\s*system\s*\]/i,
  /\bsystem\s*:\s*you\s+(are|must|should)/i,
];

export class MessageValidator {
  public validate(content: string): ValidationResult {
    if (typeof content !== 'string') {
      return { valid: false, reason: 'Content must be a string' };
    }
    if (content.length === 0) {
      return { valid: false, reason: 'Content is empty' };
    }
    for (const pattern of INJECTION_PATTERNS) {
      if (pattern.test(content)) {
        return { valid: false, reason: `Suspicious pattern detected: ${pattern.source}` };
      }
    }
    return { valid: true };
  }

  /**
   * Санитизация полей задач перед инжектом в системный контекст.
   * Убирает символы, способные сломать XML-структуру или имитировать инструкции.
   */
  public sanitizeForSystemContext(text: string): string {
    return text
      .replace(/</g, '‹')
      .replace(/>/g, '›')
      .replace(/\[/g, '(')
      .replace(/\]/g, ')')
      .replace(/\r?\n/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }
}