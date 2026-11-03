import type { TokenizerFn } from './types.js';

/**
 * Грубая эвристика (1 токен ≈ 4 символа).
 * Замените на tiktoken или токенизатор конкретной модели для точности.
 */
export const defaultTokenizer: TokenizerFn = (text: string): number => {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
};