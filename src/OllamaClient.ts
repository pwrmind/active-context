/**
 * @deprecated The custom `OllamaClient` class has been removed in favor of
 * `OllamaJsClient` (native tool calling). `OllamaClientLike` is now defined
 * in `types.ts`. This file remains only as a re-export shim for backward
 * compatibility with existing imports.
 *
 * Migration: use `new OllamaJsClient(baseUrl)` or pass a custom client via
 * `deps.ollamaClient`.
 */
export type { OllamaClientLike } from './types.js';