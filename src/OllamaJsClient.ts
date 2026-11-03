// src/OllamaJsClient.ts
import { Ollama, type Message } from 'ollama';
import type { OllamaClientLike } from './OllamaClient.js';
import type { OllamaChatRequest, OllamaChatResponse } from './types.js';

/**
 * Адаптер над официальной библиотекой `ollama`.
 * Заменяет самописный NDJSON-парсинг, сохраняя контракт `OllamaClientLike`.
 *
 * Важно: `ollama-js` отдаёт `part.message.content` в том же порядке, в котором
 * Ollama шлёт чанки, поэтому `StreamRouter` продолжает корректно ловить
 * границы `<think>` / `<tool_call>`.
 */
export class OllamaJsClient implements OllamaClientLike {
  private readonly client: Ollama;

  constructor(host?: string) {
    this.client = new Ollama({ host: host ?? 'http://127.0.0.1:11434' });
  }

  async chat(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): Promise<OllamaChatResponse> {
    const res = await this.client.chat(
      {
        model: req.model,
        messages: req.messages as unknown as Message[],
        options: req.options as any,
      },
      { signal: opts.signal },
    );
    // При stream: false ollama-js возвращает ChatResponse с message.content.
    const content = (res as { message?: { content?: string } }).message?.content ?? '';
    return { content };
  }

  async *chatStream(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<string> {
    const stream = await this.client.chat(
      {
        model: req.model,
        messages: req.messages as unknown as Message[],
        options: req.options as any,
        stream: true,
      },
      { signal: opts.signal },
    );

    for await (const part of stream as AsyncIterable<{ message?: { content?: string } }>) {
      const chunk = part.message?.content;
      if (chunk) yield chunk;
    }
  }
}