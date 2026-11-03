import { OllamaError } from './errors.js';
import type { OllamaChatRequest, OllamaChatResponse } from './types.js';

export interface OllamaClientLike {
  chat(req: OllamaChatRequest, opts?: { signal?: AbortSignal }): Promise<OllamaChatResponse>;
  chatStream(req: OllamaChatRequest, opts?: { signal?: AbortSignal }): AsyncGenerator<string>;
}

export class OllamaClient implements OllamaClientLike {
  constructor(private readonly baseUrl: string) {}

  public async chat(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): Promise<OllamaChatResponse> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...req, stream: false }),
      signal: opts.signal,
    });
    if (!res.ok) {
      throw new OllamaError(`Ollama API error ${res.status}: ${await res.text()}`, res.status);
    }
    const data = (await res.json()) as { message?: { content?: string } };
    return { content: data.message?.content ?? '' };
  }

  public async *chatStream(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<string> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...req, stream: true }),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) {
      throw new OllamaError(`Ollama stream error ${res.status}`, res.status);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const parsed = JSON.parse(trimmed) as {
              message?: { content?: string };
              done?: boolean;
            };
            if (parsed.message?.content) yield parsed.message.content;
            if (parsed.done) return;
          } catch {
            // неполная строка — попадёт в следующий батч
          }
        }
      }
      if (buffer.trim()) {
        try {
          const parsed = JSON.parse(buffer) as { message?: { content?: string } };
          if (parsed.message?.content) yield parsed.message.content;
        } catch {
          /* ignored */
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}