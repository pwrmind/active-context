import { Ollama } from 'ollama';
import type {
  NativeToolCall,
  OllamaChatRequest,
  OllamaChatResponse,
  OllamaClientLike,
  StreamChunk,
} from './types.js';

interface RawToolCallChunk {
  id?: string;
  function?: {
    index?: number;
    name?: string;
    arguments?: string | Record<string, unknown>;
  };
}

export class OllamaJsClient implements OllamaClientLike {
  private readonly client: Ollama;

  constructor(host?: string) {
    this.client = new Ollama({ host: host ?? 'http://127.0.0.1:11434' });
  }

  public async chat(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): Promise<OllamaChatResponse> {
    const res = await this.client.chat(
      {
        model: req.model,
        messages: req.messages as any,
        options: req.options as any,
        tools: req.tools as any,
        think: req.think,
      },
      { signal: opts.signal },
    );

    const msg = (res as any).message ?? {};
    return {
      content: typeof msg.content === 'string' ? msg.content : '',
      thinking: typeof msg.thinking === 'string' ? msg.thinking : '',
      toolCalls: this.mapToolCalls(msg.tool_calls),
    };
  }

  public async *chatStream(
    req: OllamaChatRequest,
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<StreamChunk> {
    const stream = await this.client.chat(
      {
        model: req.model,
        messages: req.messages as any,
        options: req.options as any,
        tools: req.tools as any,
        think: req.think,
        stream: true,
      },
      { signal: opts.signal },
    );

    // Аккумулятор tool_calls: идентификатор — function.index
    const acc = new Map<number, { id?: string; name?: string; argsJson: string }>();

    for await (const part of stream as AsyncIterable<any>) {
      const msg = part.message ?? {};

      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls as RawToolCallChunk[]) {
          const idx = tc.function?.index ?? acc.size;
          let a = acc.get(idx);
          if (!a) {
            a = { argsJson: '' };
            acc.set(idx, a);
          }
          if (tc.id) a.id = tc.id;
          if (tc.function?.name) a.name = tc.function.name;

          const args = tc.function?.arguments;
          if (typeof args === 'string') {
            a.argsJson += args;
          } else if (args && typeof args === 'object') {
            a.argsJson = JSON.stringify(args);
          }
        }
      }

      const thinking = typeof msg.thinking === 'string' ? msg.thinking : '';
      const content = typeof msg.content === 'string' ? msg.content : '';
      if (thinking || content) {
        yield { content, thinking, toolCalls: [] };
      }
    }

    if (acc.size > 0) {
      const toolCalls: NativeToolCall[] = [];
      const sorted = [...acc.entries()].sort((x, y) => x[0] - y[0]);
      for (const [idx, a] of sorted) {
        if (!a.name) continue;
        toolCalls.push({
          id: a.id ?? `call_${idx}`,
          name: a.name,
          arguments: safeParseArgs(a.argsJson),
        });
      }
      if (toolCalls.length > 0) {
        yield { content: '', thinking: '', toolCalls };
      }
    }
  }

  private mapToolCalls(raw: RawToolCallChunk[] | undefined): NativeToolCall[] {
    if (!Array.isArray(raw)) return [];
    return raw.map((tc, i) => {
      const args = tc.function?.arguments;
      return {
        id: tc.id ?? `call_${i}`,
        name: tc.function?.name ?? '',
        arguments:
          typeof args === 'string'
            ? safeParseArgs(args)
            : (args ?? {}),
      };
    });
  }
}

function safeParseArgs(s: string): Record<string, unknown> {
  if (!s) return {};
  try {
    const parsed = JSON.parse(s);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}