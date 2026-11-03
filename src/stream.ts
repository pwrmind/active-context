import type { StreamEvent } from './types.js';

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
const TOOL_OPEN = '<tool_call>';
const TOOL_CLOSE = '</tool_call>';

type Mode = 'content' | 'think' | 'tool_call';

export class StreamRouter {
  private buffer = '';
  private mode: Mode = 'content';
  private toolBuffer = '';

  public push(chunk: string): StreamEvent[] {
    this.buffer += chunk;
    const events: StreamEvent[] = [];

    while (this.buffer.length > 0) {
      if (this.mode === 'content') {
        const tagStart = this.buffer.indexOf('<');
        if (tagStart === -1) {
          const hold = this.holdSuffix(this.buffer, [THINK_OPEN, TOOL_OPEN]);
          const emitUpTo = this.buffer.length - hold;
          if (emitUpTo > 0) {
            events.push({ type: 'content', text: this.buffer.slice(0, emitUpTo) });
            this.buffer = this.buffer.slice(emitUpTo);
          }
          break;
        }
        if (tagStart > 0) {
          events.push({ type: 'content', text: this.buffer.slice(0, tagStart) });
          this.buffer = this.buffer.slice(tagStart);
        }
        const close = this.buffer.indexOf('>');
        if (close === -1) break;
        const tag = this.buffer.slice(0, close + 1).replace(/\s+/g, '');
        this.buffer = this.buffer.slice(close + 1);

        if (tag === THINK_OPEN) {
          this.mode = 'think';
        } else if (tag === TOOL_OPEN) {
          this.mode = 'tool_call';
          this.toolBuffer = '';
        } else {
          events.push({ type: 'content', text: tag });
        }
      } else if (this.mode === 'think') {
        const close = this.buffer.indexOf(THINK_CLOSE);
        if (close === -1) {
          const hold = this.holdSuffix(this.buffer, [THINK_CLOSE]);
          const emitUpTo = this.buffer.length - hold;
          if (emitUpTo > 0) {
            events.push({ type: 'thinking', text: this.buffer.slice(0, emitUpTo) });
            this.buffer = this.buffer.slice(emitUpTo);
          }
          break;
        }
        if (close > 0) {
          events.push({ type: 'thinking', text: this.buffer.slice(0, close) });
        }
        this.buffer = this.buffer.slice(close + THINK_CLOSE.length);
        this.mode = 'content';
      } else {
        // tool_call: буферизуем полностью
        const close = this.buffer.indexOf(TOOL_CLOSE);
        if (close === -1) {
          this.toolBuffer += this.buffer;
          this.buffer = '';
          break;
        }
        this.toolBuffer += this.buffer.slice(0, close);
        events.push({
          type: 'tool_call',
          text: TOOL_OPEN + this.toolBuffer + TOOL_CLOSE,
        });
        this.toolBuffer = '';
        this.buffer = this.buffer.slice(close + TOOL_CLOSE.length);
        this.mode = 'content';
      }
    }

    return events;
  }

  public flush(): StreamEvent[] {
    const events: StreamEvent[] = [];
    if (this.mode === 'content' && this.buffer) {
      events.push({ type: 'content', text: this.buffer });
    } else if (this.mode === 'think' && this.buffer) {
      events.push({ type: 'thinking', text: this.buffer });
    } else if (this.mode === 'tool_call' && (this.toolBuffer || this.buffer)) {
      // Незакрытый tool_call — эмитим как контент для диагностики
      events.push({ type: 'content', text: TOOL_OPEN + this.toolBuffer + this.buffer });
    }
    this.buffer = '';
    this.toolBuffer = '';
    this.mode = 'content';
    return events;
  }

  private holdSuffix(text: string, tags: string[]): number {
    let maxHold = 0;
    for (const tag of tags) {
      for (let len = Math.min(text.length, tag.length - 1); len > 0; len--) {
        if (text.endsWith(tag.slice(0, len))) {
          if (len > maxHold) maxHold = len;
          break;
        }
      }
    }
    return maxHold;
  }
}