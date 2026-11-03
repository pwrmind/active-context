import { randomUUID } from 'node:crypto';
import type {
  ContextMessage,
  ContextStorageStats,
  MessageRole,
  SerializedContext,
  TokenizerFn,
} from './types.js';
import { defaultTokenizer } from './tokenizer.js';

export interface LLMContextOptions {
  totalTokensLimit?: number;
  tokenizer?: TokenizerFn;
}

const VALID_ROLES: MessageRole[] = ['user', 'assistant', 'tool'];

export class LLMContext {
  private systemPrompt: string | null = null;
  private systemPromptTokens = 0;
  private reservedTokens = 0;

  private messages: ContextMessage[] = [];
  private messageMap = new Map<string, ContextMessage>();

  private readonly totalTokensLimit: number;
  private readonly tokenizer: TokenizerFn;

  constructor(options: LLMContextOptions = {}) {
    this.totalTokensLimit = options.totalTokensLimit ?? 128_000;
    this.tokenizer = options.tokenizer ?? defaultTokenizer;
  }

  // --- Состояние ---

  public get stats(): ContextStorageStats {
    const historyTokens = this.messages.reduce((sum, m) => sum + m.tokensCount, 0);
    const currentTokensUsed = this.systemPromptTokens + this.reservedTokens + historyTokens;
    return {
      totalTokensLimit: this.totalTokensLimit,
      systemPromptTokens: this.systemPromptTokens,
      reservedTokens: this.reservedTokens,
      historyTokens,
      currentTokensUsed,
      availableTokens: Math.max(0, this.totalTokensLimit - currentTokensUsed),
    };
  }

  public get isFull(): boolean {
    return this.stats.currentTokensUsed >= this.totalTokensLimit;
  }

  // --- Резерв под внешние системные блоки (TaskManager, RAG и т.п.) ---

  public setReservedTokens(tokens: number): void {
    this.reservedTokens = Math.max(0, tokens);
    this.enforceLimit();
  }

  // --- System Prompt (изолирован) ---

  public setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
    this.systemPromptTokens = this.tokenizer(prompt);
    this.enforceLimit();
  }

  public getSystemPrompt(): string | null {
    return this.systemPrompt;
  }

  // --- CRUD ---

  public appendMessage(
    role: MessageRole,
    content: string,
    metadata?: Record<string, unknown>,
  ): string {
    const tokensCount = this.tokenizer(content);
    const id = randomUUID();
    const message: ContextMessage = {
      id,
      role,
      content,
      timestamp: Date.now(),
      tokensCount,
      metadata,
    };

    this.messages.push(message);
    this.messageMap.set(id, message);
    this.enforceLimit();
    return id;
  }

  public getMessageById(id: string): ContextMessage | undefined {
    return this.messageMap.get(id);
  }

  public updateMessageContent(id: string, newContent: string): void {
    const msg = this.messageMap.get(id);
    if (!msg) throw new Error(`Message ${id} not found`);
    msg.content = newContent;
    msg.tokensCount = this.tokenizer(newContent);
    this.enforceLimit();
  }

  public deleteMessage(id: string): boolean {
    if (!this.messageMap.has(id)) return false;
    this.messageMap.delete(id);
    this.messages = this.messages.filter((m) => m.id !== id);
    return true;
  }

  public findMessages(options: {
    query?: string;
    role?: MessageRole;
    limit?: number;
  }): ContextMessage[] {
    let result = this.messages;
    if (options.role) result = result.filter((m) => m.role === options.role);
    if (options.query) {
      const q = options.query.toLowerCase();
      result = result.filter((m) => m.content.toLowerCase().includes(q));
    }
    if (options.limit && options.limit > 0) result = result.slice(-options.limit);
    return result;
  }

  // --- Пайплайн ---

  public toHistoryOnly(): Array<{ role: string; content: string }> {
    return this.messages.map((m) => ({ role: m.role, content: m.content }));
  }

  public toPipelineFormat(): Array<{ role: string; content: string }> {
    const pipeline: Array<{ role: string; content: string }> = [];
    if (this.systemPrompt) {
      pipeline.push({ role: 'system', content: this.systemPrompt });
    }
    pipeline.push(...this.toHistoryOnly());
    return pipeline;
  }

  // --- Serialization ---

  /**
   * Снимок состояния для сохранения на диск.
   * Возвращает защищённую копию — вызывающий код не может мутировать
   * внутренние объекты через возвращённую структуру.
   */
  public export(): SerializedContext {
    return {
      version: 1,
      systemPrompt: this.systemPrompt,
      reservedTokens: this.reservedTokens,
      messages: this.messages.map((m) => ({ ...m, metadata: m.metadata ? { ...m.metadata } : undefined })),
    };
  }

  /**
   * Восстанавливает состояние из снимка.
   *
   * Сначала валидирует весь вход, затем применяет атомарно.
   * При любой ошибке валидации состояние класса остаётся нетронутым.
   *
   * `tokensCount` берётся из снимка, если он положителен; иначе
   * пересчитывается текущим токенизатором (защита от испорченных данных).
   */
  public import(data: SerializedContext): void {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid context data: expected an object');
    }
    if (data.version !== 1) {
      throw new Error(`Unsupported context version: ${String(data.version)}`);
    }
    if (!Array.isArray(data.messages)) {
      throw new Error('Invalid context data: "messages" must be an array');
    }
    if (data.systemPrompt !== null && typeof data.systemPrompt !== 'string') {
      throw new Error('Invalid context data: "systemPrompt" must be a string or null');
    }

    const validated: ContextMessage[] = [];
    for (const [i, m] of data.messages.entries()) {
      if (!m || typeof m !== 'object') {
        throw new Error(`Invalid message at index ${i}: expected an object`);
      }
      if (typeof m.id !== 'string' || m.id.length === 0) {
        throw new Error(`Invalid message at index ${i}: "id" must be a non-empty string`);
      }
      if (typeof m.content !== 'string') {
        throw new Error(`Invalid message at index ${i}: "content" must be a string`);
      }
      if (!VALID_ROLES.includes(m.role)) {
        throw new Error(
          `Invalid message at index ${i}: "role" must be one of ${VALID_ROLES.join(', ')}`,
        );
      }
      const storedTokens = typeof m.tokensCount === 'number' && m.tokensCount > 0
        ? m.tokensCount
        : this.tokenizer(m.content);

      validated.push({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: typeof m.timestamp === 'number' ? m.timestamp : Date.now(),
        tokensCount: storedTokens,
        metadata: m.metadata,
      });
    }

    // --- Atomic apply ---
    this.systemPrompt = data.systemPrompt;
    this.systemPromptTokens = data.systemPrompt ? this.tokenizer(data.systemPrompt) : 0;
    this.reservedTokens =
      typeof data.reservedTokens === 'number' && data.reservedTokens >= 0
        ? data.reservedTokens
        : 0;
    this.messages = validated;
    this.messageMap = new Map(validated.map((m) => [m.id, m]));
    this.enforceLimit();
  }

  // --- Eviction ---

  public evictOldest(tokensToFree: number): ContextMessage[] {
    const evicted: ContextMessage[] = [];
    while (tokensToFree > 0 && this.messages.length > 0) {
      const idx = this.messages.findIndex((m) => m.role !== 'tool');
      if (idx === -1) break;
      const [removed] = this.messages.splice(idx, 1);
      this.messageMap.delete(removed.id);
      tokensToFree -= removed.tokensCount;
      evicted.push(removed);
    }
    return evicted;
  }

  public clear(): void {
    this.messages = [];
    this.messageMap.clear();
    // System prompt и reserved намеренно сохраняются
  }

  private enforceLimit(): void {
    while (this.isFull && this.messages.length > 0) {
      const removed = this.evictOldest(1);
      if (removed.length === 0) break;
    }
  }
}