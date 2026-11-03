export type MessageRole = 'user' | 'assistant' | 'tool';
export type TaskStatus = 'todo' | 'in_progress' | 'completed' | 'failed';

export interface NativeToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required?: string[];
    };
  };
}

export interface ContextMessage {
  readonly id: string;
  readonly role: MessageRole;
  content: string;
  readonly timestamp: number;
  tokensCount: number;
  toolCalls?: Array<{
    id: string;
    function: { name: string; arguments: Record<string, unknown> };
  }>;
  toolCallId?: string;
  metadata?: Record<string, unknown>;
}

export interface AppendMessageOptions {
  toolCalls?: ContextMessage['toolCalls'];
  toolCallId?: string;
  metadata?: Record<string, unknown>;
}

export interface ContextStorageStats {
  totalTokensLimit: number;
  systemPromptTokens: number;
  reservedTokens: number;
  historyTokens: number;
  currentTokensUsed: number;
  availableTokens: number;
}

export interface SerializedContext {
  version: 1;
  systemPrompt: string | null;
  reservedTokens: number;
  messages: ContextMessage[];
}

export interface Task {
  readonly id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  result?: string;
  parentId?: string;
  readonly createdAt: number;
  updatedAt: number;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: Array<{
    id: string;
    function: { name: string; arguments: Record<string, unknown> };
  }>;
  tool_call_id?: string;
}

export interface StreamChunk {
  content: string;
  thinking: string;
  toolCalls: NativeToolCall[];
}

export interface OllamaChatRequest {
  model: string;
  messages: ChatMessage[];
  options?: Record<string, unknown>;
  tools?: ToolDefinition[];
  think?: boolean;
}

export interface OllamaChatResponse {
  content: string;
  thinking: string;
  toolCalls: NativeToolCall[];
}

export interface OllamaClientLike {
  chat(req: OllamaChatRequest, opts?: { signal?: AbortSignal }): Promise<OllamaChatResponse>;
  chatStream(req: OllamaChatRequest, opts?: { signal?: AbortSignal }): AsyncGenerator<StreamChunk>;
}

export type TokenizerFn = (text: string) => number;

export interface ToolExecutionResult {
  call: NativeToolCall;
  success: boolean;
  message: string;
}

export interface AskOptions {
  onThinking?: (chunk: string) => void;
  onContent?: (chunk: string) => void;
  onToolCall?: (result: ToolExecutionResult) => void;
  maxIterations?: number;
  streamMode?: 'buffered' | 'realtime';
  signal?: AbortSignal;
  allowUnsafe?: boolean;
}

export interface AskResult {
  content: string;
  thinking: string;
  toolCalls: ToolExecutionResult[];
  iterations: number;
  stats: ContextStorageStats;
}

export interface SearchMessagesOptions {
  /** Литеральный поиск (case-insensitive substring). Оставлен для обратной совместимости. */
  query?: string;
  /** Regex-поиск (case-insensitive). Максимум 200 символов. */
  pattern?: string;
  role?: MessageRole;
  /** Максимум результатов. По умолчанию — без ограничения. При tool-call capped до 50. */
  limit?: number;
}

// Legacy: for LFM3Parser / StreamRouter, kept for backward compatibility
export interface ParsedToolCall {
  functionName: string;
  args: Record<string, string>;
  raw: string;
}

export interface ParsedResponse {
  thinking: string;
  cleaned: string;
  calls: ParsedToolCall[];
}

export type StreamEvent =
  | { type: 'thinking'; text: string }
  | { type: 'content'; text: string }
  | { type: 'tool_call'; text: string };