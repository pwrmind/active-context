export type MessageRole = 'user' | 'assistant' | 'tool';
export type TaskStatus = 'todo' | 'in_progress' | 'completed' | 'failed';

export interface ContextMessage {
  readonly id: string;
  readonly role: MessageRole;
  content: string;
  readonly timestamp: number;
  tokensCount: number;
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

/**
 * Снимок состояния LLMContext, пригодный для сохранения на диск
 * и восстановления через LLMContext.import().
 */
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
  readonly createdAt: number;
  updatedAt: number;
}

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

export interface ToolExecutionResult {
  call: ParsedToolCall;
  success: boolean;
  message: string;
}

export interface OllamaChatRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  options?: Record<string, unknown>;
}

export interface OllamaChatResponse {
  content: string;
}

export type TokenizerFn = (text: string) => number;

export type StreamEvent =
  | { type: 'thinking'; text: string }
  | { type: 'content'; text: string }
  | { type: 'tool_call'; text: string };

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