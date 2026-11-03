import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { MaxIterationsError, ValidationError } from './errors.js';
import { LLMContext } from './LLMContext.js';
import { TaskManager } from './TaskManager.js';
import { defaultTokenizer } from './tokenizer.js';
import { MessageValidator } from './validator.js';
import type {
  AskOptions,
  AskResult,
  ChatMessage,
  ContextMessage,
  ContextStorageStats,
  MessageRole,
  NativeToolCall,
  OllamaChatRequest,
  OllamaChatResponse,
  OllamaClientLike,
  SerializedContext,
  Task,
  ToolDefinition,
  ToolExecutionResult,
  TokenizerFn,
} from './types.js';

// ---------- Tools ----------

const CORE_TOOLS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'getMessageById',
      description:
        'Retrieve the full text of a conversation message by its ID (or a unique prefix). ' +
        'Use this after searchMessages to read a matched message in full.',
      parameters: {
        type: 'object',
        properties: {
          id: {
            type: 'string',
            description: 'Message ID or unique prefix (e.g. first 8 chars)',
          },
        },
        required: ['id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'searchMessages',
      description:
        'Search conversation history with a case-insensitive regular expression. ' +
        'Returns a compact list of matching messages (ID, role, timestamp, truncated text). ' +
        'Use getMessageById to read a specific message in full.',
      parameters: {
        type: 'object',
        properties: {
          pattern: {
            type: 'string',
            description: 'Regex pattern (case-insensitive, max 200 chars)',
          },
          role: {
            type: 'string',
            enum: ['user', 'assistant', 'tool'],
            description: 'Optional filter by role',
          },
          limit: {
            type: 'number',
            description: 'Max results (default 20, max 50)',
          },
        },
        required: ['pattern'],
      },
    },
  },
];

const TASK_TOOLS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'createTask',
      description:
        'Create a new task. When decomposing a larger task into steps, create one task per step ' +
        'and set parentId to the parent task ID.',
      parameters: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Short task title' },
          description: { type: 'string', description: 'Optional detailed description' },
          parentId: {
            type: 'string',
            description: 'Optional parent task ID for sub-task decomposition',
          },
        },
        required: ['title'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'updateTaskStatus',
      description: 'Update the status of an existing task, optionally recording a result.',
      parameters: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'Task ID from CURRENT TASKS STATE' },
          status: {
            type: 'string',
            enum: ['todo', 'in_progress', 'completed', 'failed'],
          },
          result: { type: 'string', description: 'Optional execution result report' },
        },
        required: ['id', 'status'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'deleteTask',
      description: 'Delete an existing task by ID.',
      parameters: {
        type: 'object',
        properties: { id: { type: 'string' } },
        required: ['id'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'listTasks',
      description: 'List all tasks with their current status.',
      parameters: { type: 'object', properties: {} },
    },
  },
];

const DEFAULT_TOOLS: ToolDefinition[] = [...CORE_TOOLS, ...TASK_TOOLS];

const SYSTEM_INSTRUCTION = `[ACTIVE CONTEXT — TOOLS ACTIVE]
You have native tools for tasks and conversation history.

Task tools:     createTask, updateTaskStatus, deleteTask, listTasks
Context tools:  searchMessages, getMessageById

## Core rules

- Never invent task IDs. Only use IDs from CURRENT TASKS STATE below.
- Never add details, technologies, or requirements to a task description
  that the user did not explicitly state. If clarification is needed, ask first.
- CURRENT TASKS STATE in the system block is always up-to-date.
  Answer "what tasks do I have" from it directly, without calling listTasks,
  and without claiming you called listTasks unless you actually did.
- After receiving tool results, provide a concise final answer to the user.

## Decomposition

When the user asks to "decompose", "plan", or "break down" a task:

1. Create the parent task first:
   createTask({ title: "<user's task>" }) → returns parentId
2. Create AT LEAST 3 child tasks, one per phase:
   createTask({ title: "<phase>", parentId: "<parentId>" })
3. Do NOT ask permission — decompose fully in one turn.
4. Do NOT invent technologies or requirements. Use only what the user said,
   or use generic phase names ("Define scope", "Implement", "Test", "Deploy").

Example:
User: "Создай todo-app. Декомпозируй на задачи и подзадачи."
You call:
- createTask({ title: "Todo-app" }) → parent 2b2ecabc
- createTask({ title: "Define scope and requirements", parentId: "2b2ecabc" })
- createTask({ title: "Design data model", parentId: "2b2ecabc" })
- createTask({ title: "Implement backend API", parentId: "2b2ecabc" })
- createTask({ title: "Build UI", parentId: "2b2ecabc" })
- createTask({ title: "Write tests", parentId: "2b2ecabc" })
Then reply with a short summary listing all created IDs.`;

// ---------- Config ----------

export interface ActiveContextConfig {
  modelName: string;
  baseUrl?: string;
  totalTokensLimit?: number;
  options?: Record<string, unknown>;
  contextPersistPath?: string;
  think?: boolean;
  tools?: ToolDefinition[];
}

export interface ActiveContextDeps {
  ollamaClient?: OllamaClientLike;
  taskManager?: TaskManager;
  tokenizer?: TokenizerFn;
  validator?: MessageValidator;
}

const TOOL_RESULT_TRUNCATE = 200;
const SEARCH_PATTERN_MAX = 200;
const SEARCH_LIMIT_DEFAULT = 20;
const SEARCH_LIMIT_MAX = 50;

function truncate(s: string, max: number): string {
  const oneLine = s.replace(/\s+/g, ' ').trim();
  return oneLine.length <= max ? oneLine : oneLine.slice(0, max) + '…';
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toISOString().replace('T', ' ').slice(0, 19);
}

function formatMessageForTool(m: ContextMessage): string {
  const meta = [m.id.slice(0, 8), m.role, formatTimestamp(m.timestamp)];
  if (m.toolCallId) meta.push(`tool_call_id=${m.toolCallId}`);
  return `[${meta.join('|')}] ${m.content}`;
}

// ---------- Class ----------

export class ActiveContext {
  private readonly context: LLMContext;
  private readonly taskManager: TaskManager;
  private readonly tokenizer: TokenizerFn;
  private readonly validator: MessageValidator;
  private readonly config: Required<Pick<ActiveContextConfig, 'modelName' | 'baseUrl'>> &
    ActiveContextConfig & { tools: ToolDefinition[] };
  private think: boolean;

  private ollama: OllamaClientLike | null;
  private readonly contextPersistPath?: string;
  private contextSaveTimer: NodeJS.Timeout | null = null;

  constructor(config: ActiveContextConfig, deps: ActiveContextDeps = {}) {
    const baseUrl = config.baseUrl ?? 'http://127.0.0.1:11434';
    this.config = {
      ...config,
      baseUrl,
      tools: config.tools ?? DEFAULT_TOOLS,
    };
    this.think = config.think ?? true;
    this.tokenizer = deps.tokenizer ?? defaultTokenizer;
    this.validator = deps.validator ?? new MessageValidator();
    this.context = new LLMContext({
      totalTokensLimit: config.totalTokensLimit ?? 8192,
      tokenizer: this.tokenizer,
    });
    this.taskManager = deps.taskManager ?? new TaskManager({ validator: this.validator });
    this.ollama = deps.ollamaClient ?? null;
    this.contextPersistPath = config.contextPersistPath;

    if (this.contextPersistPath) this.loadContext();
  }

  public init(systemPrompt: string): void {
    this.context.setSystemPrompt(systemPrompt);
    this.scheduleContextSave();
  }

  public setThink(value: boolean): void {
    this.think = value;
  }

  public isThinking(): boolean {
    return this.think;
  }

  public get tasks(): TaskManager {
    return this.taskManager;
  }

  public getContext(): LLMContext {
    return this.context;
  }

  public getStatus(): ContextStorageStats {
    return this.context.stats;
  }

  public flush(): void {
    if (this.contextSaveTimer) {
      clearTimeout(this.contextSaveTimer);
      this.contextSaveTimer = null;
    }
    this.saveContext();
    this.taskManager.flush();
  }

  public close(): void {
    this.flush();
  }

  public async ask(userMessage: string, options: AskOptions = {}): Promise<AskResult> {
    if (!options.allowUnsafe) {
      const v = this.validator.validate(userMessage);
      if (!v.valid) {
        throw new ValidationError(`Rejected user message: ${v.reason}`);
      }
    }

    this.context.appendMessage('user', userMessage);
    this.scheduleContextSave();

    const maxIter = options.maxIterations ?? 5;
    const allToolResults: ToolExecutionResult[] = [];
    const allThinking: string[] = [];
    let finalContent = '';
    let iterations = 0;

    while (iterations < maxIter) {
      iterations++;

      this.refreshReservedTokens();
      const messages = this.assemblePipelineMessages();
      const response = await this.callOllama(messages, options);

      const hasToolCalls = response.toolCalls.length > 0;
      if (response.thinking) allThinking.push(response.thinking);

      if (response.content || response.thinking || hasToolCalls) {
        this.context.appendMessage('assistant', response.content, {
          toolCalls: hasToolCalls
            ? response.toolCalls.map((tc) => ({
                id: tc.id,
                function: { name: tc.name, arguments: tc.arguments },
              }))
            : undefined,
        });
        this.scheduleContextSave();
      }

      if (!hasToolCalls) {
        finalContent = response.content;
        break;
      }

      for (const tc of response.toolCalls) {
        const result = this.executeNativeToolCall(tc);
        allToolResults.push(result);
        options.onToolCall?.(result);
        this.context.appendMessage('tool', this.formatToolResultMessage(result), {
          toolCallId: tc.id,
        });
      }
      this.scheduleContextSave();
    }

    if (iterations >= maxIter && !finalContent) {
      throw new MaxIterationsError(maxIter);
    }

    return {
      content: finalContent,
      thinking: allThinking.join('\n').trim(),
      toolCalls: allToolResults,
      iterations,
      stats: this.context.stats,
    };
  }

  // --- Внутренние ---

  private assemblePipelineMessages(): ChatMessage[] {
    const pipeline: ChatMessage[] = [];

    const basePrompt = this.context.getSystemPrompt();
    if (basePrompt) pipeline.push({ role: 'system', content: basePrompt });

    const taskState = this.taskManager.getTaskContextForLLM();
    pipeline.push({
      role: 'system',
      content: `${SYSTEM_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`,
    });

    pipeline.push(...this.context.toPipelineMessages());

    return pipeline;
  }

  private refreshReservedTokens(): void {
    const taskState = this.taskManager.getTaskContextForLLM();
    const combined = `${SYSTEM_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`;
    const toolsTokens = this.tokenizer(JSON.stringify(this.config.tools));
    this.context.setReservedTokens(this.tokenizer(combined) + toolsTokens);
  }

  private async ensureClient(): Promise<OllamaClientLike> {
    if (this.ollama) return this.ollama;
    let mod: typeof import('./OllamaJsClient.js');
    try {
      mod = await import('./OllamaJsClient.js');
    } catch (err) {
      throw new Error(
        'ActiveContext: no ollamaClient provided and the "ollama" package is not installed. ' +
          'Run `npm install ollama` or pass `deps.ollamaClient` explicitly. ' +
          `Original error: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    this.ollama = new mod.OllamaJsClient(this.config.baseUrl);
    return this.ollama;
  }

  private async callOllama(
    messages: ChatMessage[],
    options: AskOptions,
  ): Promise<OllamaChatResponse> {
    const client = await this.ensureClient();
    const shouldStream = !!(options.onThinking || options.onContent);
    const req: OllamaChatRequest = {
      model: this.config.modelName,
      messages,
      options: this.config.options,
      tools: this.config.tools,
      think: this.think,
    };

    if (!shouldStream) {
      return await client.chat(req, { signal: options.signal });
    }

    let content = '';
    let thinking = '';
    const toolCalls: NativeToolCall[] = [];
    const realtime = options.streamMode === 'realtime';

    for await (const chunk of client.chatStream(req, { signal: options.signal })) {
      if (chunk.thinking) {
        thinking += chunk.thinking;
        options.onThinking?.(chunk.thinking);
      }
      if (chunk.content) {
        content += chunk.content;
        if (realtime) options.onContent?.(chunk.content);
      }
      if (chunk.toolCalls.length > 0) toolCalls.push(...chunk.toolCalls);
    }

    if (options.onContent && content && !realtime) options.onContent(content);

    return { content, thinking, toolCalls };
  }

  private executeNativeToolCall(call: NativeToolCall): ToolExecutionResult {
    try {
      return { call, success: true, message: this.dispatchTool(call) };
    } catch (err) {
      return {
        call,
        success: false,
        message: err instanceof Error ? err.message : String(err),
      };
    }
  }

  private dispatchTool(call: NativeToolCall): string {
    const args = call.arguments;

    switch (call.name) {
      // ---------- Context tools ----------

      case 'getMessageById': {
        const id = typeof args.id === 'string' ? args.id : '';
        if (!id) throw new Error('getMessageById: "id" parameter is required');
        const msg = this.context.resolveMessageId(id);
        if (!msg) throw new Error(`Message "${id}" not found`);
        return formatMessageForTool(msg);
      }

      case 'searchMessages': {
        const pattern = typeof args.pattern === 'string' ? args.pattern : '';
        if (!pattern) throw new Error('searchMessages: "pattern" parameter is required');
        if (pattern.length > SEARCH_PATTERN_MAX) {
          throw new Error(
            `searchMessages: pattern too long (${pattern.length} chars, max ${SEARCH_PATTERN_MAX})`,
          );
        }
        const role =
          typeof args.role === 'string' &&
          ['user', 'assistant', 'tool'].includes(args.role)
            ? (args.role as MessageRole)
            : undefined;
        const requested =
          typeof args.limit === 'number' && Number.isFinite(args.limit)
            ? Math.floor(args.limit)
            : SEARCH_LIMIT_DEFAULT;
        const limit = Math.min(Math.max(1, requested), SEARCH_LIMIT_MAX);

        const matches = this.context.findMessages({ pattern, role, limit });
        if (matches.length === 0) return 'No messages matched.';

        const lines = matches.map(
          (m) =>
            `- [${m.id.slice(0, 8)}|${m.role}|${formatTimestamp(m.timestamp)}] ` +
            truncate(m.content, TOOL_RESULT_TRUNCATE),
        );
        return `Found ${matches.length} message(s):\n${lines.join('\n')}`;
      }

      // ---------- Task tools ----------

      case 'createTask': {
        const title = typeof args.title === 'string' ? args.title : '';
        const description =
          typeof args.description === 'string' ? args.description : undefined;
        const parentId = typeof args.parentId === 'string' ? args.parentId : undefined;
        if (!title) throw new Error('createTask: "title" parameter is required');
        const task = this.taskManager.createTask(title, description, parentId);
        const parentInfo = task.parentId ? ` under parent ${task.parentId}` : '';
        return `Task created with ID ${task.id}${parentInfo}`;
      }

      case 'updateTaskStatus': {
        const id = typeof args.id === 'string' ? args.id : '';
        const status = typeof args.status === 'string' ? args.status : '';
        const result = typeof args.result === 'string' ? args.result : undefined;
        if (!id) throw new Error('updateTaskStatus: "id" parameter is required');
        if (!status) throw new Error('updateTaskStatus: "status" parameter is required');
        const updated = this.taskManager.updateTaskStatus(id, status as any, result);
        return `Task ${updated.id} updated to status "${updated.status}"`;
      }

      case 'deleteTask': {
        const id = typeof args.id === 'string' ? args.id : '';
        if (!id) throw new Error('deleteTask: "id" parameter is required');
        const deleted = this.taskManager.deleteTask(id);
        if (!deleted) throw new Error(`Task ${id} not found`);
        return `Task ${id} deleted`;
      }

      case 'listTasks': {
        const all = this.taskManager.getAllTasks();
        if (all.length === 0) return 'No tasks available.';
        const lines = all.map((t) => this.formatTaskLine(t));
        return `Tasks (${all.length}):\n${lines.join('\n')}`;
      }

      default:
        throw new Error(`Unknown function: ${call.name}`);
    }
  }

  private formatTaskLine(t: Task): string {
    const parent = t.parentId ? ` parent=${t.parentId}` : '';
    return `- [${t.id}${parent}] ${t.title} — ${t.status}`;
  }

  private formatToolResultMessage(r: ToolExecutionResult): string {
    const tag = r.success ? 'SUCCESS' : 'ERROR';
    return `[TOOL_RESULT][${tag}][${r.call.name}] ${r.message}`;
  }

  // --- Persistence ---

  private loadContext(): void {
    if (!this.contextPersistPath || !existsSync(this.contextPersistPath)) return;
    try {
      const raw = readFileSync(this.contextPersistPath, 'utf-8');
      const data = JSON.parse(raw) as SerializedContext;
      this.context.import(data);
    } catch (e) {
      console.error(
        `ActiveContext: failed to load context from ${this.contextPersistPath}`,
        e,
      );
    }
  }

  private scheduleContextSave(): void {
    if (!this.contextPersistPath) return;
    if (this.contextSaveTimer) clearTimeout(this.contextSaveTimer);
    this.contextSaveTimer = setTimeout(() => {
      this.contextSaveTimer = null;
      this.saveContext();
    }, 100);
  }

  private saveContext(): void {
    if (!this.contextPersistPath) return;
    try {
      mkdirSync(dirname(this.contextPersistPath), { recursive: true });
      const tmpPath = `${this.contextPersistPath}.tmp`;
      writeFileSync(tmpPath, JSON.stringify(this.context.export(), null, 2));
      renameSync(tmpPath, this.contextPersistPath);
    } catch (e) {
      console.error(
        `ActiveContext: failed to persist context to ${this.contextPersistPath}`,
        e,
      );
    }
  }
}