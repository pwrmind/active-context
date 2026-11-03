import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { MaxIterationsError, ValidationError } from './errors.js';
import { LLMContext } from './LLMContext.js';
import { TaskManager } from './TaskManager.js';
import { defaultTokenizer } from './tokenizer.js';
import { MessageValidator } from './validator.js';
import type { OllamaClientLike } from './types.js';
import type {
  AskOptions,
  AskResult,
  ChatMessage,
  ContextStorageStats,
  NativeToolCall,
  OllamaChatRequest,
  OllamaChatResponse,
  SerializedContext,
  StreamChunk,
  ToolDefinition,
  ToolExecutionResult,
  TokenizerFn,
} from './types.js';

const TASK_TOOLS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'createTask',
      description: 'Create a new task with a title and optional description.',
      parameters: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Short task title' },
          description: { type: 'string', description: 'Optional detailed description' },
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
        properties: {
          id: { type: 'string' },
        },
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

const TASK_API_INSTRUCTION = `[TASK MANAGER MODULE ACTIVE]
You have access to a Task Manager via native function calls (tools). Use the provided tools for all task operations.

Rules:
- Never invent task IDs. Only use IDs from CURRENT TASKS STATE below.
- After receiving tool results, provide a concise final answer to the user.`;

export interface ActiveContextConfig {
  modelName: string;
  baseUrl?: string;
  totalTokensLimit?: number;
  options?: Record<string, unknown>;
  contextPersistPath?: string;
  /** Enable native thinking. Default true (LFM3). Set false for models without thinking. */
  think?: boolean;
  /** Override tool definitions. Defaults to TASK_TOOLS. */
  tools?: ToolDefinition[];
}

export interface ActiveContextDeps {
  ollamaClient?: OllamaClientLike;
  taskManager?: TaskManager;
  tokenizer?: TokenizerFn;
  validator?: MessageValidator;
}

export class ActiveContext {
  private readonly context: LLMContext;
  private readonly taskManager: TaskManager;
  private readonly tokenizer: TokenizerFn;
  private readonly validator: MessageValidator;
  private readonly config: Required<Pick<ActiveContextConfig, 'modelName' | 'baseUrl'>> &
    ActiveContextConfig & { think: boolean; tools: ToolDefinition[] };

  private ollama: OllamaClientLike | null;
  private readonly contextPersistPath?: string;
  private contextSaveTimer: NodeJS.Timeout | null = null;

  constructor(config: ActiveContextConfig, deps: ActiveContextDeps = {}) {
    const baseUrl = config.baseUrl ?? 'http://127.0.0.1:11434';
    this.config = {
      ...config,
      baseUrl,
      think: config.think ?? true,
      tools: config.tools ?? TASK_TOOLS,
    };
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
      content: `${TASK_API_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`,
    });

    pipeline.push(...this.context.toPipelineMessages());

    return pipeline;
  }

  private refreshReservedTokens(): void {
    const taskState = this.taskManager.getTaskContextForLLM();
    const combined = `${TASK_API_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`;
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
      think: this.config.think,
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
      if (chunk.toolCalls.length > 0) {
        toolCalls.push(...chunk.toolCalls);
      }
    }

    if (options.onContent && content && !realtime) {
      options.onContent(content);
    }

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
      case 'createTask': {
        const title = typeof args.title === 'string' ? args.title : '';
        const description =
          typeof args.description === 'string' ? args.description : undefined;
        if (!title) throw new Error('createTask: "title" parameter is required');
        const task = this.taskManager.createTask(title, description);
        return `Task created with ID ${task.id}`;
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
        return all.length === 0
          ? 'No tasks available.'
          : all.map((t) => `[${t.id}] ${t.title} — ${t.status}`).join('\n');
      }
      default:
        throw new Error(`Unknown function: ${call.name}`);
    }
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