import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { MaxIterationsError, ValidationError } from './errors.js';
import { LLMContext } from './LLMContext.js';
import { LFM3Parser } from './parser.js';
import { StreamRouter } from './stream.js';
import { TaskManager } from './TaskManager.js';
import { defaultTokenizer } from './tokenizer.js';
import { MessageValidator } from './validator.js';
import type { OllamaClientLike } from './OllamaClient.js';
import type {
  AskOptions,
  AskResult,
  ContextStorageStats,
  ParsedToolCall,
  SerializedContext,
  StreamEvent,
  ToolExecutionResult,
  TokenizerFn,
} from './types.js';

const TASK_API_INSTRUCTION = `[TASK MANAGER MODULE ACTIVE]
You have access to a built-in Task Manager. You MUST use the following XML format to interact with tasks.

Format for reasoning (optional, hidden from user):
<think>your reasoning here</think>

Format for tool calls (required for actions):
<tool_call><function=FUNCTION_NAME><parameter=PARAM>value</parameter></function></tool_call>

Available functions:
1. createTask
   Parameters: title (required), description (optional)
2. updateTaskStatus
   Parameters: id (required), status (required: todo | in_progress | completed | failed), result (optional)
3. deleteTask
   Parameters: id (required)
4. listTasks
   No parameters.

Rules:
- Never hallucinate task IDs. Only use IDs from CURRENT TASKS STATE.
- Always place your reasoning inside <think> tags before tool calls.
- After receiving tool results, provide a concise final answer to the user.`;

export interface ActiveContextConfig {
  modelName: string;
  baseUrl?: string;
  totalTokensLimit?: number;
  options?: Record<string, unknown>;
  /** Если задан — история диалога и system prompt сохраняются в этот JSON-файл. */
  contextPersistPath?: string;
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
    ActiveContextConfig;

  private ollama: OllamaClientLike | null;
  private readonly contextPersistPath?: string;
  private contextSaveTimer: NodeJS.Timeout | null = null;

  constructor(config: ActiveContextConfig, deps: ActiveContextDeps = {}) {
    const baseUrl = config.baseUrl ?? 'http://127.0.0.1:11434';
    this.config = { ...config, baseUrl };
    this.tokenizer = deps.tokenizer ?? defaultTokenizer;
    this.validator = deps.validator ?? new MessageValidator();
    this.context = new LLMContext({
      totalTokensLimit: config.totalTokensLimit ?? 8192,
      tokenizer: this.tokenizer,
    });
    this.taskManager = deps.taskManager ?? new TaskManager({ validator: this.validator });
    this.ollama = deps.ollamaClient ?? null;
    this.contextPersistPath = config.contextPersistPath;

    if (this.contextPersistPath) {
      this.loadContext();
    }
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

  /**
   * Принудительно сбрасывает отложенные сохранения контекста и задач на диск.
   * Полезно в тестах и перед `process.exit()`.
   */
  public flush(): void {
    if (this.contextSaveTimer) {
      clearTimeout(this.contextSaveTimer);
      this.contextSaveTimer = null;
    }
    this.saveContext();
    this.taskManager.flush();
  }

  /**
   * Семантический алиас для graceful shutdown.
   */
  public close(): void {
    this.flush();
  }

  /**
   * Главный цикл агента: запрос → tool calls → повторный запрос → финальный ответ.
   */
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
      const { raw, bufferedContent } = await this.callOllama(messages, options);

      const parsed = LFM3Parser.parseAll(raw);
      if (parsed.thinking) allThinking.push(parsed.thinking);

      this.context.appendMessage('assistant', raw);
      this.scheduleContextSave();

      if (parsed.calls.length === 0) {
        finalContent = parsed.cleaned;
        if (options.onContent && bufferedContent && options.streamMode !== 'realtime') {
          options.onContent(bufferedContent);
        }
        break;
      }

      const results = await this.executeToolCalls(parsed.calls, options);
      allToolResults.push(...results);

      for (const r of results) {
        this.context.appendMessage('tool', this.formatToolResultMessage(r));
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

  private assemblePipelineMessages(): Array<{ role: string; content: string }> {
    const pipeline: Array<{ role: string; content: string }> = [];

    const basePrompt = this.context.getSystemPrompt();
    if (basePrompt) pipeline.push({ role: 'system', content: basePrompt });

    const taskState = this.taskManager.getTaskContextForLLM();
    pipeline.push({
      role: 'system',
      content: `${TASK_API_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`,
    });

    pipeline.push(...this.context.toHistoryOnly());

    return pipeline;
  }

  private refreshReservedTokens(): void {
    const taskState = this.taskManager.getTaskContextForLLM();
    const combined = `${TASK_API_INSTRUCTION}\n\nCURRENT TASKS STATE:\n${taskState}`;
    this.context.setReservedTokens(this.tokenizer(combined));
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
    messages: Array<{ role: string; content: string }>,
    options: AskOptions,
  ): Promise<{ raw: string; bufferedContent: string }> {
    const client = await this.ensureClient();
    const shouldStream = !!(options.onThinking || options.onContent);
    const req = {
      model: this.config.modelName,
      messages,
      options: this.config.options,
    };

    if (!shouldStream) {
      const res = await client.chat(req, { signal: options.signal });
      return { raw: res.content, bufferedContent: '' };
    }

    const router = new StreamRouter();
    let raw = '';
    let bufferedContent = '';
    const realtime = options.streamMode === 'realtime';

    const handle = (ev: StreamEvent): void => {
      if (ev.type === 'thinking') {
        options.onThinking?.(ev.text);
      } else if (ev.type === 'content') {
        bufferedContent += ev.text;
        if (realtime) options.onContent?.(ev.text);
      }
    };

    for await (const chunk of client.chatStream(req, { signal: options.signal })) {
      raw += chunk;
      for (const ev of router.push(chunk)) handle(ev);
    }
    for (const ev of router.flush()) handle(ev);

    return { raw, bufferedContent };
  }

  private async executeToolCalls(
    calls: ParsedToolCall[],
    options: AskOptions,
  ): Promise<ToolExecutionResult[]> {
    const results: ToolExecutionResult[] = [];

    for (const call of calls) {
      let result: ToolExecutionResult;
      try {
        result = { call, success: true, message: this.dispatchTool(call) };
      } catch (err) {
        result = {
          call,
          success: false,
          message: err instanceof Error ? err.message : String(err),
        };
      }
      results.push(result);
      options.onToolCall?.(result);
    }

    return results;
  }

  private dispatchTool(call: ParsedToolCall): string {
    switch (call.functionName) {
      case 'createTask': {
        const title = call.args.title;
        if (!title) throw new Error('createTask: "title" parameter is required');
        const task = this.taskManager.createTask(title, call.args.description);
        return `Task created with ID ${task.id}`;
      }
      case 'updateTaskStatus': {
        const { id, status, result } = call.args;
        if (!id) throw new Error('updateTaskStatus: "id" parameter is required');
        if (!status) throw new Error('updateTaskStatus: "status" parameter is required');
        const updated = this.taskManager.updateTaskStatus(id, status as any, result);
        return `Task ${updated.id} updated to status "${updated.status}"`;
      }
      case 'deleteTask': {
        const { id } = call.args;
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
        throw new Error(`Unknown function: ${call.functionName}`);
    }
  }

  private formatToolResultMessage(r: ToolExecutionResult): string {
    const tag = r.success ? 'SUCCESS' : 'ERROR';
    return `[TOOL_RESULT][${tag}][${r.call.functionName}] ${r.message}`;
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