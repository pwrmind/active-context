import * as readline from 'node:readline';
import { ActiveContext } from '../src/ActiveContext.js';
import type { AskOptions, ToolExecutionResult } from '../src/types.js';

// ---------- ANSI ----------
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code: number, text: string): string =>
  useColor ? `\x1b[${code}m${text}\x1b[0m` : text;
const gray = (t: string) => c(90, t);
const red = (t: string) => c(31, t);
const green = (t: string) => c(32, t);
const yellow = (t: string) => c(33, t);
const cyan = (t: string) => c(36, t);
const bold = (t: string) => c(1, t);

// ---------- Thinking printers ----------

class CompactThinkingPrinter {
  private buf = '';
  private printed = 0;
  private readonly max = 80;
  private started = false;
  private finished = false;

  push(chunk: string): void {
    if (this.finished) return;
    this.buf += chunk;
    if (!this.started) {
      process.stdout.write('🤔 ');
      this.started = true;
    }
    if (this.printed < this.max) {
      const piece = this.buf.slice(this.printed, Math.min(this.buf.length, this.max));
      if (piece) {
        process.stdout.write(gray(piece));
        this.printed += piece.length;
      }
      if (this.printed >= this.max && this.buf.length > this.max) {
        process.stdout.write(gray('…'));
        this.printed = this.max + 1;
      }
    }
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    if (this.started) process.stdout.write('\n');
  }

  summary(durationSec: number, thinkingChars: number): string {
    return gray(
      `   (думал ${durationSec.toFixed(1)}s, ~${Math.ceil(thinkingChars / 4)} ток thinking)`,
    );
  }
}

class VerboseThinkingPrinter {
  private open = false;
  private finished = false;

  push(chunk: string): void {
    if (this.finished) return;
    if (!this.open) {
      process.stdout.write('🧠 ');
      this.open = true;
    }
    process.stdout.write(gray(chunk));
  }

  finish(): void {
    if (this.finished) return;
    this.finished = true;
    if (this.open) process.stdout.write('\n');
  }

  summary(): string {
    return '';
  }
}

// ---------- Config ----------

interface CliOptions {
  model: string;
  host: string;
  totalTokensLimit: number;
  numCtx: number;
  think: boolean;
  verboseThinking: boolean;
  persistPath?: string;
  systemPrompt?: string;
}

const DEFAULT_SYSTEM_PROMPT =
  'Ты — автономный ИИ-ассистент. Управляй задачами и ищи в истории диалога ' +
  'через предоставленные функции. Отвечай кратко и по делу на языке пользователя.';

const HELP = `
${bold('Команды:')}
  /help, /h                эта справка
  /tasks, /t               список задач
  /new <title> [| <desc>] [@<parentId>]  создать задачу вручную
  /forget                  удалить все задачи
  /clear, /c               очистить историю диалога
  /find <regex>            поиск по сообщениям (локальный, без модели)
  /msg <id>                показать сообщение по ID
  /think [on|off]          переключить thinking
  /status, /s              состояние контекста
  /save                    сбросить контекст и задачи на диск
  /exit, /quit, /q         выход

${bold('Горячие клавиши:')}
  Ctrl+C                   прервать текущий запрос; если запрос не выполняется — выйти
`;

// ---------- Repl ----------

class Repl {
  private readonly rl: readline.Interface;
  private readonly agent: ActiveContext;
  private pendingLine: ((line: string | null) => void) | null = null;
  private currentAbort: AbortController | null = null;
  private running = true;

  constructor(private readonly opts: CliOptions) {
    this.agent = new ActiveContext({
      modelName: opts.model,
      baseUrl: opts.host,
      totalTokensLimit: opts.totalTokensLimit,
      contextPersistPath: opts.persistPath,
      think: opts.think,
      options: { num_ctx: opts.numCtx, temperature: 0.2 },
    });
    this.agent.init(opts.systemPrompt ?? DEFAULT_SYSTEM_PROMPT);

    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: process.stdin.isTTY === true,
    });

    this.rl.on('line', (line) => {
      const r = this.pendingLine;
      this.pendingLine = null;
      r?.(line);
    });
    this.rl.on('close', () => {
      const r = this.pendingLine;
      this.pendingLine = null;
      r?.(null);
      this.running = false;
    });
    this.rl.on('SIGINT', () => {
      if (this.currentAbort) {
        process.stdout.write('\n' + yellow('⏹  Запрос прерван') + '\n');
        this.currentAbort.abort();
        return;
      }
      process.stdout.write('\n');
      this.rl.close();
    });
  }

  public async run(): Promise<void> {
    console.log(bold('🚀 ActiveContext interactive client'));
    console.log(`Модель:   ${cyan(this.opts.model)}`);
    console.log(`Ollama:   ${this.opts.host}`);
    console.log(`Thinking: ${this.agent.isThinking() ? green('on') : gray('off')}`);
    if (this.opts.persistPath) console.log(`Сессия:   ${this.opts.persistPath}`);
    const s = this.agent.getStatus();
    console.log(
      `Контекст: ${s.currentTokensUsed}/${s.totalTokensLimit} токенов, ` +
        `задач: ${this.agent.tasks.getAllTasks().length}`,
    );
    console.log('\nВведите задачу или /help для команд.\n');

    while (this.running) {
      const line = await this.readInput();
      if (line === null) break;
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        if (trimmed.startsWith('/')) await this.handleCommand(trimmed);
        else await this.handleUserMessage(trimmed);
      } catch (err) {
        console.error(red(`❌ ${err instanceof Error ? err.message : String(err)}`));
      }
    }
    this.shutdown();
  }

  private readInput(): Promise<string | null> {
    if (!this.running) return Promise.resolve(null);
    return new Promise((resolve) => {
      this.pendingLine = resolve;
      this.rl.setPrompt(bold('👤 '));
      this.rl.prompt();
    });
  }

  private async handleCommand(raw: string): Promise<void> {
    const spaceIdx = raw.indexOf(' ');
    const cmd = spaceIdx === -1 ? raw : raw.slice(0, spaceIdx);
    const arg = spaceIdx === -1 ? '' : raw.slice(spaceIdx + 1).trim();

    switch (cmd) {
      case '/help':
      case '/h':
        console.log(HELP);
        return;
      case '/exit':
      case '/quit':
      case '/q':
        this.running = false;
        this.rl.close();
        return;
      case '/tasks':
      case '/t':
        this.printTasks();
        return;
      case '/new': {
        if (!arg) {
          console.log(yellow('Использование: /new <title> [| <desc>] [@<parentId>]'));
          return;
        }
        let parentId: string | undefined;
        let body = arg;
        const atIdx = body.lastIndexOf('@');
        if (atIdx !== -1) {
          const candidate = body.slice(atIdx + 1).trim();
          if (candidate) {
            parentId = candidate;
            body = body.slice(0, atIdx).trim();
          }
        }
        const [title, description] = body.split('|').map((s) => s.trim());
        const task = this.agent.tasks.createTask(title, description, parentId);
        console.log(
          green(`✅ Задача создана: [${task.id}] ${task.title}`) +
            (task.parentId ? gray(` (parent: ${task.parentId})`) : ''),
        );
        return;
      }
      case '/forget': {
        const n = this.agent.tasks.getAllTasks().length;
        this.agent.tasks.clear();
        console.log(green(`🧹 Удалено задач: ${n}`));
        return;
      }
      case '/clear':
      case '/c':
        this.agent.getContext().clear();
        console.log(green('🧹 История диалога очищена'));
        return;
      case '/find': {
        if (!arg) {
          console.log(yellow('Использование: /find <regex>'));
          return;
        }
        try {
          const matches = this.agent.getContext().findMessages({ pattern: arg, limit: 20 });
          if (matches.length === 0) {
            console.log(gray('Ничего не найдено.'));
            return;
          }
          console.log(bold(`🔍 Найдено ${matches.length}:`));
          for (const m of matches) {
            const text = m.content.replace(/\s+/g, ' ').slice(0, 100);
            console.log(`   [${m.id.slice(0, 8)}|${m.role}] ${text}`);
          }
        } catch (e) {
          console.log(red(`Ошибка regex: ${(e as Error).message}`));
        }
        return;
      }
      case '/msg': {
        if (!arg) {
          console.log(yellow('Использование: /msg <id>'));
          return;
        }
        const msg = this.agent.getContext().resolveMessageId(arg);
        if (!msg) {
          console.log(red(`Сообщение "${arg}" не найдено`));
          return;
        }
        console.log(bold(`[${msg.id}] ${msg.role} · ${new Date(msg.timestamp).toISOString()}`));
        console.log(msg.content);
        return;
      }
      case '/think': {
        if (arg === '' || arg === 'on') {
          this.agent.setThink(true);
          console.log(green('thinking: on'));
        } else if (arg === 'off') {
          this.agent.setThink(false);
          console.log(green('thinking: off'));
        } else {
          console.log(yellow('Использование: /think [on|off]'));
        }
        return;
      }
      case '/status':
      case '/s':
        this.printStatus();
        return;
      case '/save':
        this.agent.flush();
        console.log(green('💾 Контекст и задачи сброшены на диск'));
        return;
      default:
        console.log(yellow(`Неизвестная команда: ${cmd}. Введите /help.`));
    }
  }

  private async handleUserMessage(text: string): Promise<void> {
    const abort = new AbortController();
    this.currentAbort = abort;

    const printer = this.opts.verboseThinking
      ? new VerboseThinkingPrinter()
      : new CompactThinkingPrinter();

    let thinkingChars = 0;
    let contentOpen = false;
    const startedAt = Date.now();

    const onThinking = (chunk: string): void => {
      thinkingChars += chunk.length;
      printer.push(chunk);
    };
    const onContent = (chunk: string): void => {
      printer.finish();
      if (!contentOpen) {
        process.stdout.write('\n');
        contentOpen = true;
      }
      process.stdout.write(chunk);
    };
    const onToolCall = (r: ToolExecutionResult): void => {
      printer.finish();
      if (contentOpen) {
        process.stdout.write('\n');
        contentOpen = false;
      }
      const icon = r.success ? green('✅') : red('❌');
      console.log(`${icon} 🛠  ${bold(r.call.name)} → ${r.message.split('\n')[0]}`);
    };

    const askOpts: AskOptions = {
      onThinking,
      onContent,
      onToolCall,
      streamMode: 'realtime',
      signal: abort.signal,
    };

    try {
      const result = await this.agent.ask(text, askOpts);
      printer.finish();
      if (contentOpen) process.stdout.write('\n');
      const elapsed = (Date.now() - startedAt) / 1000;
      const summary = printer.summary(elapsed, thinkingChars);
      if (summary) console.log(summary);
      console.log(
        gray(
          `📊 ${result.iterations} итер. · ${result.toolCalls.length} tool calls · ` +
            `${result.stats.currentTokensUsed}/${result.stats.totalTokensLimit} ток. · ` +
            `${elapsed.toFixed(1)}s`,
        ),
      );
      console.log();
    } catch (err) {
      printer.finish();
      if (!abort.signal.aborted) {
        console.error(red(`❌ ${err instanceof Error ? err.message : String(err)}`));
      }
    } finally {
      this.currentAbort = null;
    }
  }

  private printTasks(): void {
    const tasks = this.agent.tasks.getAllTasks();
    if (tasks.length === 0) {
      console.log(gray('Задач нет.'));
      return;
    }
    console.log(bold(`📋 Задачи (${tasks.length}):`));
    for (const t of tasks) {
      const color =
        t.status === 'completed' ? green
        : t.status === 'failed' ? red
        : t.status === 'in_progress' ? cyan
        : yellow;
      const parent = t.parentId ? gray(` (parent: ${t.parentId})`) : '';
      const desc = t.description ? gray(` — ${t.description}`) : '';
      const result = t.result ? gray(` [result: ${t.result}]`) : '';
      console.log(`   [${t.id}] ${t.title} ${color(`[${t.status}]`)}${parent}${desc}${result}`);
    }
  }

  private printStatus(): void {
    const s = this.agent.getStatus();
    console.log(bold('📈 Контекст:'));
    console.log(`   Всего:           ${s.totalTokensLimit}`);
    console.log(`   System prompt:   ${s.systemPromptTokens}`);
    console.log(`   Зарезервировано: ${s.reservedTokens}`);
    console.log(`   История:         ${s.historyTokens}`);
    console.log(`   Использовано:    ${s.currentTokensUsed}`);
    console.log(`   Свободно:        ${s.availableTokens}`);
    console.log(`   Задач:           ${this.agent.tasks.getAllTasks().length}`);
    console.log(`   Thinking:        ${this.agent.isThinking() ? green('on') : gray('off')}`);
  }

  private shutdown(): void {
    try {
      this.agent.close();
    } catch (e) {
      console.error(red(`Ошибка при сохранении: ${e}`));
    }
    console.log(bold('👋 До свидания'));
  }
}

// ---------- Entry ----------

function parseArgs(argv: string[]): CliOptions | null {
  const opts: CliOptions = {
    model: process.env.OLLAMA_MODEL ?? 'lfm3:latest',
    host: process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434',
    totalTokensLimit: 8192,
    numCtx: 8192,
    think: true,
    verboseThinking: false,
    persistPath: process.env.AC_PERSIST,
    systemPrompt: process.env.AC_SYSTEM_PROMPT,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--model': opts.model = argv[++i]; break;
      case '--host': opts.host = argv[++i]; break;
      case '--tokens': {
        const n = parseInt(argv[++i], 10);
        if (!Number.isFinite(n) || n < 256) {
          console.error(red('--tokens: ожидается число >= 256'));
          return null;
        }
        opts.totalTokensLimit = n;
        opts.numCtx = n;
        break;
      }
      case '--persist': opts.persistPath = argv[++i]; break;
      case '--system': opts.systemPrompt = argv[++i]; break;
      case '--think': opts.think = true; break;
      case '--no-think': opts.think = false; break;
      case '--verbose-thinking': opts.verboseThinking = true; break;
      case '-h':
      case '--help':
        console.log(
          'Использование: ac-cli [--model X] [--host URL] [--tokens N] ' +
            '[--persist PATH] [--system PROMPT] [--no-think] [--verbose-thinking]',
        );
        return null;
      default:
        if (arg.startsWith('--')) {
          console.error(red(`Неизвестный флаг: ${arg}`));
          return null;
        }
    }
  }
  return opts;
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  if (!opts) return;
  await new Repl(opts).run();
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});