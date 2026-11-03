import * as readline from 'node:readline';
import { ActiveContext } from '../src/ActiveContext.js';
import type { AskOptions, ToolExecutionResult } from '../src/types.js';

// ---------- ANSI ----------
// Работает в Windows Terminal, macOS, Linux. В старом cmd.exe — просто
// отключится по isTTY === false.
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code: number, text: string): string =>
  useColor ? `\x1b[${code}m${text}\x1b[0m` : text;
const gray = (t: string) => c(90, t);
const red = (t: string) => c(31, t);
const green = (t: string) => c(32, t);
const yellow = (t: string) => c(33, t);
const cyan = (t: string) => c(36, t);
const bold = (t: string) => c(1, t);

// ---------- Config ----------

interface CliOptions {
  model: string;
  host: string;
  totalTokensLimit: number;
  numCtx: number;
  persistPath?: string;
  systemPrompt?: string;
}

const DEFAULT_SYSTEM_PROMPT =
  'Ты — автономный ИИ-ассистент. Управляй задачами через предоставленные функции. ' +
  'Отвечай кратко и по делу на языке пользователя.';

const HELP = `
${bold('Команды:')}
  /help, /h              эта справка
  /tasks, /t             список задач
  /new <title> [| <desc>]  создать задачу вручную
  /forget                удалить все задачи
  /clear, /c             очистить историю диалога (system prompt и задачи сохраняются)
  /status, /s            состояние контекста
  /save                  сбросить контекст и задачи на диск (если включён --persist)
  /exit, /quit, /q       выход

${bold('Горячие клавиши:')}
  Ctrl+C                 прервать текущий запрос; если запрос не выполняется — выйти
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
      think: true,
      options: { num_ctx: opts.numCtx, temperature: 0.2 },
    });
    this.agent.init(opts.systemPrompt ?? DEFAULT_SYSTEM_PROMPT);

    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: process.stdin.isTTY === true,
    });

    this.rl.on('line', (line) => {
      const resolve = this.pendingLine;
      this.pendingLine = null;
      resolve?.(line);
    });

    this.rl.on('close', () => {
      const resolve = this.pendingLine;
      this.pendingLine = null;
      resolve?.(null);
      this.running = false;
    });

    this.rl.on('SIGINT', () => {
      if (this.currentAbort) {
        process.stdout.write('\n' + yellow('⏹  Запрос прерван') + '\n');
        this.currentAbort.abort();
        return;
      }
      // Никакого ask не выполняется — выходим
      process.stdout.write('\n');
      this.rl.close();
    });
  }

  public async run(): Promise<void> {
    console.log(bold('🚀 ActiveContext interactive client'));
    console.log(`Модель:   ${cyan(this.opts.model)}`);
    console.log(`Ollama:   ${this.opts.host}`);
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
        if (trimmed.startsWith('/')) {
          await this.handleCommand(trimmed);
        } else {
          await this.handleUserMessage(trimmed);
        }
      } catch (err) {
        console.error(red(`❌ ${err instanceof Error ? err.message : String(err)}`));
      }
    }

    this.shutdown();
  }

  // ---------- Input ----------

  private readInput(): Promise<string | null> {
    if (!this.running) return Promise.resolve(null);
    return new Promise((resolve) => {
      this.pendingLine = resolve;
      this.rl.setPrompt(bold('👤 '));
      this.rl.prompt();
    });
  }

  // ---------- Commands ----------

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
          console.log(yellow('Использование: /new <title> [| <description>]'));
          return;
        }
        const [title, description] = arg.split('|').map((s) => s.trim());
        const task = this.agent.tasks.createTask(title, description);
        console.log(green(`✅ Задача создана: [${task.id}] ${task.title}`));
        return;
      }

      case '/forget': {
        const count = this.agent.tasks.getAllTasks().length;
        this.agent.tasks.clear();
        console.log(green(`🧹 Удалено задач: ${count}`));
        return;
      }

      case '/clear':
      case '/c':
        this.agent.getContext().clear();
        console.log(green('🧹 История диалога очищена (system prompt и задачи сохранены)'));
        return;

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

  // ---------- User message ----------

  private async handleUserMessage(text: string): Promise<void> {
    const abort = new AbortController();
    this.currentAbort = abort;

    let lastWasThinking = false;
    const closeThinking = (): void => {
      if (lastWasThinking) {
        process.stdout.write('\n');
        lastWasThinking = false;
      }
    };

    const onThinking = (chunk: string): void => {
      if (!lastWasThinking) process.stdout.write('🧠 ');
      process.stdout.write(gray(chunk));
      lastWasThinking = true;
    };
    const onContent = (chunk: string): void => {
      if (lastWasThinking) {
        process.stdout.write('\n\n');
        lastWasThinking = false;
      }
      process.stdout.write(chunk);
    };
    const onToolCall = (r: ToolExecutionResult): void => {
      closeThinking();
      const icon = r.success ? green('✅') : red('❌');
      console.log(`${icon} 🛠  ${bold(r.call.name)} → ${r.message}`);
    };

    const askOpts: AskOptions = {
      onThinking,
      onContent,
      onToolCall,
      streamMode: 'realtime',
      signal: abort.signal,
    };

    const startedAt = Date.now();
    try {
      const result = await this.agent.ask(text, askOpts);
      closeThinking();
      const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
      console.log(
        '\n' +
          gray(
            `📊 ${result.iterations} итер. · ${result.toolCalls.length} tool calls · ` +
              `${result.stats.currentTokensUsed}/${result.stats.totalTokensLimit} ток. · ${elapsed}s`,
          ),
      );
      console.log();
    } catch (err) {
      closeThinking();
      if (!abort.signal.aborted) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(red(`❌ ${msg}`));
      }
      // Прерывание уже отражено SIGINT-обработчиком
    } finally {
      this.currentAbort = null;
    }
  }

  // ---------- Inspect ----------

  private printTasks(): void {
    const tasks = this.agent.tasks.getAllTasks();
    if (tasks.length === 0) {
      console.log(gray('Задач нет.'));
      return;
    }
    console.log(bold(`📋 Задачи (${tasks.length}):`));
    for (const t of tasks) {
      const statusColor =
        t.status === 'completed'
          ? green
          : t.status === 'failed'
          ? red
          : t.status === 'in_progress'
          ? cyan
          : yellow;
      const desc = t.description ? gray(` — ${t.description}`) : '';
      const result = t.result ? gray(` [result: ${t.result}]`) : '';
      console.log(
        `   [${t.id}] ${t.title} ${statusColor(`[${t.status}]`)}${desc}${result}`,
      );
    }
  }

  private printStatus(): void {
    const s = this.agent.getStatus();
    console.log(bold('📈 Контекст:'));
    console.log(`   Всего:           ${s.totalTokensLimit}`);
    console.log(`   System prompt:   ${s.systemPromptTokens}`);
    console.log(`   Зарезервировано: ${s.reservedTokens}  (TaskManager + tools schema)`);
    console.log(`   История:         ${s.historyTokens}`);
    console.log(`   Использовано:    ${s.currentTokensUsed}`);
    console.log(`   Свободно:        ${s.availableTokens}`);
    console.log(`   Задач:           ${this.agent.tasks.getAllTasks().length}`);
  }

  // ---------- Shutdown ----------

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
    persistPath: process.env.AC_PERSIST,
    systemPrompt: process.env.AC_SYSTEM_PROMPT,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--model':
        opts.model = argv[++i];
        break;
      case '--host':
        opts.host = argv[++i];
        break;
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
      case '--persist':
        opts.persistPath = argv[++i];
        break;
      case '--system':
        opts.systemPrompt = argv[++i];
        break;
      case '-h':
      case '--help':
        console.log(
          'Использование: ac-cli [--model lfm3:latest] [--host URL] [--tokens N] [--persist PATH] [--system PROMPT]',
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
  const repl = new Repl(opts);
  await repl.run();
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});