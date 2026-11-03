# ⚡ ActiveContext

**Локальный AI-агент с памятью, задачами и нативным tool calling поверх Ollama.**

`ActiveContext` — это TypeScript-библиотека для создания автономных агентов на базе локальных LLM (LFM3, Qwen3, Llama 3.1 и других моделей, поддерживающих native tool calling). Контекст здесь не пассивный массив строк, а активный компонент: он сам управляет бюджетом токенов, вытесняет устаревшую историю, защищает system prompt и предоставляет модели инструменты для работы с задачами и собственной памятью.

```ts
import { ActiveContext } from 'active-context';

const agent = new ActiveContext({ modelName: 'lfm3:latest' });
agent.init('Ты — автономный ИИ-ассистент. Управляй задачами через инструменты.');

const result = await agent.ask('Создай задачу: подготовить отчёт по проекту');
// → агент сам вызывает createTask, получает ID, отвечает пользователю
```

---

## Возможности

- 🧠 **Управляемый контекст** — скользящее окно токенов с FIFO-вытеснением и изолированным system prompt.
- 🛠 **Нативный tool calling** — JSON Schema для функций, совместимая с Ollama `/api/chat`.
- 📋 **Встроенный TaskManager** — CRUD задач, parent/child иерархия, персистентность в JSON.
- 🔍 **Поиск по истории** — `searchMessages` (regex) и `getMessageById` доступны модели как инструменты.
- 🤔 **Thinking-стриминг** — нативные `thinking`-чанки LFM3 с переключением через `/think on|off`.
- 💾 **Persistence** — контекст и задачи переживают перезапуск, атомарная запись на диск.
- 🛡 **Защита от prompt injection** — паттерн-матчинг + санитизация полей задач.
- 🔄 **Agent loop с recovery** — модель видит ошибки инструментов и восстанавливается, лимит итераций против зацикливания.
- 🧪 **54 unit-теста** — покрытие всех слоёв, включая persistence, regex, tool calling и agent loop.

---

## Требования

- **Node.js** ≥ 20
- **Ollama** с моделью, поддерживающей native tool calling (`lfm3`, `qwen3`, `llama3.1`)

```bash
# Установить Ollama: https://ollama.com/download
ollama serve
ollama pull lfm3:latest
```

---

## Установка

```bash
npm install active-context ollama
```

`ollama` — peer-зависимость, загружается лениво. Если вы передаёте свой `ollamaClient` через `deps`, `ollama` не обязателен.

---

## Быстрый старт

```ts
import { ActiveContext } from 'active-context';

const agent = new ActiveContext({
  modelName: 'lfm3:latest',
  baseUrl: 'http://127.0.0.1:11434',
  totalTokensLimit: 8192,
  options: { num_ctx: 8192, temperature: 0.2 },
  think: true,
});

agent.init(
  'Ты — автономный ИИ-разработчик. Управляй задачами через инструменты. ' +
  'Отвечай кратко на языке пользователя.',
);

const result = await agent.ask(
  'Создай задачу "Написать тесты" и подзадачу "Покрыть JWT"',
  {
    onThinking: (chunk) => process.stdout.write(chunk),
    onToolCall: (r) => console.log(`${r.success ? '✅' : '❌'} ${r.call.name}`),
    onContent: (chunk) => process.stdout.write(chunk),
    streamMode: 'realtime',
  },
);

console.log(result.content);
console.log(`Итераций: ${result.iterations}, tool calls: ${result.toolCalls.length}`);
```

---

## CLI

Интерактивный клиент для ручного тестирования агента.

```bash
npx active-context-cli
# или
npm run cli
```

```
🚀 ActiveContext interactive client
Модель:   lfm3:latest
Ollama:   http://127.0.0.1:11434
Thinking: on
Контекст: 38/8192 токенов, задач: 0

👤 создай задачи для покупок: молоко, мыло
🤔 We need to create two tasks...
✅ 🛠  createTask → Task created with ID f6333f14
✅ 🛠  createTask → Task created with ID a76fd002

Созданы задачи: Молоко (f6333f14), Мыло (a76fd002).
```

**Команды:**

| Команда | Описание |
|---|---|
| `/help`, `/h` | Справка |
| `/tasks`, `/t` | Список задач |
| `/new <title> [\| <desc>] [@<parentId>]` | Создать задачу вручную |
| `/find <regex>` | Поиск по истории (локальный, без модели) |
| `/msg <id>` | Показать сообщение по ID |
| `/think [on\|off]` | Переключить thinking |
| `/status`, `/s` | Состояние контекста |
| `/save` | Сбросить контекст и задачи на диск |
| `/exit`, `/q` | Выход |

**Флаги:**

```bash
npx active-context-cli \
  --model lfm3:latest \
  --host http://127.0.0.1:11434 \
  --tokens 8192 \
  --persist ./.active-context/session.json \
  --no-think \
  --verbose-thinking
```

`Ctrl+C` — прерывает текущий запрос; повторное нажатие выходит из CLI.

---

## Архитектура

```
┌──────────────────────────────────────────────────────────┐
│                   ActiveContext                          │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ LLMContext   │   │ TaskManager  │   │ OllamaClient │  │
│  │              │   │              │   │              │  │
│  │ - история    │   │ - CRUD задач │   │ - chat()     │  │
│  │ - токены     │   │ - parent/child│  │ - chatStream │  │
│  │ - eviction   │   │ - persistence│   │ - tool calls │  │
│  │ - regex      │   │              │   │ - thinking   │  │
│  └──────────────┘   └──────────────┘   └──────────────┘  │
│           │                 │                 │          │
│           └────────┬────────┴─────────────────┘          │
│                    │                                      │
│           ┌────────▼────────┐                            │
│           │  Agent Loop     │                            │
│           │  ask()          │                            │
│           └────────┬────────┘                            │
└────────────────────┼─────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ Ollama API   │
              │ /api/chat    │
              └──────────────┘
```

**Слои:**

- **`LLMContext`** — память модели. Скользящее окно токенов, `Map + массив` для O(1) доступа по ID, изолированный `system prompt`, атомарный `export`/`import`.
- **`TaskManager`** — CRUD задач, иерархия `parent/child` с защитой от циклов, JSON-persistence.
- **`OllamaJsClient`** — адаптер над официальной библиотекой `ollama`. Читает `thinking` и нативные `tool_calls`, склеивает аргументы по `function.index`.
- **`ActiveContext`** — оркестратор. Собирает «слоёный пирог» из `system prompt → TaskManager state → история`, ведёт agent loop, выполняет tool calls, пишет результаты обратно.

**Слоёный пирог** каждого запроса к Ollama:

```
1. system: базовый system prompt пользователя (изолирован)
2. system: инструкции TaskManager + текущее состояние задач
3. history: сообщения user / assistant / tool
4. tools: JSON Schema всех доступных функций
```

---

## API Reference

### `ActiveContext`

```ts
const agent = new ActiveContext(config, deps?);
```

**`config`:**

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `modelName` | `string` | — | Модель Ollama (`lfm3:latest`, `qwen3:8b`, …) |
| `baseUrl` | `string` | `http://127.0.0.1:11434` | URL Ollama API |
| `totalTokensLimit` | `number` | `8192` | Бюджет контекста |
| `options` | `Record<string, unknown>` | — | Параметры Ollama (`num_ctx`, `temperature`, …) |
| `think` | `boolean` | `true` | Включить thinking (для LFM3) |
| `tools` | `ToolDefinition[]` | `DEFAULT_TOOLS` | Кастомные tools |
| `contextPersistPath` | `string` | — | Путь к JSON для persistence контекста |

**`deps`:**

| Поле | Тип | Описание |
|---|---|---|
| `ollamaClient` | `OllamaClientLike` | Свой клиент (мок для тестов, другой провайдер) |
| `taskManager` | `TaskManager` | Свой менеджер задач |
| `tokenizer` | `TokenizerFn` | Свой токенизатор (например, `@huggingface/tokenizers`) |
| `validator` | `MessageValidator` | Свой валидатор prompt injection |

**Методы:**

```ts
agent.init(systemPrompt: string): void
agent.ask(userMessage: string, options?: AskOptions): Promise<AskResult>
agent.setThink(value: boolean): void
agent.isThinking(): boolean
agent.getContext(): LLMContext
agent.tasks: TaskManager
agent.getStatus(): ContextStorageStats
agent.flush(): void   // сбросить на диск
agent.close(): void   // flush() + graceful shutdown
```

**`AskOptions`:**

```ts
{
  onThinking?: (chunk: string) => void;
  onContent?: (chunk: string) => void;
  onToolCall?: (result: ToolExecutionResult) => void;
  maxIterations?: number;         // по умолчанию 5
  streamMode?: 'buffered' | 'realtime';
  signal?: AbortSignal;
  allowUnsafe?: boolean;          // пропустить валидацию ввода
}
```

**`AskResult`:**

```ts
{
  content: string;
  thinking: string;
  toolCalls: ToolExecutionResult[];
  iterations: number;
  stats: ContextStorageStats;
}
```

### `LLMContext`

```ts
const ctx = new LLMContext({ totalTokensLimit, tokenizer });

ctx.setSystemPrompt(prompt);
ctx.appendMessage('user', 'Hello');
ctx.findMessages({ pattern: 'Hello', role: 'user', limit: 10 });
ctx.getMessageById(id);
ctx.resolveMessageId(prefix);   // по 8-символьному префиксу
ctx.export();                    // снимок для сохранения на диск
ctx.import(data);                // восстановление, атомарно
ctx.clear();                     // очищает историю, system prompt остаётся
```

### `TaskManager`

```ts
const tm = new TaskManager({ persistPath: './tasks.json' });

tm.createTask('Title', 'Description', parentId?);
tm.updateTask(id, { title?, description?, status?, result?, parentId? });
tm.updateTaskStatus(id, status, result?);
tm.deleteTask(id);
tm.getAllTasks();
tm.getChildren(parentId);
tm.getTaskContextForLLM();   // строковое представление для system prompt
tm.flush();                   // синхронный сброс на диск
```

---

## Tools, доступные модели

`ActiveContext` регистрирует набор инструментов через JSON Schema. Модель вызывает их нативно.

**Задачи:**

| Tool | Параметры |
|---|---|
| `createTask` | `title`, `description?`, `parentId?` |
| `updateTask` | `id`, `title?`, `description?`, `status?`, `result?`, `parentId?` |
| `updateTaskStatus` | `id`, `status`, `result?` *(legacy)* |
| `deleteTask` | `id` |
| `listTasks` | — |

**Контекст:**

| Tool | Параметры |
|---|---|
| `searchMessages` | `pattern` (regex, ≤200 символов), `role?`, `limit?` |
| `getMessageById` | `id` (полный или уникальный префикс) |

**Кастомные tools:**

```ts
const agent = new ActiveContext({
  modelName: 'lfm3:latest',
  tools: [
    ...DEFAULT_TOOLS,
    {
      type: 'function',
      function: {
        name: 'fetchWeather',
        description: 'Get current weather for a city',
        parameters: {
          type: 'object',
          properties: { city: { type: 'string' } },
          required: ['city'],
        },
      },
    },
  ],
});
```

Обработчик добавляется через расширение `ActiveContext` или `deps.ollamaClient`.

---

## Persistence

```ts
const agent = new ActiveContext({
  modelName: 'lfm3:latest',
  contextPersistPath: './.active-context/session.json',
});

// При старте: загружает system prompt и историю
// При ask(): debounce 100ms → атомарная запись через tmp+rename
// При close(): гарантированный flush
```

**Что сохраняется:**

- `systemPrompt` — восстановить при следующем запуске
- история сообщений — включая `tool_calls` и `tool_call_id`
- задачи (если указан `persistPath` в `TaskManager`)

**Формат:** версионированный JSON (`version: 1`). Невалидный файл игнорируется с логированием, класс стартует с чистого состояния.

---

## Безопасность

- **System prompt изолирован** — недоступен через `searchMessages`, `getMessageById`, `updateMessageContent`, `clear()`.
- **Prompt injection** — `MessageValidator` блокирует типовые паттерны (`ignore all previous instructions`, `<|im_start|>` и т.д.) до добавления в контекст.
- **Санитизация задач** — поля `title`/`description`/`result` экранируются перед инжектом в system prompt.
- **Regex-лимиты** — паттерн ≤200 символов, `SyntaxError` возвращается как `[TOOL_RESULT][ERROR]`, не бросается наружу.
- **Циклы в parent/child** — `updateTask` отвергает попытки сделать задачу родителем самой себе или потомка.
- **Лимит итераций** — `MaxIterationsError` защищает от зацикливания агента.

**Что НЕ закрыто:**

- ReDoS через сложные регулярные выражения (лимит 200 символов снижает риск, но не устраняет).
- Jailbreak через многоходовые атаки — валидатор смотрит на одно сообщение, не на диалог.
- Модель может **галлюцинировать в summary** tool results (см. Roadmap).

Для production-сценариев рекомендуется заменить `MessageValidator` на ML-классификатор (`prompt-guard`, `llama-guard`).

---

## Тестирование

```bash
npm test
```

**54 теста**, покрывающих:

- `LLMContext` — CRUD, eviction, regex, export/import, prefix resolution
- `TaskManager` — CRUD, parent/child, циклы, persistence
- `ActiveContext` — agent loop, recovery после ошибки, persistence, `setThink`, tool dispatch
- `MessageValidator` — prompt injection паттерны, санитизация
- `LFM3Parser` — legacy XML (для обратной совместимости)

Тесты используют `ScriptedClient` (мок `OllamaClientLike`) — не требуют запущенной Ollama.

---

## Roadmap

- [ ] **Реальный токенизатор** — `@huggingface/tokenizers` + `tokenizer.json` модели (сейчас эвристика `length / 4`).
- [ ] **RAG** — `reservedTokens` уже готов принять бюджет под retrieved chunks, третий системный слой между TaskManager и историей.
- [ ] **Суммаризация истории** — `LLMContext.compress()` через вспомогательную модель вместо FIFO.
- [ ] **ReDoS-защита** — таймаут regex через `AbortSignal`.
- [ ] **Tool result faithfulness** — программный guard: если модель пересказывает tool result в формате, которого не было (markdown-таблица вместо списка), возвращать ошибку.
- [ ] **Multi-agent** — динамическая регистрация tools в рантайме (`registerTool(def, handler)`).
- [ ] **Subpath exports** — `import { LLMContext } from 'active-context'` без загрузки `ollama`.

---

## Совместимость

| Модель | Tool calling | Thinking | Статус |
|---|---|---|---|
| `lfm3:latest` | ✅ | ✅ | Проверено |
| `qwen3:8b` | ✅ | ✅ | Ожидается |
| `llama3.1:8b` | ✅ | — | Ожидается |
| Модели без native tools | ❌ | — | Не поддерживается (см. legacy `LFM3Parser`) |

`ActiveContext` использует **нативный tool calling Ollama** (`tools` в `/api/chat`). Модели без поддержки tools работать не будут — для них нужен fallback на XML-парсинг (в кодовой базе сохранён `LFM3Parser`, но `ActiveContext` его не использует).

---

## Лицензия

MIT

---

## Ссылки

- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama JS client](https://github.com/ollama/ollama-js)
- [LFM3 model](https://ollama.com/library/lfm3)
- [Model Context Protocol](https://modelcontextprotocol.io) — альтернативный стандарт для tool integration
```

---

## Файлы, которые стоит добавить рядом с README

**`LICENSE`** (MIT):

```
MIT License

Copyright (c) 2026 <your name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**`.gitignore`:**

```
node_modules/
dist/
*.log
.active-context/
.tmp/
.DS_Store
coverage/
```

**`CHANGELOG.md`** — начать с:

```markdown
# Changelog

## [0.1.0] — 2026-11-03

### Added
- `LLMContext` — token-budgeted history with FIFO eviction and isolated system prompt.
- `TaskManager` — CRUD with parent/child hierarchy, cycle protection, JSON persistence.
- `ActiveContext` — agent loop with native Ollama tool calling (no XML parsing).
- Native tools: `createTask`, `updateTask`, `updateTaskStatus`, `deleteTask`, `listTasks`, `searchMessages`, `getMessageById`.
- Thinking-стриминг (native `thinking` chunks from LFM3).
- Regex-поиск по истории (`searchMessages`) с защитой от ReDoS-паттернов (limit 200 chars).
- Persistence контекста через `contextPersistPath` — атомарная запись, версионированный JSON.
- Интерактивный CLI (`active-context-cli`) с командами `/tasks`, `/find`, `/msg`, `/think`, `/status`.
- Валидация prompt injection в `MessageValidator` + санитизация полей задач.
- 54 unit-теста.

### Changed
- Миграция с XML tool calling (`<tool_call>`) на нативные Ollama tools — совместимо с LFM3, Qwen3, Llama 3.1.
- `OllamaClient` помечен `@deprecated`, заменён на `OllamaJsClient` (обёртка над `ollama-js`).
- `reservedTokens` теперь включает JSON Schema всех tools — предотвращает переполнение окна.

### Deprecated
- `LFM3Parser` и `StreamRouter` — остаются для обратной совместимости, но `ActiveContext` их не использует.
- `updateTaskStatus` — оставлен как обёртка над `updateTask`, но рекомендуется использовать `updateTask` для полного CRUD.

### Known issues
- Модель может галлюцинировать в summary tool results (см. Roadmap → Tool result faithfulness).
- `defaultTokenizer` использует `length / 4` — на русском тексте погрешность до ±30%.
- ReDoS через сложные regex не блокируется, только ограничен длиной паттерна.
