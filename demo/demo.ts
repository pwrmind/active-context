import { ActiveContext } from '../src/ActiveContext.js';
import type { OllamaClientLike } from '../src/OllamaClient.js';

// Скриптованный клиент для демонстрации работы агента без запущенной Ollama.
// Возвращает tool call на первой итерации и финальный текст — на второй.
class MockOllamaClient implements OllamaClientLike {
  private callCount = 0;

  async chat(_req: any): Promise<{ content: string }> {
    this.callCount++;
    if (this.callCount === 1) {
      return {
        content: [
          '<think>',
          'Пользователь просит создать задачу. Использую функцию createTask.',
          '</think>',
          '<tool_call>',
          '<function=createTask>',
          '<parameter=title>Написать тесты для модуля авторизации</parameter>',
          '<parameter=description>Тесты должны покрывать JWT-сценарии</parameter>',
          '</function>',
          '</tool_call>',
        ].join(''),
      };
    }
    return {
      content: [
        '<think>',
        'Задача успешно создана, ID получен. Отвечаю пользователю кратко.',
        '</think>',
        'Готово: создал задачу на написание тестов для модуля авторизации с фокусом на JWT.',
      ].join(''),
    };
  }

  async *chatStream(req: any): AsyncGenerator<string> {
    const { content } = await this.chat(req);
    // Разбиваем на куски по границам символов, чтобы эмулировать стриминг
    const chunks = content.match(/.{1,20}/gs) ?? [content];
    for (const c of chunks) yield c;
  }
}

async function main(): Promise<void> {
  console.log('🚀 ActiveContext demo\n');

  const agent = new ActiveContext(
    {
      modelName: 'lfm3:latest',
      totalTokensLimit: 4096,
    },
    { ollamaClient: new MockOllamaClient() },
  );

  agent.init('Ты — автономный ИИ-разработчик. Помогай управлять проектом.');

  const request =
    'Создай задачу на написание тестов для модуля авторизации. Тесты должны покрывать JWT.';
  console.log(`👤 Пользователь: ${request}\n`);

  // Управление потоковым выводом thinking:
  // префикс "🧠 " печатается один раз, далее текст идёт инлайн.
  let thinkingOpen = false;
  const closeThinking = (): void => {
    if (thinkingOpen) {
      process.stdout.write('\n');
      thinkingOpen = false;
    }
  };

  const result = await agent.ask(request, {
    onThinking: (chunk) => {
      if (!thinkingOpen) {
        process.stdout.write('🧠 ');
        thinkingOpen = true;
      }
      process.stdout.write(chunk);
    },
    onToolCall: (r) => {
      closeThinking();
      const icon = r.success ? '✅' : '❌';
      console.log(`🛠  ${icon} ${r.call.functionName} → ${r.message}`);
    },
  });

  closeThinking();

  console.log(`\n🤖 Ассистент: ${result.content}`);
  console.log(`\n📊 Итераций агента: ${result.iterations}`);
  console.log(`🛠  Tool calls: ${result.toolCalls.length}`);

  console.log('\n📋 Задачи в TaskManager:');
  for (const t of agent.tasks.getAllTasks()) {
    console.log(`   [${t.id}] ${t.title} — ${t.status}`);
    if (t.description) console.log(`       └ ${t.description}`);
  }

  console.log('\n📈 Состояние контекста:');
  console.log(JSON.stringify(agent.getStatus(), null, 2));
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});