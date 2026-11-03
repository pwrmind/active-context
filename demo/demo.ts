import { ActiveContext } from '../src/ActiveContext.js';
import type {
  OllamaChatResponse,
  OllamaClientLike,
  StreamChunk,
} from '../src/types.js';

class MockOllamaClient implements OllamaClientLike {
  private callCount = 0;

  async chat(): Promise<OllamaChatResponse> {
    this.callCount++;
    if (this.callCount === 1) {
      return {
        content: '',
        thinking: 'Пользователь просит создать задачу. Вызываю createTask.',
        toolCalls: [
          {
            id: 'call_1',
            name: 'createTask',
            arguments: {
              title: 'Написать тесты для модуля авторизации',
              description: 'Тесты должны покрывать JWT-сценарии',
            },
          },
        ],
      };
    }
    return {
      content:
        'Готово: создал задачу на написание тестов для модуля авторизации с фокусом на JWT.',
      thinking: 'Задача успешно создана, ID получен. Отвечаю пользователю кратко.',
      toolCalls: [],
    };
  }

  async *chatStream(): AsyncGenerator<StreamChunk> {
    const res = await this.chat();
    // Эмулируем стриминг thinking по словам
    if (res.thinking) {
      for (const word of res.thinking.split(/(\s+)/)) {
        if (word) yield { content: '', thinking: word, toolCalls: [] };
      }
    }
    if (res.content) yield { content: res.content, thinking: '', toolCalls: [] };
    if (res.toolCalls.length > 0) {
      yield { content: '', thinking: '', toolCalls: res.toolCalls };
    }
  }
}

async function main(): Promise<void> {
  console.log('🚀 ActiveContext demo\n');

  const agent = new ActiveContext(
    {
      modelName: 'lfm3:latest',
      totalTokensLimit: 4096,
      think: true,
    },
    { ollamaClient: new MockOllamaClient() },
  );

  agent.init('Ты — автономный ИИ-разработчик. Помогай управлять проектом.');

  const request =
    'Создай задачу на написание тестов для модуля авторизации. Тесты должны покрывать JWT.';
  console.log(`👤 Пользователь: ${request}\n`);

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
      console.log(`🛠  ${icon} ${r.call.name} → ${r.message}`);
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