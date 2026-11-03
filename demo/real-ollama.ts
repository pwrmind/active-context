// demo/real-ollama.ts
import { ActiveContext } from '../src/ActiveContext.js';

async function main(): Promise<void> {
  const model = process.env.OLLAMA_MODEL ?? 'lfm3:latest';
  const baseUrl = process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434';

  console.log(`🔌 Подключение к Ollama: ${baseUrl}`);
  console.log(`🧠 Модель: ${model}\n`);

  const agent = new ActiveContext({
    modelName: model,
    baseUrl,
    totalTokensLimit: 8192,
    options: {
      num_ctx: 8192,      // синхронизировано с totalTokensLimit
      temperature: 0.2,   // ниже температура — стабильнее XML
    },
    // Раскомментируйте, чтобы диалог переживал перезапуск скрипта:
    // contextPersistPath: './.active-context/session.json',
  });

  agent.init(
    'Ты — автономный ИИ-разработчик. Управляй задачами строго через XML-теги, ' +
    'как описано в системном контексте Task Manager. Не добавляй пояснений вне тегов, ' +
    'если не уверен в формате.',
  );

  const request =
    'Создай задачу: "Проверить реальный вызов Ollama". ' +
    'В описании укажи, что нужно протестировать стриминг и tool calling.';

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
      console.log(`🛠  ${icon} ${r.call.functionName} → ${r.message}`);
    },
  });

  closeThinking();

  console.log(`\n🤖 Ассистент: ${result.content}`);
  console.log(`\n📊 Итераций: ${result.iterations}`);
  console.log(`🛠  Tool calls: ${result.toolCalls.length}`);
  console.log(`🧠 Thinking (первые 200 символов): ${result.thinking.slice(0, 200)}…`);

  console.log('\n📋 Задачи:');
  for (const t of agent.tasks.getAllTasks()) {
    console.log(`   [${t.id}] ${t.title} — ${t.status}`);
  }

  console.log('\n📈 Контекст:');
  console.log(JSON.stringify(agent.getStatus(), null, 2));

  // Гарантируем сохранение, если включён contextPersistPath
  agent.close();
}

main().catch((err) => {
  console.error('\n💥 Ошибка реального вызова:', err.message);
  console.error(
    '\nПроверьте:\n' +
      '  1. Запущен ли `ollama serve`\n' +
      `  2. Скачана ли модель: ollama pull ${process.env.OLLAMA_MODEL ?? 'lfm3:latest'}\n` +
      '  3. Доступен ли порт 11434\n',
  );
  process.exit(1);
});