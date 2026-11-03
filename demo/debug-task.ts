// demo/debug-task.ts
import { Ollama } from 'ollama';

const SYSTEM_XML = `[TASK MANAGER MODULE ACTIVE]
You have access to a built-in Task Manager. You MUST use the following XML format to interact with tasks.

Format for reasoning:
<think>your reasoning</think>

Format for tool calls:
<tool_call><function=FUNCTION_NAME><parameter=PARAM>value</parameter></function></tool_call>

Available functions: createTask(title, description), updateTaskStatus(id, status, result), deleteTask(id), listTasks().`;

const TASK_TOOLS = [
  {
    type: 'function' as const,
    function: {
      name: 'createTask',
      description: 'Create a new task',
      parameters: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          description: { type: 'string' },
        },
        required: ['title'],
      },
    },
  },
];

const USER =
  'Создай задачу: "Проверить реальный вызов Ollama". В описании укажи про стриминг.';

async function runCase(label: string, params: Record<string, unknown>): Promise<void> {
  const client = new Ollama({ host: 'http://127.0.0.1:11434' });
  console.log(`\n=== ${label} ===`);
  try {
    const stream = await client.chat({
      model: 'lfm3:latest',
      messages: [
        { role: 'system', content: SYSTEM_XML },
        { role: 'user', content: USER },
      ],
      options: { num_ctx: 8192 },
      stream: true,
      ...params,
    } as any);

    const chunks: any[] = [];
    for await (const part of stream as AsyncIterable<any>) {
      chunks.push(part.message);
    }
    // Сводка: что вообще пришло
    const combined = {
      content: chunks.map((c) => c.content ?? '').join(''),
      thinking: chunks.map((c) => c.thinking ?? '').join(''),
      tool_calls: chunks.flatMap((c) => c.tool_calls ?? []),
    };
    console.log('content:', JSON.stringify(combined.content));
    console.log('thinking:', JSON.stringify(combined.thinking.slice(0, 300)));
    console.log('tool_calls:', JSON.stringify(combined.tool_calls, null, 2));
  } catch (e: any) {
    console.error('error:', e.message);
  }
}

async function main(): Promise<void> {
  await runCase('1. XML prompt, think: false (текущее поведение ActiveContext)', {
    think: false,
  });
  await runCase('2. XML prompt, think: true (стриминг thinking)', {
    think: true,
  });
  await runCase('3. Native tools, think: true (правильный путь для LFM3)', {
    think: true,
    tools: TASK_TOOLS,
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});