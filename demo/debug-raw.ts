// demo/debug-raw.ts
import { Ollama } from 'ollama';

async function main(): Promise<void> {
  const client = new Ollama({ host: 'http://127.0.0.1:11434' });
  const model = process.env.OLLAMA_MODEL ?? 'lfm3:latest';

  console.log(`=== Non-streaming: ${model} ===\n`);
  const res = await client.chat({
    model,
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Say hello in one word.' },
    ],
    options: { num_ctx: 8192 },
    think: true,
  });
  console.log('message:', JSON.stringify((res as any).message, null, 2));

  console.log('\n=== Streaming: same prompt ===\n');
  const stream = await client.chat({
    model,
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Say hello in one word.' },
    ],
    options: { num_ctx: 8192 },
    think: true,
    stream: true,
  });
  for await (const part of stream as AsyncIterable<any>) {
    console.log('chunk:', JSON.stringify(part.message, null, 2));
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});