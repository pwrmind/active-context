import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LLMContext } from '../src/LLMContext.js';

test('system prompt is isolated and not evicted', () => {
  const ctx = new LLMContext({ totalTokensLimit: 100 });
  ctx.setSystemPrompt('You are a helpful assistant.');
  for (let i = 0; i < 50; i++) {
    ctx.appendMessage('user', 'x'.repeat(100));
  }
  assert.equal(ctx.getSystemPrompt(), 'You are a helpful assistant.');
});

test('evicts oldest messages on overflow', () => {
  const ctx = new LLMContext({ totalTokensLimit: 100 });
  const id1 = ctx.appendMessage('user', 'a'.repeat(40));
  ctx.appendMessage('user', 'b'.repeat(40));
  ctx.appendMessage('user', 'c'.repeat(40));
  ctx.appendMessage('user', 'd'.repeat(400));
  assert.equal(ctx.getMessageById(id1), undefined);
});

test('reservedTokens учитываются в статистике', () => {
  const ctx = new LLMContext({ totalTokensLimit: 100 });
  ctx.setSystemPrompt('hi');
  ctx.setReservedTokens(30);
  ctx.appendMessage('user', 'hello world');
  const s = ctx.stats;
  assert.equal(s.reservedTokens, 30);
  assert.equal(s.currentTokensUsed, s.systemPromptTokens + 30 + s.historyTokens);
});

test('findMessages фильтрует по роли и подстроке', () => {
  const ctx = new LLMContext();
  ctx.appendMessage('user', 'hello world');
  ctx.appendMessage('assistant', 'hi there');
  ctx.appendMessage('user', 'another question');
  assert.equal(ctx.findMessages({ role: 'user' }).length, 2);
  assert.equal(ctx.findMessages({ query: 'world' }).length, 1);
});

test('export and import round-trip preserves state including tool_calls', () => {
  const ctx1 = new LLMContext({ totalTokensLimit: 1000 });
  ctx1.setSystemPrompt('You are helpful');
  ctx1.setReservedTokens(42);
  const id1 = ctx1.appendMessage('user', 'hello');
  ctx1.appendMessage('assistant', '', {
    toolCalls: [
      { id: 'call_1', function: { name: 'createTask', arguments: { title: 'X' } } },
    ],
  });
  ctx1.appendMessage('tool', '[TOOL_RESULT][SUCCESS] ok', {
    toolCallId: 'call_1',
    metadata: { source: 'createTask' },
  });

  const data = ctx1.export();
  assert.equal(data.version, 1);
  assert.equal(data.systemPrompt, 'You are helpful');
  assert.equal(data.reservedTokens, 42);
  assert.equal(data.messages.length, 3);

  const ctx2 = new LLMContext({ totalTokensLimit: 1000 });
  ctx2.import(data);

  assert.equal(ctx2.getSystemPrompt(), 'You are helpful');
  assert.equal(ctx2.stats.reservedTokens, 42);
  assert.equal(ctx2.stats.historyTokens, ctx1.stats.historyTokens);
  assert.deepEqual(ctx2.toPipelineMessages(), ctx1.toPipelineMessages());

  const restored = ctx2.getMessageById(id1);
  assert.equal(restored?.content, 'hello');
});

test('export returns a defensive copy', () => {
  const ctx = new LLMContext();
  ctx.setSystemPrompt('sys');
  ctx.appendMessage('user', 'hi');
  const data = ctx.export();
  data.messages[0].content = 'TAMPERED';
  assert.equal(ctx.findMessages({ role: 'user' })[0].content, 'hi');
});

test('import rejects unsupported version and keeps state intact', () => {
  const ctx = new LLMContext();
  ctx.setSystemPrompt('original');
  ctx.appendMessage('user', 'existing');

  assert.throws(
    () =>
      ctx.import({
        // @ts-expect-error intentionally wrong version
        version: 999,
        systemPrompt: 'new',
        reservedTokens: 0,
        messages: [],
      }),
    /Unsupported context version/,
  );

  assert.equal(ctx.getSystemPrompt(), 'original');
  assert.equal(ctx.findMessages({ role: 'user' }).length, 1);
});

test('import rejects invalid messages and keeps state intact', () => {
  const ctx = new LLMContext();
  ctx.setSystemPrompt('original');

  assert.throws(
    () =>
      ctx.import({
        version: 1,
        systemPrompt: 'new',
        reservedTokens: 0,
        messages: [{ id: 'x', role: 'system' as any, content: 'bad', timestamp: 0, tokensCount: 1 }],
      }),
    /role/,
  );

  assert.equal(ctx.getSystemPrompt(), 'original');
});

test('import recalculates tokensCount if missing or zero', () => {
  const ctx = new LLMContext({ tokenizer: (t) => t.length });
  ctx.import({
    version: 1,
    systemPrompt: null,
    reservedTokens: 0,
    messages: [
      { id: 'a', role: 'user', content: 'hello', timestamp: 1, tokensCount: 0 },
      { id: 'b', role: 'user', content: 'hi', timestamp: 2, tokensCount: 99 },
    ],
  });
  assert.equal(ctx.stats.historyTokens, 5 + 99);
});

test('findMessages по regex pattern (case-insensitive)', () => {
  const ctx = new LLMContext();
  ctx.appendMessage('user', 'Hello World');
  ctx.appendMessage('assistant', 'hi there');
  ctx.appendMessage('user', 'goodbye world');

  assert.equal(ctx.findMessages({ pattern: 'world' }).length, 2);
  assert.equal(ctx.findMessages({ pattern: '^hello' }).length, 1);
  assert.equal(ctx.findMessages({ pattern: 'bye\\s+world$' }).length, 1);
  assert.equal(ctx.findMessages({ pattern: 'nope' }).length, 0);
});

test('findMessages по regex + role filter', () => {
  const ctx = new LLMContext();
  ctx.appendMessage('user', 'hello');
  ctx.appendMessage('assistant', 'hello hello');
  assert.equal(ctx.findMessages({ pattern: 'hello', role: 'user' }).length, 1);
  assert.equal(ctx.findMessages({ pattern: 'hello', role: 'assistant' }).length, 1);
});

test('findMessages rejects invalid regex', () => {
  const ctx = new LLMContext();
  ctx.appendMessage('user', 'x');
  assert.throws(() => ctx.findMessages({ pattern: '[unclosed' }), /Invalid regex/);
});

test('findMessages rejects too-long pattern', () => {
  const ctx = new LLMContext();
  ctx.appendMessage('user', 'x');
  const long = 'a'.repeat(201);
  assert.throws(() => ctx.findMessages({ pattern: long }), /too long/);
});

test('findMessages limits results to last N', () => {
  const ctx = new LLMContext();
  for (let i = 0; i < 10; i++) ctx.appendMessage('user', `msg ${i}`);
  const r = ctx.findMessages({ pattern: 'msg', limit: 3 });
  assert.equal(r.length, 3);
  assert.match(r[2].content, /msg 9/);
});

test('resolveMessageId finds by unique prefix', () => {
  const ctx = new LLMContext();
  const id = ctx.appendMessage('user', 'hello');
  const prefix = id.slice(0, 8);
  assert.equal(ctx.resolveMessageId(prefix)?.id, id);
  assert.equal(ctx.resolveMessageId('nonexistent'), undefined);
});