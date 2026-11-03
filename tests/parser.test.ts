import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LFM3Parser } from '../src/parser.js';

test('parses thinking + tool_call + content', () => {
  const raw = `<think>need to create</think>
<tool_call><function=createTask><parameter=title>X</parameter></function></tool_call>
Done!`;
  const p = LFM3Parser.parseAll(raw);
  assert.equal(p.thinking, 'need to create');
  assert.equal(p.cleaned, 'Done!');
  assert.equal(p.calls.length, 1);
  assert.equal(p.calls[0].functionName, 'createTask');
  assert.equal(p.calls[0].args.title, 'X');
});

test('ignores unknown functions', () => {
  const raw = `<tool_call><function=hackThePlanet><parameter=x>y</parameter></function></tool_call>`;
  assert.equal(LFM3Parser.extractToolCalls(raw).length, 0);
});

test('handles multiple tool calls in one response', () => {
  const raw = `
<tool_call><function=createTask><parameter=title>A</parameter></function></tool_call>
<tool_call><function=createTask><parameter=title>B</parameter></function></tool_call>`;
  assert.equal(LFM3Parser.extractToolCalls(raw).length, 2);
});

test('tolerates missing closing function tag', () => {
  const raw = `<tool_call><function=createTask><parameter=title>X</parameter></tool_call>`;
  const calls = LFM3Parser.extractToolCalls(raw);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].args.title, 'X');
});