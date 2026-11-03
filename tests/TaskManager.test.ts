import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { TaskManager } from '../src/TaskManager.js';

test('CRUD on tasks', () => {
  const tm = new TaskManager();
  const t = tm.createTask('Task 1', 'desc');
  assert.equal(tm.getTaskById(t.id)?.title, 'Task 1');
  tm.updateTaskStatus(t.id, 'completed', 'done');
  assert.equal(tm.getTaskById(t.id)?.status, 'completed');
  assert.equal(tm.getTaskById(t.id)?.result, 'done');
  assert.equal(tm.deleteTask(t.id), true);
  assert.equal(tm.getTaskById(t.id), undefined);
});

test('rejects invalid status', () => {
  const tm = new TaskManager();
  const t = tm.createTask('X');
  assert.throws(() => tm.updateTaskStatus(t.id, 'nonsense' as any));
});

test('persists and restores tasks', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ac-tm-'));
  const path = join(dir, 'tasks.json');

  try {
    const tm1 = new TaskManager({ persistPath: path });
    tm1.createTask('Persisted');
    tm1.flush();

    const tm2 = new TaskManager({ persistPath: path });
    const restored = tm2.getAllTasks();
    assert.equal(restored.length, 1);
    assert.equal(restored[0].title, 'Persisted');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('creates parent directory on save', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ac-tm-nested-'));
  const path = join(dir, 'nested', 'deeper', 'tasks.json');

  try {
    const tm = new TaskManager({ persistPath: path });
    tm.createTask('Nested');
    tm.flush();

    const tm2 = new TaskManager({ persistPath: path });
    const restored = tm2.getAllTasks();
    assert.equal(restored.length, 1);
    assert.equal(restored[0].title, 'Nested');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('sanitizes task fields in LLM context', () => {
  const tm = new TaskManager();
  tm.createTask('<bad>title</bad>', 'with <xml> and ]brackets[');
  const ctx = tm.getTaskContextForLLM();
  assert.ok(!ctx.includes('<bad>'));
  assert.ok(!ctx.includes(']brackets['));
});

test('createTask with parentId links subtask', () => {
  const tm = new TaskManager();
  const parent = tm.createTask('Parent');
  const child = tm.createTask('Child', 'details', parent.id);
  assert.equal(child.parentId, parent.id);
  assert.equal(tm.getChildren(parent.id).length, 1);
  assert.equal(tm.getChildren(parent.id)[0].id, child.id);
});

test('createTask rejects unknown parentId', () => {
  const tm = new TaskManager();
  assert.throws(() => tm.createTask('Orphan', undefined, 'nonexistent'), /Parent task/);
});

test('parentId appears in LLM context', () => {
  const tm = new TaskManager();
  const parent = tm.createTask('Parent');
  tm.createTask('Child', undefined, parent.id);
  const ctx = tm.getTaskContextForLLM();
  assert.match(ctx, new RegExp(`parent=${parent.id}`));
});