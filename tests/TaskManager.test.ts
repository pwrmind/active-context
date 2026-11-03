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

test('updateTask renames without changing id or createdAt', () => {
  const tm = new TaskManager();
  const t = tm.createTask('Old', 'old-desc');
  const updated = tm.updateTask(t.id, { title: 'New', description: 'new-desc' });
  assert.equal(updated.id, t.id);
  assert.equal(updated.createdAt, t.createdAt);
  assert.equal(updated.title, 'New');
  assert.equal(updated.description, 'new-desc');
  assert.equal(tm.getAllTasks().length, 1);
});

test('updateTask reparents a task', () => {
  const tm = new TaskManager();
  const p = tm.createTask('Parent');
  const c = tm.createTask('Child');
  tm.updateTask(c.id, { parentId: p.id });
  assert.equal(tm.getTaskById(c.id)?.parentId, p.id);
  assert.equal(tm.getChildren(p.id).length, 1);
});

test('updateTask detaches with parentId=null', () => {
  const tm = new TaskManager();
  const p = tm.createTask('Parent');
  const c = tm.createTask('Child', undefined, p.id);
  tm.updateTask(c.id, { parentId: null });
  assert.equal(tm.getTaskById(c.id)?.parentId, undefined);
});

test('updateTask rejects self as parent', () => {
  const tm = new TaskManager();
  const t = tm.createTask('X');
  assert.throws(() => tm.updateTask(t.id, { parentId: t.id }), /own parent/);
});

test('updateTask rejects cycle', () => {
  const tm = new TaskManager();
  const a = tm.createTask('A');
  const b = tm.createTask('B', undefined, a.id);
  const c = tm.createTask('C', undefined, b.id);
  assert.throws(() => tm.updateTask(a.id, { parentId: c.id }), /cycle/);
});

test('updateTask rejects empty title', () => {
  const tm = new TaskManager();
  const t = tm.createTask('X');
  assert.throws(() => tm.updateTask(t.id, { title: '   ' }), /non-empty/);
});

test('updateTask preserves status and result when not patched', () => {
  const tm = new TaskManager();
  const t = tm.createTask('X');
  tm.updateTaskStatus(t.id, 'in_progress', 'working');
  tm.updateTask(t.id, { title: 'Y' });
  const after = tm.getTaskById(t.id)!;
  assert.equal(after.status, 'in_progress');
  assert.equal(after.result, 'working');
  assert.equal(after.title, 'Y');
});