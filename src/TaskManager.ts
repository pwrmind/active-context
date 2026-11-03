import { randomUUID } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import type { Task, TaskStatus } from './types.js';
import { MessageValidator } from './validator.js';

const VALID_STATUSES: TaskStatus[] = ['todo', 'in_progress', 'completed', 'failed'];

export interface TaskManagerOptions {
  persistPath?: string;
  validator?: MessageValidator;
}

export class TaskManager {
  private tasks = new Map<string, Task>();
  private readonly persistPath?: string;
  private readonly validator: MessageValidator;
  private saveTimer: NodeJS.Timeout | null = null;

  constructor(options: TaskManagerOptions = {}) {
    this.persistPath = options.persistPath;
    this.validator = options.validator ?? new MessageValidator();
    if (this.persistPath && existsSync(this.persistPath)) {
      this.load();
    }
  }

  public createTask(title: string, description?: string, parentId?: string): Task {
    if (!title || title.trim().length === 0) {
      throw new Error('Task title is required');
    }
    if (parentId !== undefined && !this.tasks.has(parentId)) {
      throw new Error(`Parent task ${parentId} not found`);
    }
    const id = randomUUID().substring(0, 8);
    const now = Date.now();
    const task: Task = {
      id,
      title: title.trim(),
      description: description?.trim(),
      status: 'todo',
      parentId,
      createdAt: now,
      updatedAt: now,
    };
    this.tasks.set(id, task);
    this.scheduleSave();
    return task;
  }

  public getTaskById(id: string): Task | undefined {
    return this.tasks.get(id);
  }

  public getChildren(parentId: string): Task[] {
    return this.getAllTasks().filter((t) => t.parentId === parentId);
  }

  public updateTaskStatus(id: string, status: TaskStatus, result?: string): Task {
    return this.updateTask(id, { status, result });
  }

  /**
   * Универсальное обновление задачи. Не меняет id, createdAt.
   * Проверяет: непустой title, валидный status, отсутствие циклов в parentId.
   */
  public updateTask(id: string, patch: TaskUpdate): Task {
    const task = this.tasks.get(id);
    if (!task) throw new Error(`Task ${id} not found`);

    if (patch.title !== undefined) {
      const trimmed = patch.title.trim();
      if (!trimmed) throw new Error('Title must be non-empty');
      task.title = trimmed;
    }

    if (patch.description !== undefined) {
      const trimmed = patch.description.trim();
      task.description = trimmed || undefined;
    }

    if (patch.status !== undefined) {
      if (!VALID_STATUSES.includes(patch.status)) {
        throw new Error(
          `Invalid status "${patch.status}". Expected one of: ${VALID_STATUSES.join(', ')}`,
        );
      }
      task.status = patch.status;
    }

    if (patch.result !== undefined) {
      task.result = patch.result;
    }

    if (patch.parentId !== undefined) {
      if (patch.parentId === null) {
        task.parentId = undefined;
      } else {
        if (patch.parentId === id) {
          throw new Error('Task cannot be its own parent');
        }
        if (!this.tasks.has(patch.parentId)) {
          throw new Error(`Parent task ${patch.parentId} not found`);
        }
        // защита от цикла: новый родитель не может быть потомком этой задачи
        let cursor: string | undefined = patch.parentId;
        const seen = new Set<string>();
        while (cursor) {
          if (cursor === id) {
            throw new Error('Cannot set descendant as parent (cycle)');
          }
          if (seen.has(cursor)) break;
          seen.add(cursor);
          cursor = this.tasks.get(cursor)?.parentId;
        }
        task.parentId = patch.parentId;
      }
    }

    task.updatedAt = Date.now();
    this.scheduleSave();
    return task;
  }

  public deleteTask(id: string): boolean {
    const existed = this.tasks.delete(id);
    if (existed) this.scheduleSave();
    return existed;
  }

  public getAllTasks(): Task[] {
    return Array.from(this.tasks.values());
  }

  public getActiveTasks(): Task[] {
    return this.getAllTasks().filter(
      (t) => t.status === 'todo' || t.status === 'in_progress',
    );
  }

  public clear(): void {
    this.tasks.clear();
    this.flush();
  }

  public flush(): void {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    this.save();
  }

  public getTaskContextForLLM(): string {
    if (this.tasks.size === 0) return 'No tasks available.';
    return this.getAllTasks()
      .map((t) => {
        const safeTitle = this.validator.sanitizeForSystemContext(t.title);
        const safeDesc = t.description
          ? ` Description: "${this.validator.sanitizeForSystemContext(t.description)}"`
          : '';
        const safeResult = t.result
          ? ` Result: "${this.validator.sanitizeForSystemContext(t.result)}"`
          : '';
        const parent = t.parentId ? ` parent=${t.parentId}` : '';
        return `- [ID: ${t.id}${parent}][Status: ${t.status}] Title: "${safeTitle}"${safeDesc}${safeResult}`;
      })
      .join('\n');
  }

  private load(): void {
    try {
      const raw = readFileSync(this.persistPath!, 'utf-8');
      const arr = JSON.parse(raw) as Task[];
      if (!Array.isArray(arr)) return;
      for (const t of arr) {
        if (t && typeof t.id === 'string' && VALID_STATUSES.includes(t.status)) {
          this.tasks.set(t.id, t);
        }
      }
    } catch {
      /* corrupted file ignored */
    }
  }

  private scheduleSave(): void {
    if (!this.persistPath) return;
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      this.save();
    }, 100);
  }

  private save(): void {
    if (!this.persistPath) return;
    try {
      mkdirSync(dirname(this.persistPath), { recursive: true });
      writeFileSync(this.persistPath, JSON.stringify(this.getAllTasks(), null, 2));
    } catch (e) {
      console.error('TaskManager: failed to persist tasks', e);
    }
  }
}