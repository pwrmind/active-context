import type { ParsedResponse, ParsedToolCall } from './types.js';

const TOOL_CALL_REGEX = /<tool_call\s*>([\s\S]*?)<\/tool_call>/g;
const THINKING_REGEX = /<think\s*>([\s\S]*?)<\/think>/g;
const FUNCTION_REGEX = /<function\s*=\s*([^>\s]+)\s*>([\s\S]*?)(?:<\/function>|(?=<function)|$)/;
const PARAMETER_REGEX = /<parameter\s*=\s*([^>\s]+)\s*>([\s\S]*?)<\/parameter>/g;

export const KNOWN_FUNCTIONS = new Set([
  'createTask',
  'updateTaskStatus',
  'deleteTask',
  'listTasks',
]);

export class LFM3Parser {
  /**
   * Полный разбор ответа модели: извлекает thinking, tool calls и очищенный текст.
   */
  public static parseAll(text: string): ParsedResponse {
    const thinking = LFM3Parser.extractThinking(text);
    const calls = LFM3Parser.extractToolCalls(text);
    const cleaned = text
      .replace(THINKING_REGEX, '')
      .replace(TOOL_CALL_REGEX, '')
      .trim();
    return { thinking, cleaned, calls };
  }

  public static extractThinking(text: string): string {
    const parts: string[] = [];
    for (const m of text.matchAll(THINKING_REGEX)) {
      parts.push(m[1].trim());
    }
    return parts.join('\n').trim();
  }

  public static extractToolCalls(text: string): ParsedToolCall[] {
    const calls: ParsedToolCall[] = [];
    for (const m of text.matchAll(TOOL_CALL_REGEX)) {
      const inner = m[1];
      const parsed = LFM3Parser.parseSingleCall(inner);
      if (parsed) calls.push({ ...parsed, raw: m[0] });
    }
    return calls;
  }

  private static parseSingleCall(inner: string): Omit<ParsedToolCall, 'raw'> | null {
    const fnMatch = inner.match(FUNCTION_REGEX);
    if (!fnMatch) return null;

    const functionName = fnMatch[1].trim();
    if (!KNOWN_FUNCTIONS.has(functionName)) {
      return null;
    }

    const argsRaw = fnMatch[2];
    const args: Record<string, string> = {};
    for (const pm of argsRaw.matchAll(PARAMETER_REGEX)) {
      args[pm[1].trim()] = pm[2].trim();
    }

    return { functionName, args };
  }
}