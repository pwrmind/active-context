export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class MaxIterationsError extends Error {
  constructor(max: number) {
    super(`Agent loop exceeded ${max} iterations without producing a final response`);
    this.name = 'MaxIterationsError';
  }
}

export class OllamaError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'OllamaError';
  }
}

export class ContextOverflowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ContextOverflowError';
  }
}