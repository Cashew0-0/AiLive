// That's right! No imports and no dependencies 🤯

const OPENAI_EMBEDDING_DIMENSION = 1536;
const TOGETHER_EMBEDDING_DIMENSION = 768;
const OLLAMA_EMBEDDING_DIMENSION = 1024;
const GEMINI_EMBEDDING_DIMENSION = 3072;

export const EMBEDDING_DIMENSION: number = GEMINI_EMBEDDING_DIMENSION;

export function detectMismatchedLLMProvider() {
  switch (EMBEDDING_DIMENSION) {
    case OPENAI_EMBEDDING_DIMENSION:
      if (!process.env.OPENAI_API_KEY) {
        throw new Error(
          "Are you trying to use OpenAI? If so, run: npx convex env set OPENAI_API_KEY 'your-key'",
        );
      }
      break;
    case TOGETHER_EMBEDDING_DIMENSION:
      if (!process.env.TOGETHER_API_KEY) {
        throw new Error(
          "Are you trying to use Together.ai? If so, run: npx convex env set TOGETHER_API_KEY 'your-key'",
        );
      }
      break;
    case GEMINI_EMBEDDING_DIMENSION:
      if (!process.env.GEMINI_API_KEY) {
        throw new Error(
          "Are you trying to use Gemini? If so, run: npx convex env set GEMINI_API_KEY 'your-key'",
        );
      }
      break;
    case OLLAMA_EMBEDDING_DIMENSION:
      break;
    default:
      if (!process.env.LLM_API_URL) {
        throw new Error(
          "Are you trying to use a custom cloud-hosted LLM? If so, run: npx convex env set LLM_API_URL 'your-url'",
        );
      }
      break;
  }
}

export interface LLMConfig {
  provider: 'openai' | 'together' | 'gemini' | 'ollama' | 'custom';
  url: string; // Should not have a trailing slash
  // Paths appended to `url` for each kind of request. Most OpenAI-compatible
  // APIs use the /v1 prefix, but Gemini's compat layer doesn't, so this is
  // configurable per provider instead of hardcoded at the call site.
  chatPath: string;
  embeddingsPath: string;
  chatModel: string;
  embeddingModel: string;
  stopWords: string[];
  apiKey: string | undefined;
}

export function getLLMConfig(): LLMConfig {
  let provider = process.env.LLM_PROVIDER;
  if (provider ? provider === 'openai' : process.env.OPENAI_API_KEY) {
    if (EMBEDDING_DIMENSION !== OPENAI_EMBEDDING_DIMENSION) {
      throw new Error('EMBEDDING_DIMENSION must be 1536 for OpenAI');
    }
    return {
      provider: 'openai',
      url: 'https://api.openai.com',
      chatPath: '/v1/chat/completions',
      embeddingsPath: '/v1/embeddings',
      chatModel: process.env.OPENAI_CHAT_MODEL ?? 'gpt-4o-mini',
      embeddingModel: process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-ada-002',
      stopWords: [],
      apiKey: process.env.OPENAI_API_KEY,
    };
  }
  if (process.env.TOGETHER_API_KEY) {
    if (EMBEDDING_DIMENSION !== TOGETHER_EMBEDDING_DIMENSION) {
      throw new Error('EMBEDDING_DIMENSION must be 768 for Together.ai');
    }
    return {
      provider: 'together',
      url: 'https://api.together.xyz',
      chatPath: '/v1/chat/completions',
      embeddingsPath: '/v1/embeddings',
      chatModel: process.env.TOGETHER_CHAT_MODEL ?? 'meta-llama/Llama-3-8b-chat-hf',
      embeddingModel:
        process.env.TOGETHER_EMBEDDING_MODEL ?? 'togethercomputer/m2-bert-80M-8k-retrieval',
      stopWords: ['<|eot_id|>'],
      apiKey: process.env.TOGETHER_API_KEY,
    };
  }
  if (provider ? provider === 'gemini' : process.env.GEMINI_API_KEY) {
    if (EMBEDDING_DIMENSION !== GEMINI_EMBEDDING_DIMENSION) {
      throw new Error('EMBEDDING_DIMENSION must be 3072 for Gemini');
    }
    if (!process.env.GEMINI_API_KEY) {
      throw new Error("Set your Gemini key: npx convex env set GEMINI_API_KEY 'your-key'");
    }
    return {
      provider: 'gemini',
      // Gemini's OpenAI-compatible layer. No /v1 segment here — see chatPath
      // and embeddingsPath below instead.
      url: 'https://generativelanguage.googleapis.com/v1beta/openai',
      chatPath: '/chat/completions',
      embeddingsPath: '/embeddings',
      chatModel: process.env.GEMINI_CHAT_MODEL ?? 'gemini-2.0-flash',
      embeddingModel: process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-001',
      stopWords: [],
      apiKey: process.env.GEMINI_API_KEY,
    };
  }
  if (process.env.LLM_API_URL) {
    const apiKey = process.env.LLM_API_KEY;
    const url = process.env.LLM_API_URL;
    const chatModel = process.env.LLM_MODEL;
    if (!chatModel) throw new Error('LLM_MODEL is required');
    const embeddingModel = process.env.LLM_EMBEDDING_MODEL;
    if (!embeddingModel) throw new Error('LLM_EMBEDDING_MODEL is required');
    return {
      provider: 'custom',
      url,
      chatPath: '/v1/chat/completions',
      embeddingsPath: '/v1/embeddings',
      chatModel,
      embeddingModel,
      stopWords: [],
      apiKey,
    };
  }
  // Assume Ollama
  if (EMBEDDING_DIMENSION !== OLLAMA_EMBEDDING_DIMENSION) {
    detectMismatchedLLMProvider();
    throw new Error(
      `Unknown EMBEDDING_DIMENSION ${EMBEDDING_DIMENSION} found` +
        `. See convex/util/llm.ts for details.`,
    );
  }
  // Alternative embedding model:
  // embeddingModel: 'llama3'
  // const OLLAMA_EMBEDDING_DIMENSION = 4096,
  return {
    provider: 'ollama',
    url: process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434',
    chatPath: '/v1/chat/completions',
    embeddingsPath: '/v1/embeddings',
    chatModel: process.env.OLLAMA_MODEL ?? 'llama3',
    embeddingModel: process.env.OLLAMA_EMBEDDING_MODEL ?? 'mxbai-embed-large',
    stopWords: ['<|eot_id|>'],
    apiKey: undefined,
  };
}

const AuthHeaders = (): Record<string, string> =>
  getLLMConfig().apiKey
    ? {
        Authorization: 'Bearer ' + getLLMConfig().apiKey,
      }
    : {};

// Overload for non-streaming
export async function chatCompletion(
  body: Omit<CreateChatCompletionRequest, 'model'> & {
    model?: CreateChatCompletionRequest['model'];
  } & {
    stream?: false | null | undefined;
  },
): Promise<{ content: string; retries: number; ms: number }>;
// Overload for streaming
export async function chatCompletion(
  body: Omit<CreateChatCompletionRequest, 'model'> & {
    model?: CreateChatCompletionRequest['model'];
  } & {
    stream?: true;
  },
): Promise<{ content: ChatCompletionContent; retries: number; ms: number }>;
export async function chatCompletion(
  body: Omit<CreateChatCompletionRequest, 'model'> & {
    model?: CreateChatCompletionRequest['model'];
  },
) {
  const config = getLLMConfig();
  body.model = body.model ?? config.chatModel;
  const stopWords = body.stop ? (typeof body.stop === 'string' ? [body.stop] : body.stop) : [];
  if (config.stopWords) stopWords.push(...config.stopWords);
  console.log(body);
  const {
    result: content,
    retries,
    ms,
  } = await retryWithBackoff(async () => {
    const result = await fetch(config.url + config.chatPath, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...AuthHeaders(),
      },

      body: JSON.stringify(body),
    });
    if (!result.ok) {
      const error = await result.text();
      console.error({ error });
      if (result.status === 404 && config.provider === 'ollama') {
        await tryPullOllama(body.model!, error);
      }
      throw {
        retry: result.status === 429 || result.status >= 500,
        error: new Error(`Chat completion failed with code ${result.status}: ${error}`),
      };
    }
    if (body.stream) {
      return new ChatCompletionContent(result.body!, stopWords);
    } else {
      const json = (await result.json()) as CreateChatCompletionResponse;
      const content = json.choices[0].message?.content;
      if (content === undefined) {
        throw new Error('Unexpected result from OpenAI: ' + JSON.stringify(json));
      }
      console.log(content);
      return content;
    }
  });

  return {
    content,
    retries,
    ms,
  };
}

export async function tryPullOllama(model: string, error: string) {
  if (error.includes('try pulling')) {
    console.error('Embedding model not found, pulling from Ollama');
    const pullResp = await fetch(getLLMConfig().url + '/api/pull', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: model }),
    });
    console.log('Pull response', await pullResp.text());
    throw { retry: true, error: `Dynamically pulled model. Original error: ${error}` };
  }
}

export async function fetchEmbeddingBatch(texts: string[]) {
  const config = getLLMConfig();
  if (config.provider === 'ollama') {
    return {
      ollama: true as const,
      embeddings: await Promise.all(
        texts.map(async (t) => (await ollamaFetchEmbedding(t)).embedding),
      ),
    };
  }
  const {
    result: json,
    retries,
    ms,
  } = await retryWithBackoff(async () => {
    const result = await fetch(config.url + config.embeddingsPath, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...AuthHeaders(),
      },

      body: JSON.stringify({
        model: config.embeddingModel,
        input: texts.map((text) => text.replace(/\n/g, ' ')),
      }),
    });
    if (!result.ok) {
      throw {
        retry: result.status === 429 || result.status >= 500,
        error: new Error(`Embedding failed with code ${result.status}: ${await result.text()}`),
      };
    }
    return (await result.json()) as CreateEmbeddingResponse;
  });
  if (json.data.length !== texts.length) {
    console.error(json);
    throw new Error('Unexpected number of embeddings');
  }
  const allembeddings = json.data;
  allembeddings.sort((a, b) => a.index - b.index);
  return {
    ollama: false as const,
    embeddings: allembeddings.map(({ embedding }) => embedding),
    usage: json.usage?.total_tokens,
    retries,
    ms,
  };
}

export async function fetchEmbedding(text: string) {
  const { embeddings, ...stats } = await fetchEmbeddingBatch([text]);
  return { embedding: embeddings[0], ...stats };
}

export async function fetchModeration(content: string) {
  const { result: flagged } = await retryWithBackoff(async () => {
    const result = await fetch(getLLMConfig().url + '/v1/moderations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...AuthHeaders(),
      },

      body: JSON.stringify({
        input: content,
      }),
    });
    if (!result.ok) {
      throw {
        retry: result.status === 429 || result.status >= 500,
        error: new Error(`Embedding failed with code ${result.status}: ${await result.text()}`),
      };
    }
    return (await result.json()) as { results: { flagged: boolean }[] };
  });
  return flagged;
}

// Retry after this much time, based on the retry number.
const RETRY_BACKOFF = [1000, 10_000, 20_000]; // In ms
const RETRY_JITTER = 100; // In ms
type RetryError = { retry: boolean; error: any };

export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
): Promise<{ retries: number; result: T; ms: number }> {
  let i = 0;
  for (; i <= RETRY_BACKOFF.length; i++) {
    try {
      const start = Date.now();
      const result = await fn();
      const ms = Date.now() - start;
      return { result, retries: i, ms };
    } catch (e) {
      const retryError = e as RetryError;
      if (i < RETRY_BACKOFF.length) {
        if (retryError.retry) {
          console.log(
            `Attempt ${i + 1} failed, waiting ${RETRY_BACKOFF[i]}ms to retry...`,
            Date.now(),
          );
          await new Promise((resolve) =>
            setTimeout(resolve, RETRY_BACKOFF[i] + RETRY_JITTER * Math.random()),
          );
          continue;
        }
      }
      if (retryError.error) throw retryError.error;
      else throw e;
    }
  }
  throw new Error('Unreachable');
}

// Lifted from openai's package
export interface LLMMessage {
  /**
   * The contents of the message. `content` is required for all messages, and may be
   * null for assistant messages with function calls.
   */
  content: string | null;

  /**
   * The role of the messages author. One of `system`, `user`, `assistant`, or
   * `function`.
   */
  role: 'system' | 'user' | 'assistant' | 'function';

  /**
   * The name of the author of this message. `name` is required if role is
   * `function`, and it should be the name of the function whose response is in the
   * `content`. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of
   * 64 characters.
   */
  name?: string;

  /**
   * The name and arguments of a function that should be called, as generated by the model.
   */
  function_call?: {
    // The name of the function to call.
    name: string;
    /**
     * The arguments to call the function with, as generated by the model in
     * JSON format. Note that the model does not always generate valid JSON,
     * and may hallucinate parameters not defined by your function schema.
     * Validate the arguments in your code before calling your function.
     */
    arguments: string;
  };
}

// Non-streaming chat completion response
interface CreateChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index?: number;
    message?: {
      role: 'system' | 'user' | 'assistant';
      content: string;
    };
    finish_reason?: string;
  }[];
  usage?: {
    completion_tokens: number;

    prompt_tokens: number;

    total_tokens: number;
  };
}

interface CreateEmbeddingResponse {
  data: {
    index: number;
    object: string;
    embedding: number[];
  }[];
  model: string;
  object: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface CreateChatCompletionRequest {
  /**
   * ID of the model to use.
   * @type {string}
   * @memberof CreateChatCompletionRequest
   */
  model: string;
  /**
   * The messages to generate chat completions for, in the chat format:
   * https://platform.openai.com/docs/guides/chat/introduction
   * @type {Array<ChatCompletionRequestMessage>}
   * @memberof CreateChatCompletionRequest
   */
  messages: LLMMessage[];
  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both.
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  temperature?: number | null;
  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both.
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  top_p?: number | null;
  /**
   * How many chat completion choices to generate for each input message.
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  n?: number | null;
  /**
   * If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
   * @type {boolean}
   * @memberof CreateChatCompletionRequest
   */
  stream?: boolean | null;
  /**
   *
   * @type {CreateChatCompletionRequestStop}
   * @memberof CreateChatCompletionRequest
   */
  stop?: Array<string> | string;
  /**
   * The maximum number of tokens allowed for the generated answer. By default,
   * the number of tokens the model can return will be (4096 - prompt tokens).
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  max_tokens?: number;
  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on
   * whether they appear in the text so far, increasing the model\'s likelihood
   * to talk about new topics. See more information about frequency and
   * presence penalties:
   * https://platform.openai.com/docs/api-reference/parameter-details
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  presence_penalty?: number | null;
  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on
   * their existing frequency in the text so far, decreasing the model\'s
   * likelihood to repeat the same line verbatim. See more information about
   * presence penalties:
   * https://platform.openai.com/docs/api-reference/parameter-details
   * @type {number}
   * @memberof CreateChatCompletionRequest
   */
  frequency_penalty?: number | null;
  /**
   * Modify the likelihood of specified tokens appearing in the completion.
   * Accepts a json object that maps tokens (specified by their token ID in the
   * tokenizer) to an associated bias value from -100 to 100. Mathematically,
   * the bias is added to the logits generated by the model prior to sampling.
   * The exact effect will vary per model, but values between -1 and 1 should
   * decrease or increase likelihood of selection; values like -100 or 100
   * should result in a ban or exclusive selection of the relevant token.
   * @type {object}
   * @memberof CreateChatCompletionRequest
   */
  logit_bias?: object | null;
  /**
   * A unique identifier representing your end-user, which can help OpenAI to
   * monitor and detect abuse. Learn more:
   * https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids
   * @type {string}
   * @memberof CreateChatCompletionRequest
   */
  user?: string;
  tools?: {
    type: 'function';
    function: {
      name: string;
      description?: string;
      parameters: object;
    };
  }[];
  tool_choice?:
    | 'none'
    | 'auto'
    | {
        type: 'function';
        function: { name: string };
      };
  response_format?: { type: 'text' | 'json_object' };
}

// Checks whether a suffix of s1 is a prefix of s2. For example,
// ('Hello', 'Kira:') -> false
// ('Hello Kira', 'Kira:') -> true
const suffixOverlapsPrefix = (s1: string, s2: string) => {
  for (let i = 1; i <= Math.min(s1.length, s2.length); i++) {
    const suffix = s1.substring(s1.length - i);
    const prefix = s2.substring(0, i);
    if (suffix === prefix) {
      return true;
    }
  }
  return false;
};

export class ChatCompletionContent {
  private readonly body: ReadableStream<Uint8Array>;
  private readonly stopWords: string[];

  constructor(body: ReadableStream<Uint8Array>, stopWords: string[]) {
    this.body = body;
    this.stopWords = stopWords;
  }

  async *readInner() {
    for await (const data of this.splitStream(this.body)) {
      if (data.startsWith('data: ')) {
        try {
          const json = JSON.parse(data.substring('data: '.length)) as {
            choices: { delta: { content?: string } }[];
          };
          if (json.choices[0].delta.content) {
            yield json.choices[0].delta.content;
          }
        } catch (e) {
          // e.g. the last chunk is [DONE] which is not valid JSON.
        }
      }
    }
  }

  // stop words in OpenAI api don't always work.
  // So we have to truncate on our side.
  async *read() {
    let lastFragment = '';
    for await (const data of this.readInner()) {
      lastFragment += data;
      let hasOverlap = false;
      for (const stopWord of this.stopWords) {
        const idx = lastFragment.indexOf(stopWord);
        if (idx >= 0) {
          yield lastFragment.substring(0, idx);
          return;
        }
        if (suffixOverlapsPrefix(lastFragment, stopWord)) {
          hasOverlap = true;
        }
      }
      if (hasOverlap) continue;
      yield lastFragment;
      lastFragment = '';
    }
    yield lastFragment;
  }

  async readAll() {
    let allContent = '';
    for await (const chunk of this.read()) {
      allContent += chunk;
    }
    return allContent;
  }

  async *splitStream(stream: ReadableStream<Uint8Array>) {
    const reader = stream.getReader();
    let lastFragment = '';
    try {
      while (true) {
        const { value, done } = await reader.
