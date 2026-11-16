// Upstream wires (openai / anthropic / gemini) and rendering back to the client's wire.
//
// Every upstream answer is first normalized into an Answer {text, toolCalls, stop, usage}
// and then rendered in the wire the client used. The renderers are pure functions of the
// parsed answer: no clock, no randomness. Text is carried whole; usage and stop reason are
// the provider's own numbers, mapped only between the wires' spellings.

import { isRecord, type Json } from './providers.ts';
import { anthropicToOpenAI } from './wire.ts';

export type ClientWire = 'openai' | 'anthropic';
export type UpstreamWire = 'openai' | 'anthropic' | 'gemini';

export interface Usage {
  input: number;
  output: number;
}

export interface ToolCall {
  id: string;
  name: string;
  /** JSON string of the arguments object, as the OpenAI wire carries it. */
  argumentsJson: string;
}

export interface Answer {
  text: string;
  toolCalls: ToolCall[];
  /** The provider's stop token verbatim: "stop" | "end_turn" | "STOP" | ... */
  stop: string;
  usage: Usage;
}

const FIXED_CREATED = 1767225600;

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function argsJson(v: unknown): string {
  if (typeof v === 'string') return v;
  if (isRecord(v)) return JSON.stringify(v);
  return '{}';
}

function argsObject(v: unknown): Json {
  if (isRecord(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(v);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Split text into 13-char pieces, the way the simulated providers chunk their streams. */
function splitText(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [''] : out;
}

// ---------------------------------------------------------------------------
// Stop reasons: one family per wire spelling; the provider always answers
// family 0 ("stop" / "end_turn" / "STOP"), but map every family for safety.
// ---------------------------------------------------------------------------

export function openAIFinish(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

export function anthropicStop(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request builders: a client body rewritten for one upstream wire, model set.
// ---------------------------------------------------------------------------

/** A client body as an OpenAI Chat Completions request naming `model`. */
export function toOpenAI(body: Json, clientWire: ClientWire, model: string): Json {
  if (clientWire === 'openai') {
    const out: Json = { ...body, model };
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  return anthropicToOpenAI(body, model);
}

function dataUrlParts(url: string): { mimeType: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mimeType: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** An OpenAI message list as Gemini contents (+ the system text, separately). */
function openAIToGeminiParts(body: Json): { contents: Json[]; systemText: string } {
  const contents: Json[] = [];
  let systemText = '';
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemText = systemText === '' ? t : `${systemText}\n${t}`;
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const data = dataUrlParts(block.image_url.url);
          if (data !== null) parts.push({ inlineData: { mimeType: data.mimeType, data: data.data } });
          else parts.push({ fileData: { fileUri: block.image_url.url } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  return { contents, systemText };
}

/** An Anthropic message list as Gemini contents (+ the system text, separately). */
function anthropicToGeminiParts(body: Json): { contents: Json[]; systemText: string } {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        }
      }
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  return { contents, systemText };
}

function geminiToolsFromOpenAI(body: Json): Json[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const decls: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    decls.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

function geminiToolsFromAnthropic(body: Json): Json[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const decls: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    decls.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

/** A client body as a Gemini generateContent request body. */
export function toGemini(body: Json, clientWire: ClientWire): Json {
  const { contents, systemText } = clientWire === 'openai' ? openAIToGeminiParts(body) : anthropicToGeminiParts(body);
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = clientWire === 'openai' ? geminiToolsFromOpenAI(body) : geminiToolsFromAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops: string[] = [];
  if (clientWire === 'openai') {
    if (typeof body.stop === 'string') stops.push(body.stop);
    else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
      config.responseMimeType = 'application/json';
    }
  } else if (Array.isArray(body.stop_sequences)) {
    for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  }
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions body as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    // user
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const data = dataUrlParts(block.image_url.url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data.mimeType, data: data.data } });
        else blocks.push({ type: 'image', source: { type: 'url', url: block.image_url.url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A client body as an Anthropic Messages request naming `model`. */
export function toAnthropic(body: Json, clientWire: ClientWire, model: string): Json {
  if (clientWire === 'anthropic') return { ...body, model };
  return openAIToAnthropicRequest(body, model);
}

// ---------------------------------------------------------------------------
// SSE frame splitting (shared by the three stream parsers).
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const chunk of normalized.split('\n\n')) {
    if (chunk.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(payload: string): Json | null {
  try {
    const v: unknown = JSON.parse(payload);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Response parsers: one upstream wire's answer -> Answer.
// ---------------------------------------------------------------------------

export function parseOpenAIJson(body: Json): Answer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : 'call_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      argumentsJson: argsJson(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    toolCalls,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    usage: { input: num(usage.prompt_tokens), output: num(usage.completion_tokens) },
  };
}

export function parseOpenAIStream(text: string): Answer {
  let out = '';
  let stop = 'stop';
  let usage: Usage = { input: 0, output: 0 };
  const toolCalls: ToolCall[] = [];
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: num(chunk.usage.prompt_tokens), output: num(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, toolCalls, stop, usage };
}

export function parseAnthropicJson(body: Json): Answer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        name: typeof block.name === 'string' ? block.name : '',
        argumentsJson: argsJson(block.input),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    usage: { input: num(usage.input_tokens), output: num(usage.output_tokens) },
  };
}

export function parseAnthropicStream(text: string): Answer {
  let out = '';
  let stop = 'end_turn';
  let usage: Usage = { input: 0, output: 0 };
  const toolCalls = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSse(text)) {
    if (frame.event === null) continue;
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      usage = { input: num(data.message.usage.input_tokens), output: usage.output };
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = typeof data.index === 'number' ? data.index : -1;
      if (data.content_block.type === 'tool_use') {
        toolCalls.set(index, {
          id: typeof data.content_block.id === 'string' ? data.content_block.id : 'toolu_unknown',
          name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
          args: '',
        });
      }
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : -1;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const slot = toolCalls.get(index);
        if (slot !== undefined) slot.args += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) usage = { input: usage.input, output: num(data.usage.output_tokens) };
    }
  }
  return {
    text: out,
    toolCalls: [...toolCalls.values()].map((t) => ({ id: t.id, name: t.name, argumentsJson: t.args === '' ? '{}' : t.args })),
    stop,
    usage,
  };
}

export function parseGeminiJson(body: Json): Answer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls: [],
    stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
    usage: { input: num(meta.promptTokenCount), output: num(meta.candidatesTokenCount) },
  };
}

export function parseGeminiStream(text: string): Answer {
  let out = '';
  let stop = 'STOP';
  let usage: Usage = { input: 0, output: 0 };
  for (const frame of parseSse(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(cand.content) ? cand.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
    if (typeof cand.finishReason === 'string') stop = cand.finishReason;
    if (isRecord(data.usageMetadata)) {
      usage = { input: num(data.usageMetadata.promptTokenCount), output: num(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, toolCalls: [], stop, usage };
}

// ---------------------------------------------------------------------------
// Renderers: Answer -> the client's wire, JSON or SSE.
// ---------------------------------------------------------------------------

export function renderOpenAIJson(ans: Answer, model: string): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.argumentsJson } }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(ans.stop), logprobs: null }],
    usage: { prompt_tokens: ans.usage.input, completion_tokens: ans.usage.output, total_tokens: ans.usage.input + ans.usage.output },
  };
}

function sseLine(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export function renderOpenAIStream(ans: Answer, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseLine(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(ans.text)) frames.push(sseLine(null, chunk({ content: piece }, null)));
  // Tool calls, if any, ride as deltas after the text (the provider never emits them here).
  for (const call of ans.toolCalls) {
    frames.push(
      sseLine(
        null,
        chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.argumentsJson } }] }, null),
      ),
    );
  }
  frames.push(sseLine(null, chunk({}, openAIFinish(ans.stop))));
  frames.push(
    sseLine(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: ans.usage.input, completion_tokens: ans.usage.output, total_tokens: ans.usage.input + ans.usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function renderAnthropicJson(ans: Answer, model: string): Json {
  const content: Json[] = [];
  if (ans.text !== '') content.push({ type: 'text', text: ans.text });
  for (const call of ans.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.argumentsJson) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(ans.stop),
    stop_sequence: null,
    usage: { input_tokens: ans.usage.input, output_tokens: ans.usage.output },
  };
}

export function renderAnthropicStream(ans: Answer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseLine('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.usage.input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (ans.text !== '' || ans.toolCalls.length === 0) {
    frames.push(sseLine('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(ans.text)) {
      frames.push(sseLine('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseLine('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of ans.toolCalls) {
    frames.push(
      sseLine('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    for (const piece of splitText(call.argumentsJson)) {
      frames.push(sseLine('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: piece } }));
    }
    frames.push(sseLine('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseLine('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(ans.stop), stop_sequence: null },
      usage: { output_tokens: ans.usage.output },
    }),
  );
  frames.push(sseLine('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
