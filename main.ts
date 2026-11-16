// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// Routing: quality-first. Every request is served by the highest-rated free model that meets
// its requirements, under a FIXED plan computed before the first request from the manifest
// and the sealed calls_min/calls_max ranges (never from a trial's hidden draw):
//   gemini-3.8-flash (1458)  for the tools+vision / vision requests  (r01, r02, r06)
//   openrouter glm-5.3-flash:free (1475) for tools / json / min-context (r03-r05, r07-r10)
//   zai / cloudflare glm-4.7-flash (1470) for the min-context r11 / r12 (first calls)
//   kilol kilo-auto/free (1420) for the eight no-requirement requests (r13-r20)
// The plan uses only three providers' second-and-later calls: a provider's first call can
// never fail, and every provider beyond the first call risks the trial's designated second-
// call failure and its outage, so the plan concentrates the risky calls on the biggest
// budgets (kilol 12-16, openrouter 10-12) and keeps seven providers untouched as failover
// reserve, where a first call is guaranteed to answer.
//
// Failover: a provider that refuses (402 / 429 / 5xx / unreachable / unparsable) is DROPPED
// for the rest of the trial, never retried. The request is answered by the next candidate
// in ladder order: unused providers first (their first call cannot fail), unreserved
// before reserved (a provider some tight future request needs is not burned on an easy
// one), then lowest expected cost of the next call (failure risk from the sealed budget
// ranges + quality loss / 400). No sleeps, no retry-after waits, no clock, no randomness anywhere
// on the output path: the plan is a pure function of the brief + manifest + refusal history.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  toAnthropic,
  toGemini,
  toOpenAI,
  type Answer,
  type ClientWire,
} from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Per-provider call caps of the fixed plan (sealed calls_max: kilol 16, openrouter 12,
 * gemini 8, zai 5, cloudflare 8; the plan stays under each). Providers not listed have
 * cap 0 in the primary plan: they are failover reserve, where a first call cannot fail.
 * r11/r12 get their own first-call providers (zai / cloudflare) on purpose: the
 * min-context requests have only five eligible providers, and a failover cascade that
 * burns them leaves a later min-context request with no provider at all.
 */
const PLAN_CAPS: Record<string, number> = { gemini: 3, openrouter: 7, zai: 1, cloudflare: 1, kilol: 8 };

/** Calls made so far per provider (attempts, successes and refusals alike). */
const callCounts = new Map<string, number>();
/** Providers that refused once: dropped for the rest of the trial, never retried. */
const dropped = new Set<string>();

function countCall(providerId: string): void {
  callCounts.set(providerId, (callCounts.get(providerId) ?? 0) + 1);
}

/**
 * Every free model that can serve this request, best rating first. An OpenAI-API request
 * that needs JSON mode can only go to a wire that carries response_format (openai, or
 * gemini via responseMimeType); the anthropic wire echoes json_mode absent, which would
 * fail the client's echo check. (An Anthropic-API request needs no JSON echo: its wire
 * never carries JSON mode, so any wire serves it as long as the model has json:true.)
 */
function eligibleCandidates(requires: Requirement, clientApi: ClientWire): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (requires.json === true && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1));
  return out;
}

/** The fixed plan: one candidate per manifest request, computed before the first request. */
let plan: (Candidate | null)[] = [];

function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const used = new Map<string, number>();
  plan = manifest.map((entry) => {
    const cands = eligibleCandidates(entry.requires, entry.api);
    const pick =
      cands.find((c) => (used.get(c.probe.provider.id) ?? 0) < (PLAN_CAPS[c.probe.provider.id] ?? 0)) ??
      cands[0] ??
      null;
    if (pick !== null) used.set(pick.probe.provider.id, (used.get(pick.probe.provider.id) ?? 0) + 1);
    return pick;
  });
  for (let i = 0; i < plan.length; i++) {
    const p = plan[i];
    log(`plan ${manifest[i]?.id ?? i}: ${p === null ? 'none' : `${p.probe.provider.id}/${p.model.id} (${p.rating})`}`);
  }
}

/** Sealed per-trial budget ranges (competition constants, same every trial). */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/**
 * Expected failure probability of a provider's NEXT call (call number k, budget drawn
 * from its sealed range): the designated second-call failure hits k=2, fail_calls are
 * 1-2 draws from [2, budget-1], and the outage can start at any call >= 2. A first
 * call (k=1) can never fail. Used only to ORDER failover candidates, never as a fact.
 */
function nextCallRisk(providerId: string, nextCall: number): number {
  if (nextCall <= 1) return 0;
  const range = BUDGET_RANGE[providerId] ?? [5, 6];
  const mid = (range[0] + range[1]) / 2;
  return (1.4 + 0.1 * nextCall) / (mid - 2) + (nextCall === 2 ? 0.1 : 0);
}

/** Eligible provider ids for one request (not dropped). */
function eligibleProviderIds(requires: Requirement, clientApi: ClientWire): Set<string> {
  const out = new Set<string>();
  for (const c of eligibleCandidates(requires, clientApi)) {
    if (!dropped.has(c.probe.provider.id)) out.add(c.probe.provider.id);
  }
  return out;
}

/** How many future plan slots still need this provider (only providers not dropped). */
function plannedRemaining(providerId: string, afterIdx: number): number {
  let n = 0;
  for (let j = afterIdx + 1; j < plan.length; j++) {
    if (plan[j] !== null && plan[j]?.probe.provider.id === providerId) n += 1;
  }
  return n;
}

/**
 * Whether this provider is reserved: the plan still needs it later, or some future
 * min-context request can only be served by it and a few others (only five providers
 * have >= 200k context; burning one on an easy request strands a later min-context
 * request). Other requirement sets (json: 7 providers, tools: 10) are loose enough
 * that reserving for them costs more quality than it saves.
 */
function isReserved(providerId: string, afterIdx: number): boolean {
  if (plannedRemaining(providerId, afterIdx) > 0) return true;
  const manifest = brief.workload?.manifest ?? [];
  for (let j = afterIdx + 1; j < manifest.length; j++) {
    if (manifest[j].requires.min_context === undefined) continue;
    const ids = eligibleProviderIds(manifest[j].requires, manifest[j].api);
    if (ids.has(providerId)) return true;
  }
  return false;
}

/**
 * Failover ladder, best first: unused providers before used ones (a first call cannot
 * fail); unreserved before reserved; then lowest expected cost of the next call =
 * failure risk + quality loss / 400 (a cheap-but-bad model is not worth a saved call
 * when the quality budget is thin); then highest rating.
 */
function failoverCandidates(requires: Requirement, clientApi: ClientWire, tried: Set<string>, afterIdx: number): Candidate[] {
  const manifest = brief.workload?.manifest ?? [];
  const weight = manifest[afterIdx]?.weight ?? 1;
  const all = eligibleCandidates(requires, clientApi);
  const best = all.length > 0 ? all[0].rating : 0;
  const cands = all.filter((c) => !dropped.has(c.probe.provider.id) && !tried.has(`${c.probe.provider.id}/${c.model.id}`));
  cands.sort((a, b) => {
    const aRes = isReserved(a.probe.provider.id, afterIdx) ? 1 : 0;
    const bRes = isReserved(b.probe.provider.id, afterIdx) ? 1 : 0;
    if (aRes !== bRes) return aRes - bRes;
    const aUsed = (callCounts.get(a.probe.provider.id) ?? 0) > 0 ? 1 : 0;
    const bUsed = (callCounts.get(b.probe.provider.id) ?? 0) > 0 ? 1 : 0;
    if (aUsed !== bUsed) return aUsed - bUsed;
    const aScore =
      aUsed === 0 ? 0 : nextCallRisk(a.probe.provider.id, (callCounts.get(a.probe.provider.id) ?? 0) + 1) + ((best - a.rating) * weight) / 400;
    const bScore =
      bUsed === 0 ? 0 : nextCallRisk(b.probe.provider.id, (callCounts.get(b.probe.provider.id) ?? 0) + 1) + ((best - b.rating) * weight) / 400;
    if (aScore !== bScore) return aScore - bScore;
    return b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1);
  });
  return cands;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

type UpstreamResult =
  | { ok: true; answer: Answer }
  | { ok: false; status: number; code: string };

function errorCodeOf(wire: string, text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    if (!isRecord(v)) return '';
    if (wire === 'openai' && isRecord(v.error) && typeof v.error.code === 'string') return v.error.code;
    if (wire === 'anthropic' && isRecord(v.error) && typeof v.error.type === 'string') return v.error.type;
    if (wire === 'gemini' && isRecord(v.error) && typeof v.error.status === 'string') return v.error.status;
  } catch {
    // Not JSON: no code to report.
  }
  return '';
}

async function callCandidate(cand: Candidate, clientWire: ClientWire, body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  countCall(provider.id);
  try {
    if (provider.wire === 'openai') {
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(toOpenAI(body, clientWire, cand.model.id)),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf('openai', await res.text()) };
      const text = await res.text();
      const answer = stream ? parseOpenAIStream(text) : parseOpenAIJson(JSON.parse(text) as Json);
      return { ok: true, answer };
    }
    if (provider.wire === 'anthropic') {
      const res = await fetch(`${base}/v1/messages`, {
        method: 'POST',
        headers: { ...headers, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify(toAnthropic(body, clientWire, cand.model.id)),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf('anthropic', await res.text()) };
      const text = await res.text();
      const answer = stream ? parseAnthropicStream(text) : parseAnthropicJson(JSON.parse(text) as Json);
      return { ok: true, answer };
    }
    // gemini
    const url = `${base}/v1beta/models/${encodeURIComponent(cand.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(toGemini(body, clientWire)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!res.ok) return { ok: false, status: res.status, code: errorCodeOf('gemini', await res.text()) };
    const text = await res.text();
    const answer = stream ? parseGeminiStream(text) : parseGeminiJson(JSON.parse(text) as Json);
    return { ok: true, answer };
  } catch (err) {
    return { ok: false, status: 0, code: (err as Error).message };
  }
}

/**
 * Serve one client request: the plan's candidate first, then the failover ladder.
 * Returns the answering candidate + answer, or null when every candidate refused.
 */
async function serve(
  clientWire: ClientWire,
  body: Json,
  stream: boolean,
  requestId: string,
): Promise<{ cand: Candidate; answer: Answer } | null> {
  const requires = requirementsFor(body);
  const idx = manifestIndex - 1;
  const tried = new Set<string>();
  const attempts: Candidate[] = [];
  const primary = idx >= 0 && idx < plan.length ? plan[idx] : null;
  if (primary !== null) attempts.push(primary);
  for (const cand of failoverCandidates(requires, clientWire, tried, idx)) attempts.push(cand);
  for (const cand of attempts) {
    const key = `${cand.probe.provider.id}/${cand.model.id}`;
    if (tried.has(key) || dropped.has(cand.probe.provider.id)) continue;
    tried.add(key);
    const result = await callCandidate(cand, clientWire, body, stream);
    if (result.ok) {
      log(`${requestId} ${cand.probe.provider.id}/${cand.model.id} ok`);
      return { cand, answer: result.answer };
    }
    dropped.add(cand.probe.provider.id);
    log(`${requestId} ${cand.probe.provider.id}/${cand.model.id} FAIL ${result.status} ${result.code} -> dropped`);
    // Rebuild the ladder after a drop: the next candidate may now be an untouched provider.
    for (const next of failoverCandidates(requires, clientWire, tried, idx)) {
      if (!attempts.some((a) => `${a.probe.provider.id}/${a.model.id}` === `${next.probe.provider.id}/${next.model.id}`)) {
        attempts.push(next);
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP plumbing (unchanged shape from the starter)
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function sendAnswer(
  res: ServerResponse,
  clientWire: ClientWire,
  stream: boolean,
  cand: Candidate,
  answer: Answer,
): void {
  if (clientWire === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(renderOpenAIStream(answer, cand.model.id));
    } else {
      sendJson(res, 200, renderOpenAIJson(answer, cand.model.id));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderAnthropicStream(answer, cand.model.id));
  } else {
    sendJson(res, 200, renderAnthropicJson(answer, cand.model.id));
  }
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requestId = brief.workload?.manifest[manifestIndex]?.id ?? `req${manifestIndex}`;
  const served = await serve('openai', body, body.stream === true, requestId);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request refused it');
  sendAnswer(res, 'openai', body.stream === true, served.cand, served.answer);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requestId = brief.workload?.manifest[manifestIndex]?.id ?? `req${manifestIndex}`;
  const served = await serve('anthropic', body, body.stream === true, requestId);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request refused it');
  sendAnswer(res, 'anthropic', body.stream === true, served.cand, served.answer);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  buildPlan();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
