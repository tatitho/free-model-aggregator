# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/tatitho/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/tatitho/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, ten simulated free-tier providers, two client wires (OpenAI Chat
Completions + Anthropic Messages). Built for the make.fun
`free-model-aggregator-01` bounty.

## Run

```sh
node main.ts            # serves $MAKEFUN_ENV_URL, brief at $MAKEFUN_PROMPT_FILE
node main.ts --list     # prints the provider catalog as one JSON document, exit 0
```

Node 22+, standard library only — nothing is installed.

## Routing

Quality-first under a fixed plan computed before the first request, from the
workload manifest and the sealed per-provider budget ranges (never from a
trial's hidden draw):

- `gemini-3.8-flash` (1458) for the tools+vision / vision requests
- `openrouter/z-ai/glm-5.3-flash:free` (1475) for tools / JSON / min-context
- `zai` / `cloudflare` `glm-4.7-flash` (1470) for the min-context requests
  (first calls, one each — only five providers have >= 200k context)
- `kilol/kilo-auto/free` (1420) for the no-requirement requests

Failover is drop-on-first-refusal: a provider that answers 402 / 429 / 5xx
(or is unreachable) is dropped for the rest of the trial, never retried. The
ladder prefers providers the plan or a later min-context request still needs
last, unused providers before used ones, then the lowest expected cost of the
next call (failure risk from the sealed budget ranges + quality loss / 400).

All three upstream wires (OpenAI, Anthropic, Gemini) are implemented, with
byte-preserving pass-through of the system prompt, user text and images,
tools, tool results, stop sequences, `max_tokens` and JSON mode. No clock,
no randomness on any output path: the routing plan is a pure function of the
brief, the manifest and the refusal history, so a replay reproduces the same
answers.

## Layout

- `main.ts` — the server: brief, catalog probe, fixed plan, failover, register/poll
- `lib/providers.ts` — brief parsing, catalog probe, capability matching
- `lib/wire.ts` — client wires, Anthropic <-> OpenAI translation, SSE writers
- `lib/upstream.ts` — the three upstream wires and rendering back to the client

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=q1wt1zrm&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
