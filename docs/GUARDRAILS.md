# Guardrails & Safety

Cortex applies safety checks at three points: input validation, LLM call wrapping via NeMo Guardrails, and output sanitisation.

## Three-Layer Safety Model

```
User Input
    │
    ▼
[1] InjectionDetector — regex pattern matching
    │ blocked? → raise PromptInjectionError (HTTP 422)
    ▼
[2] PIIScanner — Presidio entity detection
    │ found? → redact and continue (log warning)
    ▼
[3] NeMo Guardrails — policy engine (topic scope, harmful content, jailbreak)
    │ violated? → return policy message instead of LLM response
    ▼
LLM Call
    │
    ▼
[4] Output PIIScanner — redact before returning to user
```

## Injection Detection

Blocks patterns that attempt to override agent behaviour:

- `ignore (all) previous instructions` — classic jailbreak opener
- `you are now DAN / uncensored` — role-break attempt
- `print/reveal your system prompt` — exfiltration attempt
- `[INST] ... [/INST]`, `<|system|>` — model-specific injection tokens

**Response:** HTTP 422 with code `PROMPT_INJECTION_DETECTED`. Run is not created.

## PII Detection

Uses [Microsoft Presidio](https://microsoft.github.io/presidio/) for entity recognition. Detects:
- Email addresses, phone numbers (UK + US)
- UK National Insurance numbers
- US Social Security Numbers
- Credit card numbers

On detection in **input**: redact and continue (don't block — users may legitimately reference their own data).  
On detection in **output**: always redact before returning to user.

**Custom entities:** Add patterns to `PIIScanner._PATTERNS` for domain-specific PII (employee IDs, case numbers, etc.).

## NeMo Guardrails

Policy engine defined in Colang (human-readable policy language):

`config/rails/input.co` — input policies:
- Block harmful/illegal requests
- Block jailbreak attempts
- Block requests for raw PII

`config/rails/output.co` — output policies:
- Flag low-confidence answers
- Prevent PII leakage in responses

**Adding a policy:**
```colang
define user ask competitor comparison
    "how does Cortex compare to LangSmith"
    "is Cortex better than Langfuse"

define bot decline competitor question
    "I focus on Cortex capabilities. For competitive comparisons, please consult independent reviews."

define flow handle competitor question
    user ask competitor comparison
    bot decline competitor question
```

## Safety Metrics

Monitor in Grafana (Safety panel):
- `cortex_safety_violations_total{violation_type="prompt_injection"}` — active attack signal
- `cortex_safety_violations_total{violation_type="pii_input"}` — users sending sensitive data
- `cortex_safety_violations_total{violation_type="pii_output"}` — potential leakage caught

A spike in `prompt_injection` during a short window indicates a targeted attack. Consider rate-limiting or blocking that user/IP.

## Disabling Safety (Development Only)

```env
GUARDRAILS_ENABLED=false
PII_DETECTION_ENABLED=false
INJECTION_DETECTION_ENABLED=false
```

Never disable safety in production.


## The layers, and what each is actually worth

```
input   →  scored jailbreak detection  (local, µs)  →  [classifier, optional]
        →  PII redaction               (Presidio)
        →  NeMo Colang rails           (optional)

tool    →  spotlighting                (local, µs)   ← indirect injection
output  →  harm pattern matching       (local, µs)  →  [classifier, optional]
        →  PII redaction               (Presidio)
```

**Scored, not matched.** Jailbreak detection assigns weights to signals and
sums them. A single strong signal (instruction override, named jailbreak,
injected control token) blocks alone; a single weak one (persona swap) does
not, because refusing "pretend you are a pirate" is how a safety layer gets
switched off. Two weak signals together block.

The weights were guessed first, and the corpus caught it: a bare "ignore
all previous instructions" scored 0.45 against a 0.50 threshold and was
**allowed**. Nothing else in that prompt was suspicious, so no second signal
arrived. Anything strong enough to block on its own must exceed the
threshold on its own — obvious in hindsight, invisible without the test.

**Two normalisations, because neither is sufficient.** A zero-width space
*inside* a word (`Ig<ZWSP>nore`) is defeated by deleting invisibles; a bidi
override *between* words (`ignore<RLO>previous`) is defeated by replacing
them with spaces. Each fix breaks the other case, so both readings are
scored and the higher wins. An attacker has to beat every normalisation
rather than find the one that was skipped.

**Spotlighting for indirect injection.** Tool results and prior task
outputs are fenced and labelled as data before entering the prompt. This is
the attack that matters for a RAG agent: direct injection needs a hostile
user, indirect injection needs one poisoned document in a corpus the user
trusts — and the agent reads it with the user's privileges. Spotlighting
does not make injection impossible; it makes the boundary explicit, which
is the difference between an attack needing persuasion and needing nothing.

**The classifier layer fails CLOSED.** The semantic cache fails open,
because a cache is an optimisation. A moderator is a control, and quietly
serving unmoderated output because a dependency blipped is exactly the
incident the layer exists to prevent.

## Honest scope

- Regex and scoring beat obvious and lightly-paraphrased attacks. A
  determined attacker beats them. The classifier seam exists for that, and
  is unconfigured by default.
- 13 attack strings and 10 benign controls in `tests/test_safety/`. Both
  directions are tested, because tuning only for recall produces a layer
  with false positives that somebody disables within a week.
- No output moderation existed at all before this. Input was filtered and
  PII redacted; nothing examined what the model said.
