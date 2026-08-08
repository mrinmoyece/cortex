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
