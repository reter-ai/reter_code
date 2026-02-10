# OpenClaw Security Audit - CADSL Detectors (v2)

Automated security detectors for the OpenClaw multi-channel AI agent platform.
These detectors target TypeScript-specific attack surfaces: WebSocket/HTTP gateways,
channel webhook integrations, shell execution, auth/credential management, and
external content processing.

**Audit re-run:** 2026-02-10 | **RETER version:** 0.1.0 | **Session:** 2FC7

## Detection Patterns

| # | Detector | CWE | Severity | Target | Findings | TP Rate |
|---|----------|-----|----------|--------|----------|---------|
| 1 | `complex_gateway_handlers` | CWE-20 | Critical | Gateway server functions handling WebSocket/HTTP input | 22 | 9% |
| 2 | `webhook_verification_bypass` | CWE-347 | Critical | Webhook signature verification with bypass risk | 7 | 86% |
| 3 | `command_injection_surface` | CWE-78 | High | Shell execution and process spawning functions | 7 | 29% |
| 4 | `complex_input_parsers` | CWE-502/CWE-20 | High | Complex parsing/deserialization of external input | 50 | 4% |
| 5 | `auth_handler_complexity` | CWE-287 | High | Complex auth/credential handling functions | 50 | 6% |
| 6 | `secret_in_logging` | CWE-532 | Medium | Secret redaction/masking in logging code | 7 | 0% |
| 7 | `external_content_handlers` | CWE-79/CWE-94 | Medium | External content processing (prompt injection, XSS) | 50 | 4% |
| 8 | `path_traversal_surface` | CWE-22 | Medium | File path resolution from external input | 8 | 25% |

**Total: 201 findings, 10 TP, 7 PARTIAL-TP, 184 FP (8% overall TP rate)**

### Changes from v1 Audit

| Metric | v1 | v2 | Delta |
|--------|----|----|-------|
| Total findings | 201 | 201 | = |
| True Positives | 21 TP + PARTIAL | 10 TP + 7 PARTIAL | -4 net |
| New findings | — | 4 | +4 |
| Downgraded to FP | — | 4 | -4 |
| Upgraded severity | — | 3 | +3 |

**New findings (v2):** `validateTwilioSignature`, `validatePlivoV2Signature`, `validatePlivoV3Signature`, `resolvePath` (backend-config)

**Downgraded to FP:** `parseMessageWithAttachments` (FP-STRUCTURAL: well-defended), `routeReply` (FP-INTERFACE: safe wrapper), `resolveFilePath` downgraded to low, `authorizeGatewayConnect` downgraded from TP-EXTRACT to PARTIAL-TP

## Classified Findings

### All Verified True Positives (RETER 2FC7-TASK-001 through 017)

| RETER ID | Function | File:Line | CWE | Classification | Priority |
|----------|----------|-----------|-----|---------------|----------|
| TASK-001 | `verifyTwilioWebhook` | `extensions/voice-call/src/webhook-security.ts:336` | CWE-347 | **TP-PARAMETERIZE** | Critical |
| TASK-002 | `verifyPlivoWebhook` | `extensions/voice-call/src/webhook-security.ts:578` | CWE-347 | **TP-PARAMETERIZE** | Critical |
| TASK-003 | `attachGatewayWsMessageHandler` | `src/gateway/server/ws-connection/message-handler.ts:133` | CWE-287 | **TP-PARAMETERIZE** | Critical |
| TASK-004 | `fetchRemoteMedia` | `src/media/fetch.ts:80` | CWE-918 | **TP-EXTRACT** | Critical |
| TASK-005 | `reconstructWebhookUrl` | `extensions/voice-call/src/webhook-security.ts:183` | CWE-347 | **TP-EXTRACT** | High |
| TASK-006 | `authorizeGatewayMethod` | `src/gateway/server-methods.ts:93` | CWE-287 | **TP-PARAMETERIZE** | High |
| TASK-007 | `validateTwilioSignature` | `extensions/voice-call/src/webhook-security.ts:12` | CWE-347 | **TP-PARAMETERIZE** | High (NEW) |
| TASK-008 | `validatePlivoV2Signature` | `extensions/voice-call/src/webhook-security.ts:458` | CWE-347 | **TP-PARAMETERIZE** | High (NEW) |
| TASK-009 | `fetchWithAuthFallback` | `extensions/msteams/src/attachments/download.ts:85` | CWE-287 | **TP-EXTRACT** | High |
| TASK-010 | `parseInstallSpec` | `src/agents/skills/frontmatter.ts:34` | CWE-502 | **TP-PARAMETERIZE** | Medium |
| TASK-011 | `spawnSignalDaemon` | `src/signal/daemon.ts:62` | CWE-78 | **TP-PARAMETERIZE** | Medium |
| TASK-012 | `validatePlivoV3Signature` | `extensions/voice-call/src/webhook-security.ts:538` | CWE-347 | **PARTIAL-TP** | Medium (NEW) |
| TASK-013 | `resolvePath` | `src/memory/backend-config.ts:101` | CWE-22 | **PARTIAL-TP** | Medium (NEW) |
| TASK-014 | `authorizeGatewayConnect` | `src/gateway/auth.ts:238` | CWE-287 | **PARTIAL-TP** | Medium |
| TASK-015 | `splitShellArgs` | `src/utils/shell-argv.ts:1` | CWE-78 | **PARTIAL-TP** | Low |
| TASK-016 | `parseConfigCommand` | `src/auto-reply/reply/config-commands.ts:9` | CWE-22 | **PARTIAL-TP** | Low |
| TASK-017 | `resolveFilePath` | `src/canvas-host/server.ts:158` | CWE-22 | **PARTIAL-TP** | Low |

## Detailed Finding Analysis

### Tier 1: Critical

#### TASK-001: `verifyTwilioWebhook` — Bypass Paths (TP-PARAMETERIZE)

**File:** `extensions/voice-call/src/webhook-security.ts:336` (87 lines)

Two bypass paths:
1. `skipVerification` option (line 369) returns `ok: true` immediately — documented
   as dev-only but no `NODE_ENV` guard prevents production use.
2. `allowNgrokFreeTierLoopbackBypass` (line 404) returns `ok: true` even when
   signature validation **fails**, if the URL contains `.ngrok-free.app` and the
   request comes from loopback. Exploitable via SSRF or any process on localhost.

**Root cause dependency:** Uses `reconstructWebhookUrl` (TASK-005) for URL
construction, inheriting the header injection vulnerability.

**Attack:** Remote, no auth required. Precondition: `skipVerification: true` in config
or SSRF-to-loopback for ngrok bypass.

#### TASK-002: `verifyPlivoWebhook` — skipVerification + Downgrade (TP-PARAMETERIZE)

**File:** `extensions/voice-call/src/webhook-security.ts:578` (112 lines)

Same `skipVerification` bypass (line 608). V3→V2 fallback creates theoretical
downgrade risk. Both V3 and V2 schemes use `reconstructWebhookUrl` (TASK-005),
so header injection defeats both. `timingSafeEqualString` correctly used for
HMAC comparison.

#### TASK-003: `attachGatewayWsMessageHandler` — Device Auth Bypass (TP-PARAMETERIZE)

**File:** `src/gateway/server/ws-connection/message-handler.ts:133` (876 actual lines)

Config flags `dangerouslyDisableDeviceAuth` and `allowInsecureAuth` disable device
identity verification. `client.id` is **self-reported** — attacker claims `"control-ui"`
to skip device pairing entirely. Legacy v1 signature fallback (lines 597-625)
accepted on loopback connections without nonce or replay protection.

**Attack:** Requires gateway token/password. Attacker connects WebSocket claiming
`client.id = "control-ui"`, gains full platform access without device pairing.

#### TASK-004: `fetchRemoteMedia` — SSRF via DNS Rebinding (TP-EXTRACT)

**File:** `src/media/fetch.ts:80` (92 lines)

**Upgraded from High to Critical in v2.** Fetches external URLs from channel messages.
`fetchWithSsrFGuard()` blocks private IP ranges, Content-Length checked against
maxBytes, MIME sniffed from binary. The SSRF guard is the critical control — vulnerable
to DNS rebinding (first resolve→public IP passes guard, second resolve→169.254.169.254
reaches cloud metadata). Error body truncated to 200 chars limits data exfiltration
but doesn't eliminate it.

**Attack:** Remote, via channel message. Send media URL pointing to DNS rebinding
service → access cloud metadata or internal services.

### Tier 2: High

#### TASK-005: `reconstructWebhookUrl` — Header Injection (TP-EXTRACT)

**File:** `extensions/voice-call/src/webhook-security.ts:183` (90 lines)

**Downgraded from Critical to High in v2.** Root cause for TASK-001 and TASK-002.
When `trustForwardingHeaders: true`, attacker-controlled `X-Forwarded-Host` /
`X-Original-Host` / `ngrok-forwarded-host` headers override the URL used for
HMAC computation. Has RFC 1123 hostname validation and optional `allowedHosts`
whitelist, but `allowedHosts` is not enforced by default. Silent fallback to
empty host string if all header sources fail.

#### TASK-006: `authorizeGatewayMethod` — Fragile RBAC (TP-PARAMETERIZE)

**File:** `src/gateway/server-methods.ts:93` (68 lines)

**Upgraded from PARTIAL-TP to TP-PARAMETERIZE in v2.** Hardcoded `startsWith()`
checks on 5 method categories for scope-to-permission mapping. Default deny at
catch-all (line 159) is safe. But adding new methods requires dual update: scope
check + return null for method resolution. Missing either creates auth bypass
(return null without scope check) or denial (scope check without return null).

#### TASK-007: `validateTwilioSignature` — Unvalidated Format (TP-PARAMETERIZE) [NEW]

**File:** `extensions/voice-call/src/webhook-security.ts:12` (31 lines)

**New finding in v2.** Accepts signature without validating base64 format before
HMAC comparison. Caller responsible for correct URL and parameter ordering.
No format guard before crypto comparison. Low-level primitive with inadequate
input validation at the API boundary.

#### TASK-008: `validatePlivoV2Signature` — Unvalidated Nonce (TP-PARAMETERIZE) [NEW]

**File:** `extensions/voice-call/src/webhook-security.ts:458` (15 lines)

**New finding in v2.** No nonce validation before HMAC computation. Empty or
specially crafted nonce could be exploited. No base64 format validation on
input signature. Normalizes base64 signatures without checking well-formedness.

#### TASK-009: `fetchWithAuthFallback` — Credential Leakage (TP-EXTRACT)

**File:** `extensions/msteams/src/attachments/download.ts:85` (59 lines)

**Upgraded from PARTIAL-TP to TP-EXTRACT in v2.** Downloads Teams attachments
unauthenticated-first, retries with auth on 401/403. URL allowlist enforced
(HTTPS only, hostname whitelist). Redirect chain only validates first redirect —
subsequent redirects could escape hostname allowlist and leak auth token to
attacker-controlled host. Unauthenticated-first approach also leaks timing
information about protected resources.

### Tier 3: Medium

#### TASK-010: `parseInstallSpec` — Unvalidated Package Names (TP-PARAMETERIZE)

**File:** `src/agents/skills/frontmatter.ts:34` (57 lines)

Kind field whitelisted (brew/node/go/uv/download) but values within each kind
(formula, package, url, targetDir) accepted without format validation or length
limits. Malicious YAML could specify shell metacharacters in formula or arbitrary
URLs in download kind.

#### TASK-011: `spawnSignalDaemon` — Unsanitized Account (TP-PARAMETERIZE)

**File:** `src/signal/daemon.ts:62` (31 lines)

Account param flows to `spawn()` args without alphanumeric validation. Array-form
spawn prevents shell injection, but no format check or length limit. Vulnerable
if signal-cli interprets account value unsafely or if refactored to shell execution.

#### TASK-012: `validatePlivoV3Signature` — Multi-Signature Acceptance (PARTIAL-TP) [NEW]

**File:** `extensions/voice-call/src/webhook-security.ts:538` (32 lines)

**New finding in v2.** Accepts comma-separated signatures in header, returns true
if ANY match. Timing-safe comparison on individual signatures is correct, but
multi-signature acceptance means attacker only needs to forge one valid signature.

#### TASK-013: `resolvePath` — Arbitrary Path Resolution (PARTIAL-TP) [NEW]

**File:** `src/memory/backend-config.ts:101` (10 lines)

**New finding in v2.** Resolves user-supplied config paths with `path.normalize()`
but no bounds check ensuring path stays within expected workspace. Could resolve
to arbitrary filesystem locations if config source is malicious.

#### TASK-014: `authorizeGatewayConnect` — Tailscale Priority (PARTIAL-TP)

**File:** `src/gateway/auth.ts:238` (54 lines)

**Downgraded from TP-EXTRACT to PARTIAL-TP in v2.** Proper `timingSafeEqual()` for
token/password comparison. Tailscale auth runs before token/password but is well-gated
with verified whois lookup requiring both headers AND proxy verification. Risk is
config-level: if `allowTailscale=true` and Tailscale is misconfigured, token/password
auth is never evaluated.

### Tier 4: Low

#### TASK-015: `splitShellArgs` — Trust-Sensitive Tokenizer (PARTIAL-TP)

**File:** `src/utils/shell-argv.ts:1` (62 lines)

Hand-written tokenizer returning parsed array. Returns null on unclosed quotes (safe).
But if user-controlled text reaches this function, token boundaries determine what
gets executed downstream. Caller-dependent risk.

#### TASK-016: `parseConfigCommand` — Config Path (PARTIAL-TP)

**File:** `src/auto-reply/reply/config-commands.ts:9` (63 lines)

Action validated against whitelist (`show|get|unset|set`). Config path accepted
as-is without traversal protection. Downstream config system must validate.

#### TASK-017: `resolveFilePath` — Canvas Path Traversal (PARTIAL-TP)

**File:** `src/canvas-host/server.ts:158` (37 lines)

**Downgraded from Medium to Low in v2.** Explicit `..` check + symlink rejection
via `lstat` + `openFileWithinRoot()` jail with `SafeOpenError`. Theoretical TOCTOU
race between lstat and open. `openFileWithinRoot` is the real guard — inline checks
are defense-in-depth.

### Notable Downgrades to FP (v2)

| Function | v1 Classification | v2 Classification | Reason |
|----------|-------------------|-------------------|--------|
| `parseMessageWithAttachments` | TP-EXTRACT High | **FP-STRUCTURAL** | Comprehensive validation: base64 charset check, length validation, size limits, MIME sniffing. Well-defended trust boundary. |
| `routeReply` | PARTIAL-TP Medium | **FP-INTERFACE** | Safe wrapper delegating to `deliverOutboundPayloads`. No direct trust boundary handling. |

## False Positive Summary by Detector

#### `secret_in_logging` — 0/7 TP (all FP-INTERFACE)

All 7 findings are the redaction/masking layer itself — mitigations, not vulnerabilities.

#### `command_injection_surface` — 5/7 FP

All `spawn()` calls use array-form arguments (shell-free). Only `splitShellArgs`
(trust-sensitive tokenizer, PARTIAL-TP) and `spawnSignalDaemon` (unsanitized account,
TP-PARAMETERIZE) flagged.

#### `auth_handler_complexity` — 47/50 FP

Majority are `applyAuthChoice*` (config setup), `buildAuthHealthSummary` (diagnostic),
`refreshOAuthTokenWithLock` (correct expiration checks), `resolveModelAuth*` (internal
resolution). Only `authorizeGatewayMethod`, `fetchWithAuthFallback`, and
`authorizeGatewayConnect` flagged.

#### `complex_input_parsers` — 48/50 FP

Most parsers process internal/validated data or use safe fallbacks. Only
`parseInstallSpec` (unvalidated YAML values) and `parseConfigCommand`
(unprotected config path) flagged.

#### `external_content_handlers` — 48/50 FP

Most are internal reply-pipeline functions. Only `fetchRemoteMedia` (SSRF surface)
and `parseConfigCommand` (via dual-detection) flagged.

#### `path_traversal_surface` — 6/8 FP

`listAgentFiles` uses constant ALLOWED_FILE_NAMES. `resolveFilenameFromSource` applies
`path.basename()`. Only `resolvePath` (arbitrary resolution) and `resolveFilePath`
(TOCTOU) flagged.

## Triage Workflow

### Phase 1: Critical Webhook + SSRF (TASK-001, 002, 004, 005)
1. **TASK-005**: Enforce `allowedHosts` by default in `reconstructWebhookUrl`
2. **TASK-001/002**: Add `NODE_ENV === "production"` guard to `skipVerification`
3. **TASK-001**: Remove or restrict ngrok loopback bypass
4. **TASK-004**: Audit `fetchWithSsrFGuard()` for DNS rebinding resistance
5. **TASK-007/008**: Add base64 format and nonce validation to low-level HMAC functions

### Phase 2: Critical Gateway Auth (TASK-003, 006)
1. **TASK-003**: Prevent `client.id` spoofing — device auth bypass must not rely on self-reported identity
2. **TASK-003**: Remove or deprecate legacy v1 signature fallback
3. **TASK-006**: Refactor `authorizeGatewayMethod` to table-driven RBAC

### Phase 3: High + Medium (TASK-009 through 014)
1. **TASK-009**: Validate full redirect chain in `fetchWithAuthFallback`
2. **TASK-010**: Validate URLs, paths, and package names in `parseInstallSpec`
3. **TASK-011**: Add alphanumeric validation to Signal daemon account parameter
4. **TASK-012**: Restrict `validatePlivoV3Signature` to single signature acceptance
5. **TASK-013**: Add bounds checking to `resolvePath` in backend-config
6. **TASK-014**: Document Tailscale auth priority ordering risk

### Phase 4: Low Priority (TASK-015, 016, 017)
1. **TASK-015**: Audit callers of `splitShellArgs` for user input exposure
2. **TASK-016**: Verify downstream config system validates paths
3. **TASK-017**: Verify `openFileWithinRoot` eliminates TOCTOU race

## Classification Criteria

### True Positive (TP) Categories

| Classification | Description | Example |
|---------------|-------------|---------|
| **TP-EXTRACT** | Function processes untrusted input at a trust boundary | `fetchRemoteMedia` fetches URLs from channel messages via SSRF-guarded fetch |
| **TP-PARAMETERIZE** | Function has bypass/skip parameters that weaken security | `verifyTwilioWebhook` with `skipVerification` and ngrok loopback bypass |
| **PARTIAL-TP** | Function has mitigations but incomplete coverage or edge cases | `authorizeGatewayConnect` has timing-safe comparison but Tailscale priority ordering risk |

### False Positive (FP) Categories

| Classification | Description | Example |
|---------------|-------------|---------|
| **FP-INTERFACE** | Function is a safe wrapper/facade, no actual risk | `routeReply` delegates to `deliverOutboundPayloads` |
| **FP-LAYERS** | Function is protected by upstream validation | `listAgentFiles` uses constant ALLOWED_FILE_NAMES |
| **FP-STRUCTURAL** | Complexity comes from exhaustive safe handling | `parseMessageWithAttachments` with comprehensive base64/size/MIME validation |
| **FP-TRIVIAL** | Function is too simple or display-only to contain meaningful bugs | `formatGatewayAuthFailureMessage` formats pre-determined error messages |

## Technical Notes

- **No call graph**: RETER TypeScript indexing does not produce call graph edges, so detectors rely on node-local predicates (function name, file path, line count)
- **Name-based heuristics**: Detectors use REGEX on function names and file paths as proxies for behavior — FPs are expected (8% overall TP rate)
- **Complexity proxy**: Line count is used as a proxy for cyclomatic complexity
- **Tunable thresholds**: All detectors accept `min_lines` and `limit` parameters
- **Detector precision by tier**: `webhook_verification_bypass` (86% TP) is the most precise detector; broad detectors like `complex_input_parsers` (4% TP) have high FP rates
- **Root cause chains**: TASK-001 and TASK-002 both depend on TASK-005 (`reconstructWebhookUrl`) as root cause — fixing TASK-005 partially mitigates both verifiers
- **v2 improvements**: Fresh source code verification with 4 parallel Explore agents. More conservative classification resulted in 4 downgrades and 4 new findings. Net: higher confidence in remaining TPs.
