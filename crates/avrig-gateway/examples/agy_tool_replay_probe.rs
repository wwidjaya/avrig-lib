//! Opt-in probe for the Antigravity Claude tool-replay field contract.
//!
//! The Antigravity `v1internal` endpoints accept Gemini-shaped payloads and translate them
//! upstream. For Claude models that translation ends in Anthropic `tool_use` / `tool_result`
//! blocks, which require an `id` on both sides of the pair. The Gemini schema documents
//! `functionCall.id` as optional and the translator's real behaviour for Claude is not
//! documented, so this probe measures it instead of assuming it.
//!
//! Properties this harness is built around:
//!
//! - It never sends conversation content. Prompts, tool arguments, and tool results are
//!   synthetic, and success is judged by an echoed sentinel, not by an HTTP status.
//! - It always reproduces the stale-payload baseline first (no id at all) and aborts if that
//!   baseline does not fail, because then the harness is not exercising pairing.
//! - It aborts on 429, 5xx, or transport errors instead of recording them as contract verdicts.
//! - Every request is counted against a hard cap.
//! - It reads a credentials file the caller supplies and copies it to a temporary file first, so
//!   the refreshed token is written to the copy and never to the source. Pass the live file and
//!   live config stay untouched; pass a copy and nothing is touched either.
//!
//! Usage:
//!   cargo run -p avrig-gateway --example agy_tool_replay_probe -- <credentials.json> [model]
//!
//! This is a scratch diagnostic. It is not compiled into the library or the service and it
//! contains no runtime switch.

use avrig_gateway::antigravity::oauth::load_credentials;
use avrig_gateway::antigravity::state::AppState;
use avrig_gateway::antigravity::types::{
    GATEWAY_URL, HUB_USER_AGENT, default_client_id, default_client_secret,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Duration;

/// Hard ceiling on outbound requests, including repeats and the streaming check.
const MAX_REQUESTS: usize = 20;
const SENTINEL: &str = "SENTINEL-7731-QK";
const TOOL: &str = "lookup_weather";

struct Variant {
    label: &'static str,
    /// Field name inside `functionCall` that carries the id, if any.
    call_field: Option<&'static str>,
    /// Field name inside `functionResponse` that carries the id, if any.
    response_field: Option<&'static str>,
    /// Put the id beside `functionCall` / `functionResponse` instead of inside them.
    part_level: bool,
    /// Attach a `thoughtSignature` to the replayed Claude call part.
    signature: bool,
    /// Extra alias field also carrying the same id inside `functionCall`.
    extra_call_alias: Option<&'static str>,
    id: &'static str,
}

const fn variant(
    label: &'static str,
    call_field: Option<&'static str>,
    response_field: Option<&'static str>,
    id: &'static str,
) -> Variant {
    Variant {
        label,
        call_field,
        response_field,
        part_level: false,
        signature: false,
        extra_call_alias: None,
        id,
    }
}

const LONG_BASE64URL_ID: &str = "call_0123456789abcdefghijklmnopqrstuvwxyzABCDEFG";

fn variants() -> Vec<Variant> {
    vec![
        variant(
            "functionCall.id + functionResponse.id (current behaviour)",
            Some("id"),
            Some("id"),
            "call_probe1",
        ),
        variant(
            "functionCall.callId + functionResponse.callId",
            Some("callId"),
            Some("callId"),
            "call_probe1",
        ),
        variant(
            "functionCall.toolCallId + functionResponse.toolCallId",
            Some("toolCallId"),
            Some("toolCallId"),
            "call_probe1",
        ),
        Variant {
            label: "part-level id beside functionCall/functionResponse",
            call_field: None,
            response_field: None,
            part_level: true,
            signature: false,
            extra_call_alias: None,
            id: "call_probe1",
        },
        Variant {
            label: "functionCall.id + callId alias together",
            call_field: Some("id"),
            response_field: Some("id"),
            part_level: false,
            signature: false,
            extra_call_alias: Some("callId"),
            id: "call_probe1",
        },
        variant(
            "Anthropic-style toolu_ id",
            Some("id"),
            Some("id"),
            "toolu_01ProbeAnthropicStyleIdentifierXYZ",
        ),
        variant(
            "long base64url id (call_ + 43 chars)",
            Some("id"),
            Some("id"),
            LONG_BASE64URL_ID,
        ),
        variant(
            "charset id with - and _",
            Some("id"),
            Some("id"),
            "call_ab-CD_ef0123456789",
        ),
        Variant {
            label: "functionCall.id + thoughtSignature on the Claude call part",
            call_field: Some("id"),
            response_field: Some("id"),
            part_level: false,
            signature: true,
            extra_call_alias: None,
            id: "call_probe1",
        },
    ]
}

fn tool_declaration() -> Value {
    json!([{
        "functionDeclarations": [{
            "name": TOOL,
            "description": "Return the verification code for a city.",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }]
    }])
}

/// Minimal synthetic history: prompt, one assistant tool call, and its result.
fn replay_request(variant: &Variant) -> Value {
    let mut call = json!({
        "name": TOOL,
        "args": { "city": "Jakarta" }
    });
    let mut call_part = json!({ "functionCall": call.clone() });

    if let Some(field) = variant.call_field {
        call[field] = json!(variant.id);
        call_part = json!({ "functionCall": call });
    }
    if let Some(alias) = variant.extra_call_alias {
        call[alias] = json!(variant.id);
        call_part = json!({ "functionCall": call });
    }
    if variant.part_level {
        call_part["id"] = json!(variant.id);
    }
    if variant.signature {
        call_part["thoughtSignature"] =
            json!(avrig_gateway::antigravity::types::SKIP_THOUGHT_SIGNATURE);
    }

    let mut response = json!({
        "name": TOOL,
        "response": { "result": { "verification_code": SENTINEL } }
    });
    if let Some(field) = variant.response_field {
        response[field] = json!(variant.id);
    }
    let mut response_part = json!({ "functionResponse": response });
    if variant.part_level {
        response_part["id"] = json!(variant.id);
    }

    json!({
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": format!(
                        "Call the {TOOL} tool for Jakarta. Its result contains a verification \
                         code. Reply with only that code."
                    )
                }]
            },
            { "role": "model", "parts": [call_part] },
            { "role": "user", "parts": [response_part] }
        ],
        "generationConfig": { "temperature": 0.0, "maxOutputTokens": 64 },
        "tools": tool_declaration()
    })
}

/// The shape the deployed stale binary produced: no id on either side of the pair.
fn stale_baseline_request() -> Value {
    json!({
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": format!(
                        "Call the {TOOL} tool for Jakarta. Its result contains a verification \
                         code. Reply with only that code."
                    )
                }]
            },
            {
                "role": "model",
                "parts": [{ "functionCall": { "name": TOOL, "args": { "city": "Jakarta" } } }]
            },
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": TOOL,
                        "response": { "result": { "verification_code": SENTINEL } }
                    }
                }]
            }
        ],
        "generationConfig": { "temperature": 0.0, "maxOutputTokens": 64 },
        "tools": tool_declaration()
    })
}

/// Deliberately unpaired result: the harness must be able to provoke a pairing failure.
fn unpaired_result_request() -> Value {
    json!({
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": format!("Call the {TOOL} tool for Jakarta and report the code.")
                }]
            },
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": TOOL,
                        "response": { "result": { "verification_code": SENTINEL } },
                        "id": "call_never_called"
                    }
                }]
            }
        ],
        "generationConfig": { "temperature": 0.0, "maxOutputTokens": 64 },
        "tools": tool_declaration()
    })
}

fn payload(project: &str, model: &str, request: Value) -> Value {
    json!({ "project": project, "model": model, "request": request })
}

enum Outcome {
    Accepted { status: u16, body: String },
    Rejected { status: u16, body: String },
    Aborted { reason: String },
}

struct Requester {
    state: AppState,
    used: usize,
}

impl Requester {
    async fn post(&mut self, project: &str, model: &str, request: Value) -> Outcome {
        if self.used >= MAX_REQUESTS {
            return Outcome::Aborted {
                reason: format!("request cap {MAX_REQUESTS} reached"),
            };
        }
        self.used += 1;

        let access_token = {
            let creds = self.state.creds.read().await;
            creds.access_token.clone()
        };
        let url = format!("{}/v1internal:generateContent", GATEWAY_URL);
        let response = self
            .state
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("User-Agent", HUB_USER_AGENT)
            .timeout(Duration::from_secs(60))
            .json(&payload(project, model, request))
            .send()
            .await;

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                return Outcome::Aborted {
                    reason: format!("transport: {error}"),
                };
            }
        };
        let status = response.status().as_u16();
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                return Outcome::Aborted {
                    reason: format!("body read: {error}"),
                };
            }
        };

        match status {
            200..=299 => Outcome::Accepted { status, body },
            400..=499 if status != 429 => Outcome::Rejected { status, body },
            _ => Outcome::Aborted {
                reason: format!("status {status}: {}", truncate(&body, 300)),
            },
        }
    }

    async fn post_stream(&mut self, project: &str, model: &str, request: Value) -> Outcome {
        if self.used >= MAX_REQUESTS {
            return Outcome::Aborted {
                reason: format!("request cap {MAX_REQUESTS} reached"),
            };
        }
        self.used += 1;

        let access_token = {
            let creds = self.state.creds.read().await;
            creds.access_token.clone()
        };
        let url = format!("{}/v1internal:streamGenerateContent?alt=sse", GATEWAY_URL);
        let response = self
            .state
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("User-Agent", HUB_USER_AGENT)
            .header("Accept", "text/event-stream")
            .timeout(Duration::from_secs(60))
            .json(&payload(project, model, request))
            .send()
            .await;

        let response = match response {
            Ok(response) => response,
            Err(error) => {
                return Outcome::Aborted {
                    reason: format!("transport: {error}"),
                };
            }
        };
        let status = response.status().as_u16();
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                return Outcome::Aborted {
                    reason: format!("body read: {error}"),
                };
            }
        };
        match status {
            200..=299 => Outcome::Accepted { status, body },
            400..=499 if status != 429 => Outcome::Rejected { status, body },
            _ => Outcome::Aborted {
                reason: format!("status {status}: {}", truncate(&body, 300)),
            },
        }
    }
}

fn truncate(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

/// Behavioural success: the completion echoes the sentinel carried by the synthetic tool result.
fn echoes_sentinel(body: &str) -> bool {
    body.contains(SENTINEL)
}

/// Concatenate the text parts of an SSE stream so split deltas do not hide the sentinel.
fn sse_text(body: &str) -> String {
    let mut text = String::new();
    for line in body.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let value = value.get("response").cloned().unwrap_or(value);
        if let Some(parts) = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
            .and_then(|first| first.get("content"))
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                    text.push_str(chunk);
                }
            }
        }
    }
    text
}

fn describe(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return format!("non-JSON body: {}", truncate(body, 160));
    };
    let body = value.get("response").unwrap_or(&value);
    let candidates = body
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let Some(first) = candidates.first() else {
        return "no candidates".to_string();
    };
    let finish = first
        .get("finishReason")
        .and_then(Value::as_str)
        .unwrap_or("(none)");
    let text: String = first
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();
    format!("finishReason={finish} text={:?}", truncate(&text, 120))
}

fn error_message(body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok().and_then(|value| {
        value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    parsed.unwrap_or_else(|| truncate(body, 300))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(creds_arg) = args.next() else {
        eprintln!(
            "usage: agy_tool_replay_probe <credentials.json> [model]\n\
             The file is copied and refreshed in place of the original."
        );
        std::process::exit(2);
    };
    let model = args
        .next()
        .unwrap_or_else(|| "claude-opus-5-5-high".to_string());
    let stream_only = args.any(|arg| arg == "--stream-only");
    let source = PathBuf::from(creds_arg);

    // Work on a copy so a refreshed token never lands in the file the service reads.
    let scratch = std::env::temp_dir().join(format!("agy-probe-{}.json", std::process::id()));
    let original = load_credentials(&source)?;
    avrig_gateway::antigravity::oauth::save_credentials(&scratch, &original)?;
    let creds = load_credentials(&scratch)?;

    let state = AppState::new(
        scratch.clone(),
        creds
            .client_id
            .clone()
            .unwrap_or_else(default_client_id),
        creds
            .client_secret
            .clone()
            .unwrap_or_else(default_client_secret),
        creds,
    );
    state.refresh_token().await?;
    let project = {
        let creds = state.creds.read().await;
        creds.project.clone()
    };

    println!("target model={model} project={project} cap={MAX_REQUESTS}");
    let mut requester = Requester { state, used: 0 };

    // 1. Baseline: the stale-binary payload shape. It must fail, or nothing below is meaningful.
    match requester
        .post(&project, &model, stale_baseline_request())
        .await
    {
        Outcome::Rejected { status, body } => {
            println!("baseline (no ids): {status} {}", error_message(&body));
            if !error_message(&body).contains("tool_use.id") {
                println!("aborting: baseline failed but not on tool_use.id");
                return Ok(());
            }
        }
        Outcome::Accepted { status, body } => {
            println!(
                "aborting: baseline was accepted ({status}), harness cannot prove pairing: {}",
                describe(&body)
            );
            return Ok(());
        }
        Outcome::Aborted { reason } => {
            println!("aborting: baseline {reason}");
            return Ok(());
        }
    }

    // 2. Negative control: an unpaired result must fail, proving the route checks pairing.
    match requester
        .post(&project, &model, unpaired_result_request())
        .await
    {
        Outcome::Rejected { status, body } => {
            println!(
                "negative control (unpaired result): {status} {}",
                error_message(&body)
            );
        }
        Outcome::Accepted { status, body } => {
            println!(
                "aborting: an unpaired result was accepted ({status}): {}",
                describe(&body)
            );
            return Ok(());
        }
        Outcome::Aborted { reason } => {
            println!("aborting: negative control {reason}");
            return Ok(());
        }
    }

    // 3. One field variant per request, stopping at the first behavioural success.
    let mut winner: Option<Variant> = None;
    for candidate in variants() {
        match requester
            .post(&project, &model, replay_request(&candidate))
            .await
        {
            Outcome::Rejected { status, body } => {
                println!(
                    "rejected  {status}  {}  -> {}",
                    candidate.label,
                    error_message(&body)
                );
            }
            Outcome::Accepted { status, body } => {
                if echoes_sentinel(&body) {
                    println!(
                        "ACCEPTED  {status}  {}  -> {}",
                        candidate.label,
                        describe(&body)
                    );
                    winner = Some(candidate);
                    break;
                }
                println!(
                    "accepted-but-not-behavioural  {status}  {}  -> {}",
                    candidate.label,
                    describe(&body)
                );
            }
            Outcome::Aborted { reason } => {
                println!("aborting during variants: {reason}");
                return Ok(());
            }
        }
    }

    let Some(winner) = winner else {
        println!("result: no structured field variant succeeded behaviourally");
        println!("requests used: {}", requester.used);
        return Ok(());
    };

    // 4. Determinism: repeat the winner three times, then re-run the baseline.
    for attempt in 1..=3 {
        match requester
            .post(&project, &model, replay_request(&winner))
            .await
        {
            Outcome::Accepted { body, .. } if echoes_sentinel(&body) => {
                println!("repeat {attempt}: ok");
            }
            Outcome::Accepted { body, .. } => {
                println!(
                    "repeat {attempt}: accepted without the sentinel: {}",
                    describe(&body)
                );
            }
            Outcome::Rejected { status, body } => {
                println!(
                    "repeat {attempt}: rejected {status} {}",
                    error_message(&body)
                );
            }
            Outcome::Aborted { reason } => {
                println!("repeat {attempt}: aborting, {reason}");
                return Ok(());
            }
        }
    }

    match requester
        .post(&project, &model, stale_baseline_request())
        .await
    {
        Outcome::Rejected { status, body } => {
            println!("baseline re-run: {status} {}", error_message(&body));
        }
        other => {
            let label = match other {
                Outcome::Accepted { status, .. } => format!("accepted ({status})"),
                Outcome::Rejected { status, .. } => format!("rejected ({status})"),
                Outcome::Aborted { reason } => format!("aborted ({reason})"),
            };
            println!("baseline re-run: {label}, determinism not proven");
        }
    }

    // 5. The streaming endpoint is what the gateway actually uses for chat.
    match requester
        .post_stream(&project, &model, replay_request(&winner))
        .await
    {
        Outcome::Accepted { status, body } => {
            let text = sse_text(&body);
            println!(
                "stream winner: {status} sentinel={} text={:?}",
                text.contains(SENTINEL),
                truncate(&text, 120)
            );
        }
        Outcome::Rejected { status, body } => {
            println!("stream winner: rejected {status} {}", error_message(&body))
        }
        Outcome::Aborted { reason } => println!("stream winner: {reason}"),
    }

    println!("winner: {}", winner.label);
    println!("requests used: {}/{}", requester.used, MAX_REQUESTS);

    // A rotated refresh token would only exist in the scratch copy. Say so loudly: the live
    // service would still hold the previous one.
    let refreshed = load_credentials(&scratch)?;
    if refreshed.refresh_token != original.refresh_token {
        println!(
            "WARNING: upstream issued a new refresh token. The live credentials file was not \
             modified; it still holds the previous token."
        );
    }

    let _ = std::fs::remove_file(&scratch);
    Ok(())
}
