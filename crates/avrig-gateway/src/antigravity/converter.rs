use super::state::AppState;
use super::types::{ChatMessage, ChatRequest};
use anyhow::{Result, anyhow};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore;
use serde_json::{Map, Value, json};
use std::collections::HashSet;

#[derive(Debug, thiserror::Error)]
pub enum ToolHistoryValidationError {
    #[error("tool call ID must not be empty")]
    EmptyCallId,
    #[error("duplicate tool call ID: {0}")]
    DuplicateCallId(String),
    #[error("tool result is missing tool_call_id")]
    MissingResultCallId,
    #[error("orphan, duplicate, or mismatched tool result: {0}")]
    InvalidResult(String),
    #[error("unmatched tool call(s): {0}")]
    UnmatchedCalls(usize),
}

pub const ALLOWED_SCHEMA_FIELDS: &[&str] = &[
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "items",
    "minProperties",
    "maxProperties",
    "minimum",
    "maximum",
    "default",
    "title",
    "anyOf",
    "allOf",
    "oneOf",
    "not",
    "pattern",
    "minLength",
    "maxLength",
    "example",
    "additionalProperties",
];

pub fn normalize_model_id(model: &str) -> String {
    let mut m = model.trim();
    if let Some(idx) = m.rfind('/') {
        m = &m[idx + 1..];
    }
    if m == "default" || m.is_empty() {
        "gemini-3.8-flash-tiered".to_string()
    } else {
        m.to_string()
    }
}

/// Claude models proxied through Antigravity are translated upstream into Anthropic
/// `tool_use` / `tool_result` blocks, which require a matching `id` on both sides of the
/// pair. Gemini models on the same route stay on the native structured-parts contract.
pub fn is_claude_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("claude")
}

pub fn finish_reason(value: &str) -> String {
    match value.to_ascii_uppercase().as_str() {
        "MAX_TOKENS" => "length".to_string(),
        "STOP" => "stop".to_string(),
        "SAFETY" | "RECITATION" => "content_filter".to_string(),
        _ => "stop".to_string(),
    }
}

pub fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    URL_SAFE_NO_PAD.encode(b)
}

pub fn message_text(content: &Option<Value>) -> Result<String> {
    match content {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(arr)) => {
            let mut out = String::new();
            for item in arr {
                if let Value::Object(map) = item {
                    if let Some(Value::String(s)) = map.get("text") {
                        out.push_str(s);
                    }
                }
            }
            Ok(out)
        }
        None | Some(Value::Null) => Ok(String::new()),
        Some(other) => Err(anyhow!("unsupported message content type: {}", other)),
    }
}

pub fn message_character_count(messages: &[ChatMessage]) -> usize {
    let mut total = 0;
    for message in messages {
        if let Ok(text) = message_text(&message.content) {
            total += text.len();
        }
    }
    total
}

/// Reject malformed OpenAI tool history before it is translated to provider turns.
pub fn validate_tool_history(
    messages: &[ChatMessage],
) -> std::result::Result<(), ToolHistoryValidationError> {
    let mut seen = HashSet::new();
    let mut pending = HashSet::new();

    for message in messages {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                if call.id.trim().is_empty() {
                    return Err(ToolHistoryValidationError::EmptyCallId);
                }
                if !seen.insert(call.id.as_str()) {
                    return Err(ToolHistoryValidationError::DuplicateCallId(call.id.clone()));
                }
                pending.insert(call.id.as_str());
            }
        }

        if message.role == "tool" {
            let id = message
                .tool_call_id
                .as_deref()
                .filter(|id| !id.trim().is_empty())
                .ok_or(ToolHistoryValidationError::MissingResultCallId)?;
            if !pending.remove(id) {
                return Err(ToolHistoryValidationError::InvalidResult(id.to_string()));
            }
        }
    }

    if !pending.is_empty() {
        return Err(ToolHistoryValidationError::UnmatchedCalls(pending.len()));
    }
    Ok(())
}

/// Environment gate for the outbound payload trace. Off unless explicitly set to a truthy
/// value, because the trace contains message structure even after redaction.
pub const TRACE_REQUEST_ENV: &str = "AVRIG_AGY_TRACE_REQUEST";

pub fn trace_request_enabled() -> bool {
    match std::env::var(TRACE_REQUEST_ENV) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn redact_text(value: &Value) -> Value {
    match value.as_str() {
        Some(text) => json!(format!("<redacted:{} chars>", text.chars().count())),
        None => json!("<redacted>"),
    }
}

fn redact_shape(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, nested)| (key.clone(), redact_shape(nested)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(|_| json!("<redacted>")).collect()),
        Value::Null => Value::Null,
        _ => json!("<redacted>"),
    }
}

fn redact_function_call(value: &Value) -> Value {
    let mut out = Map::new();
    if let Some(map) = value.as_object() {
        for (key, nested) in map {
            match key.as_str() {
                // Identity fields survive so a payload can be matched to a client request.
                "name" | "id" => {
                    out.insert(key.clone(), nested.clone());
                }
                "args" => {
                    out.insert(key.clone(), redact_shape(nested));
                }
                _ => {
                    out.insert(key.clone(), redact_shape(nested));
                }
            }
        }
    }
    Value::Object(out)
}

fn redact_part(part: &Value) -> Value {
    let mut out = Map::new();
    if let Some(map) = part.as_object() {
        for (key, nested) in map {
            match key.as_str() {
                "text" | "thoughtSignature" => {
                    out.insert(key.clone(), redact_text(nested));
                }
                "functionCall" => {
                    out.insert(key.clone(), redact_function_call(nested));
                }
                "functionResponse" => {
                    out.insert(key.clone(), redact_function_call(nested));
                }
                "inlineData" => {
                    let mut inline = Map::new();
                    if let Some(inline_map) = nested.as_object() {
                        for (inline_key, inline_value) in inline_map {
                            if inline_key == "mimeType" {
                                inline.insert(inline_key.clone(), inline_value.clone());
                            } else {
                                inline.insert(inline_key.clone(), redact_text(inline_value));
                            }
                        }
                    }
                    out.insert(key.clone(), Value::Object(inline));
                }
                _ => {
                    out.insert(key.clone(), redact_shape(nested));
                }
            }
        }
    }
    Value::Object(out)
}

/// Build a redaction-safe view of the serialized Antigravity request.
///
/// Message text, tool argument values, tool result values, inline image bytes, and
/// thought signatures are replaced with placeholders. Roles, part kinds, field names,
/// and tool call IDs are retained so the wire contract can be inspected without
/// leaking conversation content.
pub fn redacted_request_trace(payload: &Value) -> Value {
    let request = payload.get("request");
    let mut contents = Vec::new();
    if let Some(turns) = request
        .and_then(|r| r.get("contents"))
        .and_then(Value::as_array)
    {
        for turn in turns {
            let mut redacted_turn = Map::new();
            redacted_turn.insert(
                "role".to_string(),
                turn.get("role").cloned().unwrap_or(Value::Null),
            );
            let parts = turn
                .get("parts")
                .and_then(Value::as_array)
                .map(|parts| parts.iter().map(redact_part).collect::<Vec<_>>())
                .unwrap_or_default();
            redacted_turn.insert("parts".to_string(), Value::Array(parts));
            contents.push(Value::Object(redacted_turn));
        }
    }

    let mut out = Map::new();
    out.insert(
        "project".to_string(),
        payload.get("project").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "model".to_string(),
        payload.get("model").cloned().unwrap_or(Value::Null),
    );
    out.insert("contents".to_string(), Value::Array(contents));

    if let Some(gen_config) = request
        .and_then(|r| r.get("generationConfig"))
        .filter(|c| c.is_object())
    {
        // Generation config holds numbers and flags only, never conversation content.
        out.insert("generationConfig".to_string(), gen_config.clone());
    }

    if let Some(system) = request.and_then(|r| r.get("systemInstruction")) {
        let parts = system
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .map(|part| {
                        let mut redacted = Map::new();
                        if let Some(map) = part.as_object() {
                            for (key, nested) in map {
                                if key == "text" {
                                    redacted.insert(key.clone(), redact_text(nested));
                                } else {
                                    redacted.insert(key.clone(), redact_shape(nested));
                                }
                            }
                        }
                        Value::Object(redacted)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        out.insert("systemInstruction".to_string(), json!({ "parts": parts }));
    }

    if let Some(tools) = request
        .and_then(|r| r.get("tools"))
        .and_then(Value::as_array)
    {
        // Tool declarations are client-supplied schemas: names and parameter shapes are
        // kept, free-text descriptions are redacted.
        let redacted_tools = tools
            .iter()
            .map(|tool| {
                let mut tool_out = Map::new();
                if let Some(map) = tool.as_object() {
                    for (key, nested) in map {
                        if key == "functionDeclarations" {
                            let decls = nested
                                .as_array()
                                .map(|decls| {
                                    decls
                                        .iter()
                                        .map(|decl| {
                                            let mut decl_out = Map::new();
                                            if let Some(decl_map) = decl.as_object() {
                                                for (decl_key, decl_value) in decl_map {
                                                    if decl_key == "description" {
                                                        decl_out.insert(
                                                            decl_key.clone(),
                                                            redact_text(decl_value),
                                                        );
                                                    } else {
                                                        decl_out.insert(
                                                            decl_key.clone(),
                                                            decl_value.clone(),
                                                        );
                                                    }
                                                }
                                            }
                                            Value::Object(decl_out)
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            tool_out.insert(key.clone(), Value::Array(decls));
                        } else {
                            tool_out.insert(key.clone(), nested.clone());
                        }
                    }
                }
                Value::Object(tool_out)
            })
            .collect::<Vec<_>>();
        out.insert("tools".to_string(), Value::Array(redacted_tools));
    }

    Value::Object(out)
}

/// Emit the redacted request trace when `AVRIG_AGY_TRACE_REQUEST` is enabled.
pub fn trace_antigravity_request(payload: &Value) {
    if !trace_request_enabled() {
        return;
    }
    let trace = redacted_request_trace(payload);
    tracing::info!(
        target: "avrig_gateway::antigravity",
        model = %payload.get("model").and_then(|m| m.as_str()).unwrap_or(""),
        trace = %trace,
        "serialized antigravity request (redacted)"
    );
}

pub fn requested_max_tokens(input: &ChatRequest) -> i64 {
    if let Some(mt) = input.max_tokens {
        if mt > 0 {
            return mt;
        }
    }
    if let Some(mct) = input.max_completion_tokens {
        if mct > 0 {
            return mct;
        }
    }
    input.max_output_tokens.unwrap_or(0)
}

/// Calculate thinking budget and effective maxOutputTokens for Gemini models.
/// Prevents low-latency / small-budget calls from exhausting maxOutputTokens on internal reasoning.
pub fn calculate_thinking_config(model: &str, requested_max: i64) -> (i64, Option<Value>) {
    let m = model.to_ascii_lowercase();
    let is_gemini = m.contains("gemini");
    let is_low = m.ends_with("-low") || m.contains("flash-low");

    if !is_gemini {
        return (requested_max, None);
    }

    if is_low {
        // Low-latency agentic loop variant
        let (budget, effective_max) = if requested_max > 0 && requested_max <= 100 {
            (0, requested_max)
        } else if requested_max > 100 && requested_max <= 500 {
            (64, requested_max + 64)
        } else if requested_max > 500 {
            (128, requested_max + 128)
        } else {
            // Unspecified or 0
            (128, 0)
        };

        (
            effective_max,
            Some(json!({
                "thinkingBudget": budget
            })),
        )
    } else {
        // Standard Gemini models (e.g. gemini-3.8-flash, gemini-3.8-pro)
        if requested_max > 0 && requested_max < 300 {
            let budget = requested_max / 2;
            let effective_max = requested_max + 256;
            (
                effective_max,
                Some(json!({
                    "thinkingBudget": budget
                })),
            )
        } else {
            (requested_max, None)
        }
    }
}

pub fn sanitize_gemini_schema(val: Value) -> Value {
    match val {
        Value::Object(mut map) => {
            if let Some(Value::Array(type_arr)) = map.get("type").cloned() {
                let mut non_null_types = Vec::new();
                let mut is_nullable = false;
                for t in type_arr {
                    if let Value::String(s) = t {
                        if s == "null" {
                            is_nullable = true;
                        } else {
                            non_null_types.push(s);
                        }
                    }
                }
                if non_null_types.len() == 1 {
                    map.insert("type".to_string(), Value::String(non_null_types[0].clone()));
                    if is_nullable {
                        map.insert("nullable".to_string(), Value::Bool(true));
                    }
                }
            }

            let mut result = Map::new();
            for (k, v) in map {
                if !ALLOWED_SCHEMA_FIELDS.contains(&k.as_str()) {
                    continue;
                }
                match k.as_str() {
                    "properties" => {
                        if let Value::Object(props) = v {
                            let mut clean_props = Map::new();
                            for (pk, pv) in props {
                                clean_props.insert(pk, sanitize_gemini_schema(pv));
                            }
                            result.insert(k, Value::Object(clean_props));
                        }
                    }
                    "items" => {
                        result.insert(k, sanitize_gemini_schema(v));
                    }
                    "anyOf" | "allOf" | "oneOf" => {
                        if let Value::Array(list) = v {
                            let clean_list: Vec<Value> =
                                list.into_iter().map(sanitize_gemini_schema).collect();
                            result.insert(k, Value::Array(clean_list));
                        }
                    }
                    "not" => {
                        result.insert(k, sanitize_gemini_schema(v));
                    }
                    _ => {
                        result.insert(k, v);
                    }
                }
            }

            // Clean up 'required' array: Google Gemini rejects any required property name
            // that is not explicitly present in 'properties' (HTTP 400: property is not defined)
            if let Some(Value::Array(req_arr)) = result.get("required").cloned() {
                if let Some(Value::Object(props)) = result.get("properties") {
                    let filtered_req: Vec<Value> = req_arr
                        .into_iter()
                        .filter(|item| {
                            if let Some(prop_name) = item.as_str() {
                                props.contains_key(prop_name)
                            } else {
                                false
                            }
                        })
                        .collect();
                    if filtered_req.is_empty() {
                        result.remove("required");
                    } else {
                        result.insert("required".to_string(), Value::Array(filtered_req));
                    }
                } else {
                    result.remove("required");
                }
            }

            Value::Object(result)
        }
        Value::Array(arr) => {
            let out_arr = arr.into_iter().map(sanitize_gemini_schema).collect();
            Value::Array(out_arr)
        }
        other => other,
    }
}

/// Append a replayed tool result to the conversation.
///
/// Anthropic requires strict user/assistant alternation, and parallel tool calls arrive as
/// several consecutive `tool` messages. When `merge` is set, adjacent `functionResponse` parts
/// are folded into the preceding result turn so Claude never sees two user turns in a row.
/// Gemini keeps one turn per result: its native contract accepts consecutive same-role turns
/// and its serialized payload must stay byte-identical.
fn push_tool_result(contents: &mut Vec<Value>, part: Value, merge: bool) {
    if merge {
        let merges_into_previous = contents
            .last()
            .map(|turn| {
                turn.get("role").and_then(Value::as_str) == Some("user")
                    && turn
                        .get("parts")
                        .and_then(Value::as_array)
                        .map(|parts| {
                            !parts.is_empty()
                                && parts
                                    .iter()
                                    .all(|part| part.get("functionResponse").is_some())
                        })
                        .unwrap_or(false)
            })
            .unwrap_or(false);

        if merges_into_previous {
            if let Some(parts) = contents
                .last_mut()
                .and_then(|turn| turn.get_mut("parts"))
                .and_then(Value::as_array_mut)
            {
                parts.push(part);
                return;
            }
        }
    }

    contents.push(json!({
        "role": "user",
        "parts": [part]
    }));
}

pub async fn to_antigravity_request(
    state: &AppState,
    input: &ChatRequest,
    project: &str,
) -> Result<Value> {
    validate_tool_history(&input.messages)?;
    let model = normalize_model_id(&input.model);
    let is_claude = is_claude_model(&model);
    let mut contents: Vec<Value> = Vec::new();
    let mut system_instruction_text = String::new();

    for msg in &input.messages {
        match msg.role.as_str() {
            "system" => {
                let text = message_text(&msg.content)?;
                if !text.is_empty() {
                    if !system_instruction_text.is_empty() {
                        system_instruction_text.push_str("\n\n");
                    }
                    system_instruction_text.push_str(&text);
                }
            }
            "user" => {
                let mut parts = Vec::new();
                if let Some(content_val) = &msg.content {
                    match content_val {
                        Value::String(s) => parts.push(json!({ "text": s })),
                        Value::Array(arr) => {
                            for item in arr {
                                if let Value::Object(map) = item {
                                    if let Some(Value::String(s)) = map.get("text") {
                                        parts.push(json!({ "text": s }));
                                    } else if let Some(Value::Object(img_map)) =
                                        map.get("image_url")
                                    {
                                        if let Some(Value::String(url_str)) = img_map.get("url") {
                                            if let Some((mime, b64_data)) =
                                                url_str.strip_prefix("data:").and_then(|rest| {
                                                    let (mime, b64) =
                                                        rest.split_once(";base64,")?;
                                                    Some((mime, b64))
                                                })
                                            {
                                                parts.push(json!({
                                                    "inlineData": {
                                                        "mimeType": mime,
                                                        "data": b64_data
                                                    }
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                contents.push(json!({
                    "role": "user",
                    "parts": parts
                }));
            }
            "assistant" => {
                let mut parts = Vec::new();
                if let Ok(text) = message_text(&msg.content) {
                    if !text.is_empty() {
                        parts.push(json!({ "text": text }));
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    let mut found_first_function_call = false;
                    for call in tool_calls {
                        let parsed_args: Value =
                            serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));

                        let sig = call.thought_signature.as_deref().unwrap_or("");
                        state
                            .set_call_info(&call.id, &call.function.name, sig)
                            .await;

                        let fc_obj = if is_claude {
                            // Anthropic `tool_use` blocks need the client-supplied call ID, and the
                            // only channel on this route is the Gemini-shaped `functionCall.id`.
                            // Whether the Antigravity translator honours that field for Claude is
                            // undocumented and not yet verified against the live service; see
                            // Phase 3 of the tool replay plan.
                            json!({
                                "name": call.function.name,
                                "args": parsed_args,
                                "id": call.id,
                            })
                        } else {
                            json!({
                                "name": call.function.name,
                                "args": parsed_args
                            })
                        };

                        // Google Gemini enforces thought_signature on replayed function calls.
                        // Gemini rule: Only the first functionCall part in a block carries the signature;
                        // parallel function calls must not have a signature. If the first function call
                        // is missing a valid signature, inject the documented skip_thought_signature_validator
                        // sentinel to prevent HTTP 400 rejection.
                        // The Claude path omits the field entirely because Anthropic blocks carry no
                        // Gemini thought signature.
                        let part = if is_claude {
                            json!({ "functionCall": fc_obj })
                        } else if !found_first_function_call {
                            found_first_function_call = true;
                            let effective_sig = if !sig.is_empty() && sig.len() >= 50 {
                                sig
                            } else {
                                crate::antigravity::types::SKIP_THOUGHT_SIGNATURE
                            };
                            json!({
                                "functionCall": fc_obj,
                                "thoughtSignature": effective_sig,
                            })
                        } else {
                            json!({
                                "functionCall": fc_obj,
                            })
                        };

                        parts.push(part);
                    }
                }
                contents.push(json!({
                    "role": "model",
                    "parts": parts
                }));
            }
            "tool" => {
                let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
                let (name, sig) = state.get_call_info(tool_call_id).await;
                let text = message_text(&msg.content)?;
                let content_val: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));

                let part = if is_claude {
                    // Matching ID keeps the tool_result paired with its tool_use block.
                    json!({
                        "functionResponse": {
                            "name": name,
                            "response": { "result": content_val },
                            "id": tool_call_id
                        }
                    })
                } else {
                    let effective_sig = if !sig.is_empty() && sig.len() >= 50 {
                        sig
                    } else {
                        crate::antigravity::types::SKIP_THOUGHT_SIGNATURE.to_string()
                    };

                    json!({
                        "functionResponse": {
                            "name": name,
                            "response": { "result": content_val }
                        },
                        "thoughtSignature": effective_sig
                    })
                };
                push_tool_result(&mut contents, part, is_claude);
            }
            _ => {}
        }
    }

    let mut gen_config = json!({});
    if let Some(temp) = input.temperature {
        gen_config["temperature"] = json!(temp);
    }
    if let Some(top_p) = input.top_p {
        gen_config["topP"] = json!(top_p);
    }
    let req_tok = requested_max_tokens(input);
    let (effective_max_tok, thinking_cfg) = calculate_thinking_config(&model, req_tok);
    if effective_max_tok > 0 {
        gen_config["maxOutputTokens"] = json!(effective_max_tok);
    }
    if let Some(cfg) = thinking_cfg {
        gen_config["thinkingConfig"] = cfg;
    }

    // Google Antigravity daily endpoint enforces a strict ~3,500 character limit on request.systemInstruction.
    // As documented in vault 03-knowledge/AI/Harness/Google-OAuth-Error.md:
    // If systemInstruction exceeds 3,500 chars (e.g. OpenCode plan agent rules / harness),
    // convert it into an initial synthetic conversation turn (user prompt + model acknowledgment),
    // and keep a compact identity prompt in systemInstruction.
    let system_instruction = if !system_instruction_text.trim().is_empty() {
        if system_instruction_text.chars().count() > 3000 {
            contents.insert(
                0,
                json!({
                    "role": "model",
                    "parts": [{ "text": "Understood. I will strictly follow all instructions and guidelines." }]
                }),
            );
            contents.insert(
                0,
                json!({
                    "role": "user",
                    "parts": [{ "text": system_instruction_text }]
                }),
            );
            Some(json!({
                "parts": [{ "text": "You are a helpful coding assistant." }]
            }))
        } else {
            Some(json!({
                "parts": [{ "text": system_instruction_text }]
            }))
        }
    } else {
        None
    };

    let mut gemini_req = json!({
        "contents": contents,
        "generationConfig": gen_config
    });
    if let Some(sys) = system_instruction {
        gemini_req["systemInstruction"] = sys;
    }

    if let Some(tools) = &input.tools {
        let mut decls = Vec::new();
        for t in tools {
            let mut decl = json!({ "name": t.function.name });
            if let Some(desc) = &t.function.description {
                decl["description"] = json!(desc);
            }
            if let Some(params) = &t.function.parameters {
                decl["parameters"] = sanitize_gemini_schema(params.clone());
            }
            decls.push(decl);
        }
        if !decls.is_empty() {
            gemini_req["tools"] = json!([{ "functionDeclarations": decls }]);
        }
    }

    let payload = json!({
        "project": project,
        "model": model,
        "request": gemini_req
    });

    Ok(payload)
}

#[cfg(test)]
mod tool_history_tests {
    use super::*;
    use crate::antigravity::types::{FunctionCallData, ToolCall};

    fn call(id: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: id.into(),
                tool_type: "function".into(),
                function: FunctionCallData {
                    name: "lookup".into(),
                    arguments: "{}".into(),
                },
                thought_signature: None,
            }]),
            tool_call_id: None,
        }
    }

    fn result(id: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: "tool".into(),
            content: Some(json!("ok")),
            tool_calls: None,
            tool_call_id: id.map(str::to_owned),
        }
    }

    #[test]
    fn accepts_multiple_calls_and_multiturn_results_verbatim() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some(json!("start")),
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![
                    call("call-A").tool_calls.unwrap().remove(0),
                    call("call-B").tool_calls.unwrap().remove(0),
                ]),
                tool_call_id: None,
            },
            result(Some("call-B")),
            result(Some("call-A")),
            call("call-next"),
            result(Some("call-next")),
        ];
        validate_tool_history(&messages).unwrap();
    }

    #[test]
    fn rejects_empty_duplicate_orphan_mismatched_and_unmatched_calls() {
        let cases = [
            vec![call(" "), result(Some(" "))],
            vec![call("same"), call("same"), result(Some("same"))],
            vec![result(Some("orphan"))],
            vec![call("expected"), result(Some("other"))],
            vec![call("pending")],
            vec![call("one"), result(Some("one")), result(Some("one"))],
            vec![result(None)],
        ];
        for messages in cases {
            assert!(validate_tool_history(&messages).is_err(), "{messages:?}");
        }
    }

    #[test]
    fn upstream_function_call_shape_has_no_call_id_field() {
        let part = json!({"functionCall": {"name": "lookup", "args": {}}});
        assert!(part.get("id").is_none());
    }

    #[test]
    fn test_calculate_thinking_config_low_models() {
        // max_tokens <= 100 -> budget 0, effective max unchanged
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash-low", 50);
        assert_eq!(eff, 50);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 0);

        let (eff, cfg) = calculate_thinking_config("gemini-2.5-flash-low", 100);
        assert_eq!(eff, 100);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 0);

        // 100 < max_tokens <= 500 -> budget 64, headroom +64
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash-low", 250);
        assert_eq!(eff, 314);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 64);

        // max_tokens > 500 -> budget 128, headroom +128
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash-low", 1000);
        assert_eq!(eff, 1128);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 128);

        // max_tokens == 0 or unspecified -> budget 128, effective max 0
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash-low", 0);
        assert_eq!(eff, 0);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 128);
    }

    #[test]
    fn test_calculate_thinking_config_standard_models() {
        // Standard model with small token limit (< 300) gets budget half of max and +256 headroom
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash", 200);
        assert_eq!(eff, 456);
        assert_eq!(cfg.unwrap()["thinkingBudget"], 100);

        // Standard model with large token limit (>= 300) leaves thinking unconstrained
        let (eff, cfg) = calculate_thinking_config("gemini-3.8-flash", 1000);
        assert_eq!(eff, 1000);
        assert!(cfg.is_none());

        // Non-gemini model
        let (eff, cfg) = calculate_thinking_config("claude-3-7-sonnet", 200);
        assert_eq!(eff, 200);
        assert!(cfg.is_none());
    }
}

#[cfg(test)]
mod claude_replay_tests {
    use super::*;
    use crate::antigravity::types::Credentials;

    /// Hermetic state: no credentials, no network. The reqwest client is built but never used
    /// because conversion only touches the in-memory call caches.
    fn test_state() -> AppState {
        let creds = Credentials {
            access_token: String::new(),
            refresh_token: String::new(),
            project: "test-project".to_string(),
            client_id: None,
            client_secret: None,
            email: None,
            expires_at: chrono::Utc::now(),
        };
        AppState::new(
            std::path::PathBuf::from("unused-credentials.json"),
            "test-client".to_string(),
            "test-secret".to_string(),
            creds,
        )
    }

    fn replay_request(model: &str) -> ChatRequest {
        serde_json::from_value(json!({
            "model": model,
            "messages": [
                { "role": "user", "content": "what is the weather" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_claude_replay_1",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"city\":\"Jakarta\"}" }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_claude_replay_1",
                    "content": "{\"tempC\":31}"
                }
            ]
        }))
        .expect("build test request")
    }

    fn parts_with(payload: &Value, field: &str) -> Vec<Value> {
        payload["request"]["contents"]
            .as_array()
            .expect("contents array")
            .iter()
            .filter_map(|turn| turn.get("parts").and_then(|p| p.as_array()))
            .flatten()
            .filter(|part| part.get(field).is_some())
            .cloned()
            .collect()
    }

    #[tokio::test]
    async fn claude_replay_preserves_function_call_and_response_ids() {
        let state = test_state();
        let payload = to_antigravity_request(
            &state,
            &replay_request("agy/claude-opus-5-5-high"),
            "test-project",
        )
        .await
        .expect("convert request");

        assert_eq!(payload["model"], json!("claude-opus-5-5-high"));

        // This is the shape the live defect reported: the assistant turn is contents[1] and its
        // first part must carry the id the client supplied, otherwise upstream Anthropic
        // translation fails with `messages.1.content.0.tool_use.id: Field required`.
        let contents = payload["request"]["contents"]
            .as_array()
            .expect("contents array");
        assert_eq!(contents.len(), 3, "user, model, user: {contents:?}");
        assert_eq!(contents[1]["role"], json!("model"));
        assert_eq!(contents[2]["role"], json!("user"));

        let call_id = contents[1]["parts"][0]["functionCall"]["id"]
            .as_str()
            .expect("contents[1].parts[0].functionCall.id must be present");
        assert!(
            !call_id.trim().is_empty(),
            "functionCall.id must not be empty"
        );
        assert_eq!(call_id, "call_claude_replay_1");
        assert_eq!(
            contents[1]["parts"][0]["functionCall"]["name"],
            json!("lookup")
        );

        let calls = parts_with(&payload, "functionCall");
        assert_eq!(calls.len(), 1, "one replayed call expected: {calls:?}");

        let results = parts_with(&payload, "functionResponse");
        assert_eq!(results.len(), 1, "one replayed result expected");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["id"],
            json!("call_claude_replay_1")
        );
        assert_eq!(
            results[0]["functionResponse"]["name"],
            json!("lookup"),
            "the result name comes from the cache keyed by the client id"
        );

        let serialized = serde_json::to_string(&payload).expect("serialize payload");
        assert!(
            !serialized.contains("thoughtSignature"),
            "Claude models must not receive a thoughtSignature: {serialized}"
        );
    }

    fn parallel_replay_request() -> ChatRequest {
        serde_json::from_value(json!({
            "model": "agy/claude-opus-5-5-high",
            "messages": [
                { "role": "user", "content": "check both cities" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_parallel_a",
                            "type": "function",
                            "function": { "name": "lookup", "arguments": "{\"city\":\"Jakarta\"}" }
                        },
                        {
                            "id": "call_parallel_b",
                            "type": "function",
                            "function": { "name": "lookup", "arguments": "{\"city\":\"Bandung\"}" }
                        }
                    ]
                },
                { "role": "tool", "tool_call_id": "call_parallel_a", "content": "{\"tempC\":31}" },
                { "role": "tool", "tool_call_id": "call_parallel_b", "content": "{\"tempC\":24}" }
            ]
        }))
        .expect("build parallel request")
    }

    #[tokio::test]
    async fn claude_parallel_results_merge_into_one_user_turn() {
        let state = test_state();
        let payload = to_antigravity_request(&state, &parallel_replay_request(), "test-project")
            .await
            .expect("convert request");

        let contents = payload["request"]["contents"]
            .as_array()
            .expect("contents array");
        assert_eq!(
            contents.len(),
            3,
            "two parallel results must not create two user turns: {contents:?}"
        );
        assert_eq!(contents[0]["role"], json!("user"));
        assert_eq!(contents[1]["role"], json!("model"));
        assert_eq!(contents[2]["role"], json!("user"));

        let call_parts = contents[1]["parts"].as_array().expect("call parts");
        assert_eq!(call_parts.len(), 2);
        assert_eq!(
            call_parts[0]["functionCall"]["id"],
            json!("call_parallel_a")
        );
        assert_eq!(
            call_parts[1]["functionCall"]["id"],
            json!("call_parallel_b")
        );

        // Results stay in the order the client sent them, paired by id.
        let result_parts = contents[2]["parts"].as_array().expect("result parts");
        assert_eq!(result_parts.len(), 2);
        assert_eq!(
            result_parts[0]["functionResponse"]["id"],
            json!("call_parallel_a")
        );
        assert_eq!(
            result_parts[1]["functionResponse"]["id"],
            json!("call_parallel_b")
        );

        // No thoughtSignature anywhere on the Claude path, including merged results.
        let serialized = serde_json::to_string(&payload).expect("serialize payload");
        assert!(!serialized.contains("thoughtSignature"), "{serialized}");
    }

    #[tokio::test]
    async fn claude_path_keeps_client_ids_verbatim() {
        // The outbound field is a transport detail: whatever id the client minted must survive
        // unchanged, including Anthropic-style and long base64url shapes.
        for id in [
            "toolu_01A2b3C4d5E6f7G8h9J0kLmN",
            "call_0123456789abcdefghijklmnopqrstuvwxyzABCD",
            "x",
        ] {
            let mut request = replay_request("agy/claude-opus-5-5-high");
            for message in &mut request.messages {
                if let Some(calls) = &mut message.tool_calls {
                    calls[0].id = id.to_string();
                }
                if message.tool_call_id.is_some() {
                    message.tool_call_id = Some(id.to_string());
                }
            }
            let state = test_state();
            let payload = to_antigravity_request(&state, &request, "test-project")
                .await
                .expect("convert request");
            let contents = payload["request"]["contents"]
                .as_array()
                .expect("contents array");
            assert_eq!(contents[1]["parts"][0]["functionCall"]["id"], json!(id));
            assert_eq!(contents[2]["parts"][0]["functionResponse"]["id"], json!(id));
        }
    }

    #[tokio::test]
    async fn claude_outbound_tool_parts_expose_only_the_canonical_id() {
        // Client contract: `tool_calls[].id` is the single identity. A future upstream field
        // rename stays an outbound alias and must never appear as a second identity on the
        // wire, because the response builder, the signature cache, and `validate_tool_history`
        // all key on the client-supplied id.
        let state = test_state();
        let payload = to_antigravity_request(&state, &parallel_replay_request(), "test-project")
            .await
            .expect("convert request");

        for field in ["functionCall", "functionResponse"] {
            let expected: Vec<String> = if field == "functionCall" {
                vec!["args".into(), "id".into(), "name".into()]
            } else {
                vec!["id".into(), "name".into(), "response".into()]
            };
            let parts = parts_with(&payload, field);
            assert_eq!(parts.len(), 2, "two {field} parts expected: {parts:?}");
            for part in parts {
                let mut keys: Vec<String> = part[field]
                    .as_object()
                    .expect("nested tool object")
                    .keys()
                    .cloned()
                    .collect();
                keys.sort();
                assert_eq!(keys, expected, "{field} carries unexpected keys: {part}");
            }
        }

        let serialized = serde_json::to_string(&payload).expect("serialize payload");
        for alias in ["callId", "toolCallId", "tool_use_id", "toolUseId"] {
            assert!(
                !serialized.contains(alias),
                "{alias} must stay an outbound-only concern: {serialized}"
            );
        }
    }

    #[tokio::test]
    async fn claude_result_merge_stops_at_a_text_turn() {
        // Explicit merge boundary: only consecutive `tool` results fold into one user turn.
        // A later user message keeps its own turn. Anthropic combines consecutive same-role
        // turns upstream, so preserving client order is safe and avoids reordering text into a
        // tool_result block.
        let state = test_state();
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "agy/claude-opus-5-5-high",
            "messages": [
                { "role": "user", "content": "weather?" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_merge_boundary",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_merge_boundary", "content": "31" },
                { "role": "user", "content": "and tomorrow?" }
            ]
        }))
        .expect("build merge boundary request");

        let payload = to_antigravity_request(&state, &request, "test-project")
            .await
            .expect("convert request");
        let contents = payload["request"]["contents"]
            .as_array()
            .expect("contents array");
        assert_eq!(
            contents.len(),
            4,
            "text must not be folded into the result turn: {contents:?}"
        );
        assert_eq!(contents[2]["role"], json!("user"));
        let result_parts = contents[2]["parts"].as_array().expect("result parts");
        assert_eq!(result_parts.len(), 1);
        assert!(result_parts[0].get("functionResponse").is_some());
        assert!(
            result_parts[0].get("text").is_none(),
            "only functionResponse parts may merge: {result_parts:?}"
        );
        assert_eq!(contents[3]["parts"][0]["text"], json!("and tomorrow?"));
    }
}

#[cfg(test)]
mod model_family_tests {
    use super::*;

    #[test]
    fn claude_detection_covers_prefixes_case_and_unknown_families() {
        let cases = [
            ("claude-opus-5-5-high", true),
            ("agy/claude-opus-5-5-high", true),
            ("CLAUDE-SONNET-5-5", true),
            ("claude-3-7-sonnet", true),
            ("gemini-3.8-flash", false),
            ("agy/gemini-3.8-pro", false),
            ("default", false),
            ("gpt-5.6-luna", false),
            ("", false),
        ];
        for (model, expected) in cases {
            assert_eq!(
                is_claude_model(&normalize_model_id(model)),
                expected,
                "model {model}"
            );
        }
    }

    #[test]
    fn normalization_strips_provider_prefix_and_defaults() {
        assert_eq!(
            normalize_model_id("agy/claude-opus-5-5-high"),
            "claude-opus-5-5-high"
        );
        assert_eq!(
            normalize_model_id(" claude-opus-5-5-high "),
            "claude-opus-5-5-high"
        );
        assert_eq!(normalize_model_id("default"), "gemini-3.8-flash-tiered");
        assert_eq!(normalize_model_id(""), "gemini-3.8-flash-tiered");
    }
}

#[cfg(test)]
mod gemini_replay_isolation_tests {
    use super::*;
    use crate::antigravity::types::Credentials;

    fn test_state() -> AppState {
        let creds = Credentials {
            access_token: String::new(),
            refresh_token: String::new(),
            project: "test-project".to_string(),
            client_id: None,
            client_secret: None,
            email: None,
            expires_at: chrono::Utc::now(),
        };
        AppState::new(
            std::path::PathBuf::from("unused-credentials.json"),
            "test-client".to_string(),
            "test-secret".to_string(),
            creds,
        )
    }

    fn parallel_replay_request(model: &str) -> ChatRequest {
        serde_json::from_value(json!({
            "model": model,
            "messages": [
                { "role": "user", "content": "check both cities" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_gemini_a",
                            "type": "function",
                            "function": { "name": "lookup", "arguments": "{\"city\":\"Jakarta\"}" }
                        },
                        {
                            "id": "call_gemini_b",
                            "type": "function",
                            "function": { "name": "lookup", "arguments": "{\"city\":\"Bandung\"}" }
                        }
                    ]
                },
                { "role": "tool", "tool_call_id": "call_gemini_a", "content": "{\"tempC\":31}" },
                { "role": "tool", "tool_call_id": "call_gemini_b", "content": "{\"tempC\":24}" }
            ]
        }))
        .expect("build test request")
    }

    #[tokio::test]
    async fn gemini_replay_keeps_structured_parts_and_thought_signature() {
        let state = test_state();
        let payload = to_antigravity_request(
            &state,
            &parallel_replay_request("agy/gemini-3.8-flash"),
            "test-project",
        )
        .await
        .expect("convert request");

        assert_eq!(payload["model"], json!("gemini-3.8-flash"));

        let turns = payload["request"]["contents"].as_array().expect("contents");
        let model_turn = turns
            .iter()
            .find(|turn| turn["role"] == json!("model"))
            .expect("model turn");
        let call_parts = model_turn["parts"].as_array().expect("call parts");
        assert_eq!(call_parts.len(), 2);

        // Gemini must not receive the Claude-only identity field.
        for part in call_parts {
            assert!(
                part["functionCall"].get("id").is_none(),
                "Gemini functionCall must stay ID-free: {part}"
            );
        }

        // Only the first functionCall part carries the signature on the Gemini path.
        assert_eq!(
            call_parts[0]["thoughtSignature"],
            json!(crate::antigravity::types::SKIP_THOUGHT_SIGNATURE)
        );
        assert!(call_parts[1].get("thoughtSignature").is_none());

        let result_parts: Vec<&Value> = turns
            .iter()
            .filter(|turn| turn["role"] == json!("user"))
            .filter_map(|turn| turn["parts"].as_array())
            .flatten()
            .filter(|part| part.get("functionResponse").is_some())
            .collect();
        assert_eq!(result_parts.len(), 2);
        for part in result_parts {
            assert!(part.get("thoughtSignature").is_some());
            assert!(part["functionResponse"].get("id").is_none());
        }
    }

    fn replay_with_tools_request() -> ChatRequest {
        serde_json::from_value(json!({
            "model": "agy/gemini-3.8-flash",
            "messages": [
                { "role": "user", "content": "what is the weather" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_gemini_golden",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"city\":\"Jakarta\"}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_gemini_golden", "content": "{\"tempC\":31}" }
            ],
            "temperature": 0.2,
            "max_tokens": 300,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }]
        }))
        .expect("build golden replay request")
    }

    #[tokio::test]
    async fn gemini_replay_with_tools_payload_snapshot() {
        let state = test_state();
        let payload = to_antigravity_request(&state, &replay_with_tools_request(), "test-project")
            .await
            .expect("convert request");

        // Locked snapshot: the Claude work must not alter the Gemini wire payload by a byte.
        let expected = json!({
            "project": "test-project",
            "model": "gemini-3.8-flash",
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "what is the weather" }] },
                    {
                        "role": "model",
                        "parts": [{
                            "functionCall": { "name": "lookup", "args": { "city": "Jakarta" } },
                            "thoughtSignature": crate::antigravity::types::SKIP_THOUGHT_SIGNATURE
                        }]
                    },
                    {
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": "lookup",
                                "response": { "result": { "tempC": 31 } }
                            },
                            // Result parts fall back to the long default signature when the call
                            // carried no signature into the cache. Pre-existing Gemini behaviour.
                            "thoughtSignature": crate::antigravity::types::DEFAULT_THOUGHT_SIGNATURE
                        }]
                    }
                ],
                "generationConfig": { "temperature": 0.2, "maxOutputTokens": 300 },
                "tools": [{
                    "functionDeclarations": [{
                        "name": "lookup",
                        "description": "Weather by city",
                        "parameters": {
                            "type": "object",
                            "properties": { "city": { "type": "string" } },
                            "required": ["city"]
                        }
                    }]
                }]
            }
        });
        assert_eq!(payload, expected);
    }

    #[tokio::test]
    async fn gemini_plain_chat_payload_snapshot() {
        let state = test_state();
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "agy/gemini-3.8-flash",
            "messages": [
                { "role": "system", "content": "be terse" },
                { "role": "user", "content": "hi" }
            ],
            "max_tokens": 200
        }))
        .expect("build golden plain request");

        let payload = to_antigravity_request(&state, &request, "test-project")
            .await
            .expect("convert request");

        let expected = json!({
            "project": "test-project",
            "model": "gemini-3.8-flash",
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "hi" }] }
                ],
                "generationConfig": {
                    "maxOutputTokens": 456,
                    "thinkingConfig": { "thinkingBudget": 100 }
                },
                "systemInstruction": { "parts": [{ "text": "be terse" }] }
            }
        });
        assert_eq!(payload, expected);
    }
}

#[cfg(test)]
mod request_trace_tests {
    use super::*;

    fn sample_payload() -> Value {
        json!({
            "project": "sample-project",
            "model": "claude-opus-5-5-high",
            "request": {
                "contents": [
                    { "role": "user", "parts": [{ "text": "TOPSECRET-PROMPT-TEXT" }] },
                    {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "lookup",
                                "args": { "city": "TOPSECRET-ARG-VALUE" },
                                "id": "call_trace_1"
                            }
                        }]
                    },
                    {
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": "lookup",
                                "response": { "result": { "note": "TOPSECRET-RESULT-VALUE" } },
                                "id": "call_trace_1"
                            }
                        }]
                    }
                ],
                "systemInstruction": { "parts": [{ "text": "TOPSECRET-SYSTEM-TEXT" }] },
                "tools": [{
                    "functionDeclarations": [{
                        "name": "lookup",
                        "description": "TOPSECRET-TOOL-DESCRIPTION"
                    }]
                }],
                "generationConfig": { "temperature": 0.2 }
            }
        })
    }

    #[test]
    fn redacted_trace_keeps_structure_and_ids_but_drops_content() {
        let trace = redacted_request_trace(&sample_payload());
        let rendered = serde_json::to_string(&trace).expect("serialize trace");

        for secret in [
            "TOPSECRET-PROMPT-TEXT",
            "TOPSECRET-ARG-VALUE",
            "TOPSECRET-RESULT-VALUE",
            "TOPSECRET-SYSTEM-TEXT",
            "TOPSECRET-TOOL-DESCRIPTION",
        ] {
            assert!(!rendered.contains(secret), "{secret} leaked into the trace");
        }

        assert!(rendered.contains("call_trace_1"), "IDs must survive");
        assert!(rendered.contains("functionCall"));
        assert!(rendered.contains("functionResponse"));
        assert_eq!(trace["contents"][0]["role"], json!("user"));
        assert_eq!(
            trace["contents"][1]["parts"][0]["functionCall"]["args"]["city"],
            json!("<redacted>")
        );
        assert_eq!(trace["generationConfig"]["temperature"], json!(0.2));
    }

    #[test]
    fn trace_gate_is_off_when_the_environment_variable_is_absent() {
        if std::env::var(TRACE_REQUEST_ENV).is_err() {
            assert!(!trace_request_enabled());
        }
    }
}
