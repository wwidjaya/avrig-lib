use crate::antigravity::converter::{
    finish_reason, random_id, to_antigravity_request, trace_antigravity_request,
};
use crate::antigravity::state::AppState as AntigravityAppState;
use crate::antigravity::types::{
    ChatChoice, ChatChoiceMessage, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionChunkDelta, ChatRequest, ChatResponse, FunctionCallChunkData, FunctionCallData,
    OpenAIUsage, ToolCall, ToolCallChunk,
};
use crate::resolver::{is_siwc_model, resolve_upstream_model_id};
use crate::siwc::client::SiwcClient;
use crate::types::{AccountStatus, ModelInfo};
use anyhow::{Result, anyhow};
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AntigravityAccount {
    pub alias: String,
    pub state: AntigravityAppState,
    pub priority: u32,
}

#[derive(Clone)]
pub struct SiwcAccount {
    pub alias: String,
    pub client: SiwcClient,
    pub priority: u32,
}

#[derive(Clone)]
pub struct DirectProviderAccount {
    pub provider_name: String,
    pub base_url: String,
    pub api_keys: Vec<String>,
    pub next_key_idx: Arc<AtomicUsize>,
}

#[derive(Clone, Default)]
pub struct LlmProviderClient {
    pub antigravity_accounts: Arc<RwLock<Vec<AntigravityAccount>>>,
    pub next_ag_idx: Arc<AtomicUsize>,
    pub siwc_accounts: Arc<RwLock<Vec<SiwcAccount>>>,
    pub next_siwc_idx: Arc<AtomicUsize>,
    pub direct_providers: Arc<RwLock<HashMap<String, DirectProviderAccount>>>,
    pub http: reqwest::Client,
}

impl LlmProviderClient {
    pub fn new() -> Self {
        Self {
            antigravity_accounts: Arc::new(RwLock::new(Vec::new())),
            next_ag_idx: Arc::new(AtomicUsize::new(0)),
            siwc_accounts: Arc::new(RwLock::new(Vec::new())),
            next_siwc_idx: Arc::new(AtomicUsize::new(0)),
            direct_providers: Arc::new(RwLock::new(HashMap::new())),
            http: reqwest::Client::new(),
        }
    }

    pub async fn add_antigravity_account(
        &self,
        alias: String,
        state: AntigravityAppState,
        priority: u32,
    ) {
        let mut list = self.antigravity_accounts.write().await;
        list.retain(|a| a.alias != alias);
        list.push(AntigravityAccount {
            alias,
            state,
            priority,
        });
        list.sort_by_key(|a| a.priority);
    }

    pub async fn add_siwc_account(&self, alias: String, client: SiwcClient, priority: u32) {
        let mut list = self.siwc_accounts.write().await;
        list.retain(|a| a.alias != alias);
        list.push(SiwcAccount {
            alias,
            client,
            priority,
        });
        list.sort_by_key(|a| a.priority);
    }

    pub async fn add_direct_provider(
        &self,
        provider_name: String,
        base_url: String,
        api_keys: Vec<String>,
    ) {
        let mut map = self.direct_providers.write().await;
        map.insert(
            provider_name.clone(),
            DirectProviderAccount {
                provider_name,
                base_url,
                api_keys,
                next_key_idx: Arc::new(AtomicUsize::new(0)),
            },
        );
    }

    pub async fn list_accounts(&self) -> Vec<AccountStatus> {
        let mut statuses = Vec::new();

        let ags = self.antigravity_accounts.read().await;
        for a in ags.iter() {
            let creds = a.state.creds.read().await;
            statuses.push(AccountStatus {
                provider: "google-antigravity".into(),
                alias: a.alias.clone(),
                email_or_id: creds.email.clone().unwrap_or_else(|| creds.project.clone()),
                enabled: true,
                healthy: !creds.refresh_token.is_empty(),
                expires_at: Some(creds.expires_at.to_rfc3339()),
            });
        }

        let siwcs = self.siwc_accounts.read().await;
        for s in siwcs.iter() {
            let creds = s.client.creds.read().await;
            statuses.push(AccountStatus {
                provider: "openai-siwc".into(),
                alias: s.alias.clone(),
                email_or_id: creds.client_id.clone(),
                enabled: true,
                healthy: creds.refresh_token.is_some(),
                expires_at: Some(creds.expires_at.to_rfc3339()),
            });
        }

        let directs = self.direct_providers.read().await;
        for (name, d) in directs.iter() {
            statuses.push(AccountStatus {
                provider: name.clone(),
                alias: name.clone(),
                email_or_id: format!("{} key(s)", d.api_keys.len()),
                enabled: true,
                healthy: !d.api_keys.is_empty(),
                expires_at: None,
            });
        }

        statuses
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        // 1. Antigravity models
        let ags = self.antigravity_accounts.read().await;
        if let Some(first_ag) = ags.first() {
            if let Ok(Ok(ag_models)) = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                crate::antigravity::server_fetch_models(&first_ag.state),
            )
            .await
            {
                for m in ag_models {
                    models.push(ModelInfo {
                        id: format!("agy/{}", m),
                        object: "model".into(),
                        created: 0,
                        owned_by: "google-antigravity".into(),
                    });
                }
            }
        }

        // 2. SIWC models
        let siwcs = self.siwc_accounts.read().await;
        if let Some(first_siwc) = siwcs.first() {
            if let Ok(Ok(siwc_models)) = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                first_siwc.client.available_models(),
            )
            .await
            {
                for m in siwc_models {
                    models.push(ModelInfo {
                        id: format!("chatgpt/{}", m),
                        object: "model".into(),
                        created: 0,
                        owned_by: "openai-siwc".into(),
                    });
                }
            }
        }

        // 3. Direct providers (e.g. OpenCode, Groq, Ollama)
        let directs = self.direct_providers.read().await;
        for (name, direct) in directs.iter() {
            if let Some(key) = direct.api_keys.first() {
                let models_url = format!("{}/models", direct.base_url.trim_end_matches('/'));
                let mut req = self.http.get(&models_url);
                if !key.is_empty() {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }
                if let Ok(Ok(resp)) =
                    tokio::time::timeout(std::time::Duration::from_secs(5), req.send()).await
                {
                    if let Ok(json_val) = resp.json::<Value>().await {
                        if let Some(arr) = json_val.get("data").and_then(|d| d.as_array()) {
                            for item in arr {
                                if let Some(id_str) = item.get("id").and_then(|id| id.as_str()) {
                                    models.push(ModelInfo {
                                        id: format!("{}/{}", name, id_str),
                                        object: "model".into(),
                                        created: item
                                            .get("created")
                                            .and_then(|c| c.as_i64())
                                            .unwrap_or(0),
                                        owned_by: name.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    pub async fn chat_completion(&self, mut input: ChatRequest) -> Result<ChatResponse> {
        let original_model = input.model.clone();
        let (prefix, base_model) = crate::resolver::strip_model_prefix(&original_model);
        input.model = resolve_upstream_model_id(&original_model);

        // Check if explicitly matching a direct provider (e.g. opencode/gpt-5.6-luna)
        if let Some(p) = prefix {
            let directs = self.direct_providers.read().await;
            if let Some(direct) = directs.get(p) {
                input.model = base_model;
                return execute_direct_provider_non_stream(&self.http, direct, &input).await;
            }
        }

        if is_siwc_model(&original_model) {
            let siwcs = self.siwc_accounts.read().await;
            if siwcs.is_empty() {
                return Err(anyhow!("No SIWC accounts available"));
            }
            let idx = self.next_siwc_idx.fetch_add(1, Ordering::Relaxed) % siwcs.len();
            let account = &siwcs[idx];
            return execute_siwc_non_stream(&account.client, &input).await;
        }

        let ags = self.antigravity_accounts.read().await;
        if ags.is_empty() {
            // Check direct providers fallback if model matches
            let directs = self.direct_providers.read().await;
            if let Some(direct) = directs.values().next() {
                return execute_direct_provider_non_stream(&self.http, direct, &input).await;
            }
            return Err(anyhow!("No Antigravity accounts available"));
        }
        let idx = self.next_ag_idx.fetch_add(1, Ordering::Relaxed) % ags.len();
        let account = &ags[idx];
        execute_antigravity_non_stream(&account.state, &input).await
    }

    pub async fn chat_completion_stream(
        &self,
        mut input: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        let original_model = input.model.clone();
        let (prefix, base_model) = crate::resolver::strip_model_prefix(&original_model);
        input.model = resolve_upstream_model_id(&original_model);

        if let Some(p) = prefix {
            let directs = self.direct_providers.read().await;
            if let Some(direct) = directs.get(p) {
                input.model = base_model;
                return execute_direct_provider_stream(&self.http, direct, &input).await;
            }
        }

        if is_siwc_model(&original_model) {
            let siwcs = self.siwc_accounts.read().await;
            if siwcs.is_empty() {
                return Err(anyhow!("No SIWC accounts available"));
            }
            let idx = self.next_siwc_idx.fetch_add(1, Ordering::Relaxed) % siwcs.len();
            let account = &siwcs[idx];
            return execute_siwc_stream(&account.client, &input).await;
        }

        let ags = self.antigravity_accounts.read().await;
        if ags.is_empty() {
            let directs = self.direct_providers.read().await;
            if let Some(direct) = directs.values().next() {
                return execute_direct_provider_stream(&self.http, direct, &input).await;
            }
            return Err(anyhow!("No Antigravity accounts available"));
        }
        let idx = self.next_ag_idx.fetch_add(1, Ordering::Relaxed) % ags.len();
        let account = &ags[idx];
        execute_antigravity_stream(&account.state, &input).await
    }
}

// Helpers for model fetching & non-stream / stream execution

/// Resolve the client-facing tool call id from an upstream Antigravity part.
///
/// The provider-returned `id` is the only identity source for the OpenAI-facing
/// `tool_calls[].id`. Request-side transport aliases (whatever field name the Claude replay
/// path settles on) are never consulted here, so a renamed outbound field cannot leak into
/// response ids, the signature cache, or tool history validation.
pub fn upstream_tool_call_id(part: &Value) -> String {
    part.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_{}", random_id()))
}

/// Resolve the OpenAI `finish_reason` for a completed Antigravity response.
///
/// Upstream reports `STOP` on a tool turn, which makes a client treat the reply as a final
/// answer and stops the tool loop. A response carrying tool calls therefore always reports
/// `tool_calls`, matching the streaming path, which applies the same rule.
pub fn resolved_finish_reason(upstream: Option<&str>, has_tool_calls: bool) -> String {
    if has_tool_calls {
        "tool_calls".to_string()
    } else {
        finish_reason(upstream.unwrap_or("STOP"))
    }
}

pub async fn execute_antigravity_non_stream(
    state: &AntigravityAppState,
    input: &ChatRequest,
) -> Result<ChatResponse> {
    crate::antigravity::converter::validate_tool_history(&input.messages)?;
    if let Err(e) = state.refresh_token().await {
        return Err(anyhow!("refresh token: {}", e));
    }
    let project = {
        let creds = state.creds.read().await;
        creds.project.clone()
    };
    let payload = to_antigravity_request(state, input, &project).await?;
    trace_antigravity_request(&payload);
    let resp_val = state
        .google_post("/v1internal:generateContent", &payload)
        .await?;

    let resp_obj = match resp_val.as_object() {
        Some(obj) => {
            if let Some(inner) = obj.get("response").and_then(|r| r.as_object()) {
                inner
            } else {
                obj
            }
        }
        None => return Err(anyhow!("response is not an object: {}", resp_val)),
    };
    let candidates = match resp_obj.get("candidates").and_then(|v| v.as_array()) {
        Some(arr) if !arr.is_empty() => arr,
        _ => return Err(anyhow!("Google Antigravity upstream error: {}", resp_val)),
    };
    let first = &candidates[0];

    let parts = first
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array());
    let mut text_out = String::new();
    let mut reasoning_out = String::new();
    let mut tool_calls = Vec::new();

    if let Some(parts_arr) = parts {
        for part in parts_arr {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                if part
                    .get("thought")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false)
                {
                    reasoning_out.push_str(t);
                } else {
                    text_out.push_str(t);
                }
            }
            if let Some(fc) = part.get("functionCall") {
                let call_name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let call_args = fc.get("args").cloned().unwrap_or(json!({}));
                let thought_sig = part
                    .get("thoughtSignature")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());
                let call_id = upstream_tool_call_id(part);
                tool_calls.push(ToolCall {
                    id: call_id,
                    tool_type: "function".to_string(),
                    function: FunctionCallData {
                        name: call_name.to_string(),
                        arguments: call_args.to_string(),
                    },
                    thought_signature: thought_sig,
                });
            }
        }
    }

    let finish = first.get("finishReason").and_then(|f| f.as_str());
    let open_finish = resolved_finish_reason(finish, !tool_calls.is_empty());

    let usage_val = resp_obj.get("usageMetadata");
    let (pt, ct, tt) = if let Some(u) = usage_val {
        let p = u
            .get("promptTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let c = u
            .get("candidatesTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let t = u
            .get("totalTokenCount")
            .and_then(|v| v.as_i64())
            .unwrap_or(p + c);
        (p, c, t)
    } else {
        (0, 0, 0)
    };

    Ok(ChatResponse {
        id: format!("chatcmpl-{}", random_id()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: input.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatChoiceMessage {
                role: "assistant".to_string(),
                content: if text_out.is_empty() {
                    None
                } else {
                    Some(text_out)
                },
                reasoning_content: if reasoning_out.is_empty() {
                    None
                } else {
                    Some(reasoning_out)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: open_finish,
        }],
        usage: OpenAIUsage {
            prompt_tokens: pt,
            completion_tokens: ct,
            total_tokens: tt,
        },
    })
}

pub fn sse_data_stream<S>(byte_stream: S) -> impl Stream<Item = Result<String>>
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    let mut buffer = String::new();
    byte_stream
        .map(move |chunk_res| -> Vec<Result<String>> {
            let bytes = match chunk_res {
                Ok(b) => b,
                Err(e) => return vec![Err(anyhow!("stream read error: {}", e))],
            };
            let text = String::from_utf8_lossy(&bytes);
            buffer.push_str(&text);
            let mut payloads = Vec::new();

            while let Some(newline_pos) = buffer.find('\n') {
                let line: String = buffer.drain(..=newline_pos).collect();
                let trimmed = line.trim();
                if let Some(data) = trimmed.strip_prefix("data:") {
                    let data = data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    payloads.push(Ok(data.to_string()));
                }
            }
            payloads
        })
        .flat_map(futures_util::stream::iter)
}

pub fn latest_usage_stream<S>(
    stream: S,
    id: String,
    model: String,
    latest_usage: Arc<tokio::sync::RwLock<Option<OpenAIUsage>>>,
    include_usage: bool,
) -> impl Stream<Item = Result<ChatCompletionChunk>>
where
    S: Stream<Item = Result<ChatCompletionChunk>> + Send + 'static,
{
    let usage_drain = futures_util::stream::once(async move {
        if !include_usage {
            return None;
        }
        let lock = latest_usage.read().await;
        lock.clone()
            .map(|usage| Ok(ChatCompletionChunk::usage(&id, &model, usage)))
    })
    .filter_map(|x| async move { x });

    stream.chain(usage_drain)
}

fn is_valid_usage(usage: &OpenAIUsage) -> bool {
    usage.prompt_tokens >= 0 && usage.completion_tokens >= 0 && usage.total_tokens >= 0
}

fn requests_usage(input: &ChatRequest) -> bool {
    input
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        == Some(true)
}

pub async fn execute_antigravity_stream(
    state: &AntigravityAppState,
    input: &ChatRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
    crate::antigravity::converter::validate_tool_history(&input.messages)?;
    if let Err(e) = state.refresh_token().await {
        return Err(anyhow!("refresh token: {}", e));
    }
    let project = {
        let creds = state.creds.read().await;
        creds.project.clone()
    };
    let payload = to_antigravity_request(state, input, &project).await?;
    trace_antigravity_request(&payload);

    let access_token = {
        let creds = state.creds.read().await;
        creds.access_token.clone()
    };
    let url = format!(
        "{}/v1internal:streamGenerateContent?alt=sse",
        crate::antigravity::types::GATEWAY_URL
    );

    let resp = state
        .http
        .post(&url)
        .header("Authorization", format!("Bearer {}", access_token))
        .header("User-Agent", crate::antigravity::types::HUB_USER_AGENT)
        .header("Accept", "text/event-stream")
        .json(&payload)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Upstream HTTP {}: {}", status, text));
    }

    let completion_id = format!("chatcmpl-{}", random_id());
    let usage_id = completion_id.clone();
    let model = input.model.clone();
    let data_stream = sse_data_stream(resp.bytes_stream());
    let latest_usage = Arc::new(tokio::sync::RwLock::new(Option::<OpenAIUsage>::None));

    let usage_drain_ref = latest_usage.clone();
    let mapped = data_stream.filter_map(move |payload_res| {
        let completion_id = completion_id.clone();
        let model = model.clone();
        let latest_usage = latest_usage.clone();
        async move {
            let data = match payload_res {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            };

            if let Ok(v) = serde_json::from_str::<Value>(&data) {
                let usage_meta = v
                    .get("usageMetadata")
                    .or_else(|| v.get("response").and_then(|r| r.get("usageMetadata")));
                if let Some(u) = usage_meta.and_then(|u| {
                    serde_json::from_value::<crate::antigravity::types::GoogleUsage>(u.clone()).ok()
                }) {
                    let usage = u.to_openai();
                    if is_valid_usage(&usage) {
                        let mut lock = latest_usage.write().await;
                        // Latest valid usage event wins. Invalid later events do not erase it.
                        *lock = Some(usage);
                    }
                }

                let candidates = v
                    .get("response")
                    .and_then(|r| r.get("candidates"))
                    .or_else(|| v.get("candidates"))
                    .and_then(|c| c.as_array());

                if let Some(candidates) = candidates {
                    if let Some(first) = candidates.first() {
                        let mut delta_content = String::new();
                        let mut delta_reasoning = String::new();
                        let mut tool_chunks = Vec::new();

                        if let Some(parts) = first
                            .get("content")
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                        {
                            for part in parts {
                                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                    if part
                                        .get("thought")
                                        .and_then(|b| b.as_bool())
                                        .unwrap_or(false)
                                    {
                                        delta_reasoning.push_str(t);
                                    } else {
                                        delta_content.push_str(t);
                                    }
                                }
                                if let Some(fc) = part.get("functionCall") {
                                    let name = fc
                                        .get("name")
                                        .and_then(|n| n.as_str())
                                        .map(|s| s.to_string());
                                    let args = fc.get("args").map(|a| a.to_string());
                                    let call_id = upstream_tool_call_id(part);
                                    tool_chunks.push(ToolCallChunk {
                                        index: 0,
                                        id: Some(call_id),
                                        tool_type: Some("function".to_string()),
                                        function: FunctionCallChunkData {
                                            name,
                                            arguments: args,
                                        },
                                    });
                                }
                            }
                        }

                        let finish = first
                            .get("finishReason")
                            .and_then(|f| f.as_str())
                            .map(|f| resolved_finish_reason(Some(f), !tool_chunks.is_empty()));
                        return Some(Ok(ChatCompletionChunk {
                            id: completion_id,
                            object: "chat.completion.chunk".into(),
                            created: chrono::Utc::now().timestamp(),
                            model,
                            choices: vec![ChatCompletionChunkChoice {
                                index: 0,
                                delta: ChatCompletionChunkDelta {
                                    role: Some("assistant".to_string()),
                                    content: if delta_content.is_empty() {
                                        None
                                    } else {
                                        Some(delta_content)
                                    },
                                    reasoning_content: if delta_reasoning.is_empty() {
                                        None
                                    } else {
                                        Some(delta_reasoning)
                                    },
                                    tool_calls: if tool_chunks.is_empty() {
                                        None
                                    } else {
                                        Some(tool_chunks)
                                    },
                                },
                                finish_reason: finish,
                            }],
                            usage: None,
                        }));
                    }
                }
            }
            None
        }
    });

    let m = input.model.clone();
    let usage_drain =
        latest_usage_stream(mapped, usage_id, m, usage_drain_ref, requests_usage(input));

    Ok(Box::pin(usage_drain))
}

pub async fn execute_siwc_non_stream(
    client: &SiwcClient,
    input: &ChatRequest,
) -> Result<ChatResponse> {
    let raw_resp = client.chat_completions_raw(input).await?;
    let body_text = raw_resp.text().await?;
    // Collect SSE stream into single completion
    let mut full_text = String::new();
    let mut finish_reason_val = "stop".to_string();

    for line in body_text.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            if let Ok(val) = serde_json::from_str::<Value>(data) {
                if let Some(err_obj) = val.get("error") {
                    let msg = err_obj
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown upstream error");
                    return Err(anyhow!("SIWC upstream error: {}", msg));
                }
                if let Some(delta) = val.get("delta").and_then(|d| d.as_str()) {
                    full_text.push_str(delta);
                } else if let Some(text) = val.get("text").and_then(|t| t.as_str()) {
                    full_text.push_str(text);
                }
                if let Some(fr) = val.get("finish_reason").and_then(|f| f.as_str()) {
                    finish_reason_val = fr.to_string();
                }
            }
        }
    }

    Ok(ChatResponse {
        id: format!("chatcmpl-{}", random_id()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: input.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatChoiceMessage {
                role: "assistant".to_string(),
                content: Some(full_text),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: finish_reason_val,
        }],
        usage: OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}

pub async fn execute_siwc_stream(
    client: &SiwcClient,
    input: &ChatRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
    let raw_resp = client.chat_completions_raw(input).await?;
    let completion_id = format!("chatcmpl-{}", random_id());
    let usage_id = completion_id.clone();
    let model = input.model.clone();

    let initial_chunk = ChatCompletionChunk::initial_role(&completion_id, &model, "assistant");
    let data_stream = sse_data_stream(raw_resp.bytes_stream());

    let had_tool_calls = Arc::new(AtomicBool::new(false));
    let next_tool_index = Arc::new(AtomicUsize::new(0));
    let tool_index_map = Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::<
        usize,
        (String, String, usize),
    >::new()));
    let latest_usage = Arc::new(tokio::sync::RwLock::new(Option::<OpenAIUsage>::None));

    let siwc_usage_drain_ref = latest_usage.clone();
    let mapped = data_stream.filter_map(move |payload_res| {
        let completion_id = completion_id.clone();
        let model = model.clone();
        let had_tool_calls = had_tool_calls.clone();
        let next_tool_index = next_tool_index.clone();
        let tool_index_map = tool_index_map.clone();
        let latest_usage = latest_usage.clone();

        async move {
            let data = match payload_res {
                Ok(d) => d,
                Err(e) => return Some(Err(e)),
            };

            if let Ok(val) = serde_json::from_str::<Value>(&data) {
                let event_type = val.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match event_type {
                    "error" => {
                        let err_msg = val
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown upstream SIWC error");
                        return Some(Err(anyhow!("SIWC error: {}", err_msg)));
                    }
                    "response.failed" => {
                        let err_msg = val
                            .get("response")
                            .and_then(|r| r.get("error"))
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("SIWC response failed");
                        return Some(Err(anyhow!("SIWC error: {}", err_msg)));
                    }
                    "response.output_text.delta" => {
                        if let Some(delta) = val.get("delta").and_then(|d| d.as_str()) {
                            return Some(Ok(ChatCompletionChunk::text_delta(
                                &completion_id,
                                &model,
                                delta,
                            )));
                        }
                    }
                    "response.output_item.added" => {
                        if let Some(item) = val.get("item") {
                            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            if item_type == "function_call" {
                                had_tool_calls.store(true, Ordering::Relaxed);
                                let out_idx =
                                    val.get("output_index")
                                        .and_then(|i| i.as_u64())
                                        .unwrap_or(0) as usize;
                                let call_id = item
                                    .get("call_id")
                                    .and_then(|i| i.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = item
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                let client_index = next_tool_index.fetch_add(1, Ordering::Relaxed);
                                {
                                    let mut map = tool_index_map.write().await;
                                    map.insert(
                                        out_idx,
                                        (call_id.clone(), name.clone(), client_index),
                                    );
                                }

                                return Some(Ok(ChatCompletionChunk {
                                    id: completion_id,
                                    object: "chat.completion.chunk".into(),
                                    created: chrono::Utc::now().timestamp(),
                                    model,
                                    choices: vec![ChatCompletionChunkChoice {
                                        index: 0,
                                        delta: ChatCompletionChunkDelta {
                                            role: Some("assistant".to_string()),
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: Some(vec![ToolCallChunk {
                                                index: client_index,
                                                id: Some(call_id),
                                                tool_type: Some("function".to_string()),
                                                function: FunctionCallChunkData {
                                                    name: Some(name),
                                                    arguments: Some(String::new()),
                                                },
                                            }]),
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                }));
                            }
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        let out_idx = val
                            .get("output_index")
                            .and_then(|i| i.as_u64())
                            .unwrap_or(0) as usize;
                        if let Some(delta) = val.get("delta").and_then(|d| d.as_str()) {
                            let client_index = {
                                let map = tool_index_map.read().await;
                                map.get(&out_idx).map(|(_, _, idx)| *idx).unwrap_or(0)
                            };
                            return Some(Ok(ChatCompletionChunk {
                                id: completion_id,
                                object: "chat.completion.chunk".into(),
                                created: chrono::Utc::now().timestamp(),
                                model,
                                choices: vec![ChatCompletionChunkChoice {
                                    index: 0,
                                    delta: ChatCompletionChunkDelta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(vec![ToolCallChunk {
                                            index: client_index,
                                            id: None,
                                            tool_type: None,
                                            function: FunctionCallChunkData {
                                                name: None,
                                                arguments: Some(delta.to_string()),
                                            },
                                        }]),
                                    },
                                    finish_reason: None,
                                }],
                                usage: None,
                            }));
                        }
                    }
                    "response.completed" => {
                        if let Some(resp_obj) = val.get("response") {
                            if let Some(u) = resp_obj.get("usage") {
                                let pt =
                                    u.get("input_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                                let ct =
                                    u.get("output_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
                                let tt = u
                                    .get("total_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(pt + ct);
                                let usage = OpenAIUsage {
                                    prompt_tokens: pt,
                                    completion_tokens: ct,
                                    total_tokens: tt,
                                };
                                if is_valid_usage(&usage) {
                                    let mut lock = latest_usage.write().await;
                                    // The last valid completed-response usage is authoritative.
                                    *lock = Some(usage);
                                }
                            }
                        }
                        let reason = if had_tool_calls.load(Ordering::Relaxed) {
                            "tool_calls"
                        } else {
                            "stop"
                        };
                        return Some(Ok(ChatCompletionChunk::finish(
                            &completion_id,
                            &model,
                            reason,
                        )));
                    }
                    _ => {}
                }
            }
            None
        }
    });

    let m = input.model.clone();
    let usage_drain = latest_usage_stream(
        mapped,
        usage_id,
        m,
        siwc_usage_drain_ref,
        requests_usage(input),
    );

    let full_stream =
        futures_util::stream::once(async move { Ok(initial_chunk) }).chain(usage_drain);

    Ok(Box::pin(full_stream))
}

pub async fn execute_direct_provider_non_stream(
    http: &reqwest::Client,
    direct: &DirectProviderAccount,
    input: &ChatRequest,
) -> Result<ChatResponse> {
    let url = format!("{}/chat/completions", direct.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("User-Agent", "opencode/1.0.0")
        .json(input);
    if !direct.api_keys.is_empty() {
        let idx = direct.next_key_idx.fetch_add(1, Ordering::Relaxed) % direct.api_keys.len();
        let key = &direct.api_keys[idx];
        if !key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Direct provider upstream error: {}", err_text));
    }
    let chat_resp = resp.json::<ChatResponse>().await?;
    Ok(chat_resp)
}

pub async fn execute_direct_provider_stream(
    http: &reqwest::Client,
    direct: &DirectProviderAccount,
    input: &ChatRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
    let url = format!("{}/chat/completions", direct.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("User-Agent", "opencode/1.0.0")
        .json(input);
    if !direct.api_keys.is_empty() {
        let idx = direct.next_key_idx.fetch_add(1, Ordering::Relaxed) % direct.api_keys.len();
        let key = &direct.api_keys[idx];
        if !key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("Direct provider upstream error: {}", err_text));
    }
    let data_stream = sse_data_stream(resp.bytes_stream());
    let parsed = data_stream.filter_map(|payload_res| async move {
        match payload_res {
            Ok(data) => serde_json::from_str::<ChatCompletionChunk>(&data)
                .ok()
                .map(Ok),
            Err(e) => Some(Err(e)),
        }
    });

    let include_usage = requests_usage(input);
    Ok(filter_direct_usage_stream(parsed, include_usage))
}

fn filter_direct_usage_stream<S>(
    stream: S,
    include_usage: bool,
) -> Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>
where
    S: Stream<Item = Result<ChatCompletionChunk>> + Send + 'static,
{
    let state = Arc::new(tokio::sync::RwLock::new(DirectUsageState::default()));
    let scan_state = state.clone();
    let mapped = stream
        .scan((), move |_, item| {
            let state = scan_state.clone();
            async move {
                match item {
                    Ok(mut chunk) => {
                        let mut saved = state.write().await;
                        if let Some(usage) = chunk.usage.take() {
                            // Keep the last valid upstream usage event; later valid values supersede earlier ones.
                            if is_valid_usage(&usage) {
                                saved.usage = Some(usage);
                            }
                            return Some(None);
                        }
                        if saved.id.is_none() {
                            saved.id = Some(chunk.id.clone());
                            saved.model = Some(chunk.model.clone());
                        }
                        if let Some(id) = &saved.id {
                            chunk.id = id.clone();
                        }
                        Some(Some(Ok(chunk)))
                    }
                    Err(e) => Some(Some(Err(e))),
                }
            }
        })
        .filter_map(futures_util::future::ready);
    let trailer_state = state.clone();
    let trailer = futures_util::stream::once(async move {
        if !include_usage {
            return None;
        }
        let saved = trailer_state.read().await;
        match (saved.usage.clone(), saved.id.as_ref(), saved.model.as_ref()) {
            (Some(usage), Some(id), Some(model)) => {
                Some(Ok(ChatCompletionChunk::usage(id, model, usage)))
            }
            _ => None,
        }
    })
    .filter_map(futures_util::future::ready);
    Box::pin(mapped.chain(trailer))
}

#[derive(Default)]
struct DirectUsageState {
    id: Option<String>,
    model: Option<String>,
    usage: Option<OpenAIUsage>,
}

#[cfg(test)]
mod client_contract_tests {
    use super::*;
    use crate::antigravity::converter::{
        is_claude_model, normalize_model_id, to_antigravity_request, validate_tool_history,
    };
    use crate::antigravity::types::{ChatRequest, Credentials};

    #[test]
    fn upstream_id_is_the_only_identity_source() {
        assert_eq!(
            upstream_tool_call_id(&json!({ "id": "toolu_01ABC", "functionCall": {} })),
            "toolu_01ABC"
        );
        // Part-level ids are honoured too: the translator may place the id beside the call.
        assert_eq!(
            upstream_tool_call_id(&json!({ "functionCall": {}, "id": "part_level_id" })),
            "part_level_id"
        );
        // Blank ids are never promoted to a client id.
        let minted = upstream_tool_call_id(&json!({ "id": "   " }));
        assert!(minted.starts_with("call_"), "{minted}");
        assert!(!minted.trim().is_empty());
        // Request-side transport aliases must not leak into the response path.
        for alias in ["callId", "toolCallId"] {
            let id = upstream_tool_call_id(&json!({ alias: "alias_value" }));
            assert_ne!(id, "alias_value", "{alias} leaked into a response id");
            assert!(id.starts_with("call_"), "{id}");
        }
    }

    #[test]
    fn tool_turn_reports_tool_calls_finish_reason() {
        // A tool turn must not look like a finished answer.
        assert_eq!(resolved_finish_reason(Some("STOP"), true), "tool_calls");
        assert_eq!(resolved_finish_reason(None, true), "tool_calls");
        assert_eq!(
            resolved_finish_reason(Some("MAX_TOKENS"), true),
            "tool_calls"
        );
        // Plain turns still carry the mapped upstream reason.
        assert_eq!(resolved_finish_reason(Some("STOP"), false), "stop");
        assert_eq!(resolved_finish_reason(Some("MAX_TOKENS"), false), "length");
        assert_eq!(
            resolved_finish_reason(Some("SAFETY"), false),
            "content_filter"
        );
        assert_eq!(resolved_finish_reason(None, false), "stop");
    }

    fn claude_replay_request() -> ChatRequest {
        serde_json::from_value(json!({
            "model": "agy/claude-opus-5-5-high",
            "messages": [
                { "role": "user", "content": "weather?" },
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "toolu_01ClientSupplied",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "toolu_01ClientSupplied", "content": "ok" }
            ]
        }))
        .expect("build request")
    }

    #[tokio::test]
    async fn claude_client_id_survives_validation_conversion_and_response_builder() {
        let request = claude_replay_request();

        // Validation keys on the client id, whatever shape it has.
        validate_tool_history(&request.messages).expect("client id shape must validate");

        let creds = Credentials {
            access_token: String::new(),
            refresh_token: String::new(),
            project: "test-project".to_string(),
            client_id: None,
            client_secret: None,
            email: None,
            expires_at: chrono::Utc::now(),
        };
        let state = crate::antigravity::state::AppState::new(
            std::path::PathBuf::from("unused-credentials.json"),
            "test-client".to_string(),
            "test-secret".to_string(),
            creds,
        );
        let payload = to_antigravity_request(&state, &request, "test-project")
            .await
            .expect("convert request");
        let contents = payload["request"]["contents"]
            .as_array()
            .expect("contents array");
        assert_eq!(
            contents[1]["parts"][0]["functionCall"]["id"],
            json!("toolu_01ClientSupplied")
        );
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["id"],
            json!("toolu_01ClientSupplied")
        );

        // The response builder reproduces the same id, so the next turn pairs correctly.
        let upstream_part = json!({
            "functionCall": { "name": "lookup", "args": {} },
            "id": "toolu_01ClientSupplied"
        });
        assert_eq!(
            upstream_tool_call_id(&upstream_part),
            "toolu_01ClientSupplied"
        );

        // Validation still runs on the original messages after conversion, unchanged.
        validate_tool_history(&request.messages).expect("validation must stay id-based");
        assert!(is_claude_model(&normalize_model_id(&request.model)));
    }
}

#[cfg(test)]
mod usage_stream_tests {
    use super::*;
    use futures_util::stream;

    fn usage(prompt_tokens: i64) -> OpenAIUsage {
        OpenAIUsage {
            prompt_tokens,
            completion_tokens: 2,
            total_tokens: prompt_tokens + 2,
        }
    }

    #[tokio::test]
    async fn provider_usage_trailer_uses_completion_id_and_latest_valid_usage() {
        for usage_events in [vec![], vec![usage(3)], vec![usage(3), usage(5), usage(8)]] {
            let observed_usage = Arc::new(RwLock::new(usage_events.last().cloned()));
            let input = stream::iter(vec![Ok(ChatCompletionChunk::finish(
                "chatcmpl-shared",
                "model",
                "stop",
            ))]);
            let mut output = latest_usage_stream(
                input,
                "chatcmpl-shared".to_string(),
                "model".to_string(),
                observed_usage,
                true,
            )
            .collect::<Vec<_>>()
            .await;
            let trailer_count = output
                .iter()
                .filter(|item| item.as_ref().unwrap().usage.is_some())
                .count();
            assert_eq!(trailer_count, usize::from(!usage_events.is_empty()));
            assert_eq!(output[0].as_ref().unwrap().id, "chatcmpl-shared");
            if let Some(last) = output.pop() {
                let last = last.unwrap();
                assert_eq!(last.id, "chatcmpl-shared");
                if let Some(expected) = usage_events.last() {
                    assert_eq!(
                        last.usage.as_ref().unwrap().prompt_tokens,
                        expected.prompt_tokens
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn provider_usage_trailer_requires_explicit_opt_in() {
        let observed_usage = Arc::new(RwLock::new(Some(usage(4))));
        let input = stream::iter(vec![Ok(ChatCompletionChunk::finish(
            "chatcmpl-id",
            "m",
            "stop",
        ))]);
        let output = latest_usage_stream(
            input,
            "chatcmpl-id".to_string(),
            "m".to_string(),
            observed_usage,
            false,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(output.len(), 1);
        assert!(output[0].as_ref().unwrap().usage.is_none());
    }

    #[tokio::test]
    async fn direct_provider_keeps_only_latest_requested_usage_after_choices() {
        let chunks = vec![
            Ok(ChatCompletionChunk::text_delta("upstream-id", "m", "hello")),
            Ok(ChatCompletionChunk::usage("old", "m", usage(2))),
            Ok(ChatCompletionChunk::finish("upstream-id", "m", "stop")),
            Ok(ChatCompletionChunk::usage("new", "m", usage(9))),
        ];
        let output = filter_direct_usage_stream(stream::iter(chunks), true)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].as_ref().unwrap().id, "upstream-id");
        assert_eq!(output[1].as_ref().unwrap().id, "upstream-id");
        let trailer = output[2].as_ref().unwrap();
        assert_eq!(trailer.id, "upstream-id");
        assert_eq!(trailer.usage.as_ref().unwrap().prompt_tokens, 9);
        assert!(trailer.choices.is_empty());
    }

    #[tokio::test]
    async fn direct_provider_suppresses_usage_when_not_requested_or_missing() {
        for (chunks, enabled, expected_len) in [
            (
                vec![Ok(ChatCompletionChunk::usage("id", "m", usage(1)))],
                false,
                0,
            ),
            (
                vec![Ok(ChatCompletionChunk::finish("id", "m", "stop"))],
                true,
                1,
            ),
        ] {
            let output = filter_direct_usage_stream(stream::iter(chunks), enabled)
                .collect::<Vec<_>>()
                .await;
            assert_eq!(output.len(), expected_len);
        }
    }

    #[test]
    fn usage_requires_stream_options_include_usage_true() {
        for (options, expected) in [
            (None, false),
            (Some(serde_json::json!({"include_usage": false})), false),
            (Some(serde_json::json!({"include_usage": true})), true),
        ] {
            let mut request = serde_json::json!({
                "model": "m",
                "messages": [],
            });
            if let Some(options) = options {
                request["stream_options"] = options;
            }
            let parsed: ChatRequest = serde_json::from_value(request).unwrap();
            assert_eq!(requests_usage(&parsed), expected);
        }
    }
}
