use crate::antigravity::converter::{
    finish_reason, random_id, to_antigravity_request, trace_antigravity_request,
};
use crate::antigravity::state::AppState as AntigravityAppState;
use crate::antigravity::types::{
    ChatChoice, ChatChoiceMessage, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionChunkDelta, ChatRequest, ChatResponse, FunctionCallChunkData, FunctionCallData,
    OpenAIUsage, ToolCall, ToolCallChunk,
};
use crate::resolver::{
    ModelResolutionError, is_builtin_model_prefix, is_siwc_model, resolve_upstream_model_id,
    strip_model_prefix,
};
use crate::siwc::client::SiwcClient;
use crate::types::{AccountStatus, ModelInfo};
use anyhow::{Result, anyhow};
use futures_util::{Stream, StreamExt};
use rand::RngCore;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
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
    /// Header name that carries a session id on every chat completion, when the upstream wants one.
    pub session_header: Option<String>,
    /// Outbound User-Agent policy: unset keeps the gateway default, `{forward}` passes the
    /// caller's application agent through, and any other value is sent verbatim.
    pub user_agent: Option<String>,
}

impl DirectProviderAccount {
    /// Pick the api key this request will use.
    ///
    /// The index is claimed once per request so the catalogue lookup and the dispatch that
    /// follows authenticate with the same key, and rotating keys never share a cache entry.
    pub fn next_key_index(&self) -> usize {
        if self.api_keys.is_empty() {
            return 0;
        }
        self.next_key_idx.fetch_add(1, Ordering::Relaxed) % self.api_keys.len()
    }

    /// The api key at `index`, when it is one of the configured non-empty keys.
    pub fn api_key_at(&self, index: usize) -> Option<&str> {
        self.api_keys
            .get(index)
            .map(String::as_str)
            .filter(|key| !key.is_empty())
    }
}

/// Upstream HTTP rejection that keeps the provider status code.
///
/// The rendered message is unchanged from the previous string form, so callers see the same
/// body text, while tracing can report the status without parsing it back out.
#[derive(Debug, thiserror::Error)]
#[error("Direct provider upstream error: {body}")]
pub struct UpstreamStatusError {
    pub status: u16,
    pub body: String,
}

impl UpstreamStatusError {
    /// Whether the replayed upstream body says the model name itself is unknown or invalid.
    ///
    /// Only 400 and 404 can describe a model fault. Everything else, including a malformed
    /// `messages` array and token-limit violations, keeps its existing upstream mapping.
    pub fn reports_unknown_model(&self) -> bool {
        if !matches!(self.status, 400 | 404) {
            return false;
        }
        let body = self.body.to_ascii_lowercase();
        const PRECISE_MARKERS: &[&str] = &[
            "unable to determine provider for model",
            "unknown model",
            "model not found",
            "model_not_found",
            "no such model",
            "model does not exist",
            "unsupported model",
            "model is not supported",
            "invalid model name",
            "invalid model id",
            "model id is invalid",
            "model is invalid",
        ];
        if PRECISE_MARKERS
            .iter()
            .any(|marker| body.contains(marker))
        {
            return true;
        }
        // Generic upstreams phrase it as "the model `x` does not exist"; require both the
        // subject and a missing-or-unknown phrase so other 400s are left alone.
        body.contains("model")
            && ["does not exist", "not found", "no such"]
                .iter()
                .any(|marker| body.contains(marker))
    }
}

/// Catalogue cache lifetime. A stale entry is refetched on the next request.
const CATALOGUE_TTL: Duration = Duration::from_secs(60);

/// Upper bound on a catalogue fetch. A slow catalogue falls back to plain dispatch.
const CATALOGUE_TIMEOUT: Duration = Duration::from_secs(3);

/// Cache key for a provider catalogue: one entry per provider and api key index.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CatalogueKey {
    provider: String,
    key_index: usize,
}

enum CatalogueState {
    Ready(Arc<Vec<String>>),
    Unavailable,
}

struct CatalogueEntry {
    state: CatalogueState,
    fetched_at: Instant,
}

/// Routing facts collected while one chat completion is dispatched.
struct ChatDispatchTrace {
    provider: String,
    requested_model: String,
    resolved_model: String,
    upstream_status: Option<u16>,
}

/// Look up a registered direct provider by model prefix.
///
/// Provider names come from configuration and model strings come from clients, so the
/// comparison ignores ASCII case.
fn find_direct_provider(
    providers: &HashMap<String, DirectProviderAccount>,
    prefix: &str,
) -> Option<DirectProviderAccount> {
    providers.get(prefix).cloned().or_else(|| {
        providers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(prefix))
            .map(|(_, account)| account.clone())
    })
}

fn upstream_status_of(err: &anyhow::Error) -> Option<u16> {
    err.downcast_ref::<UpstreamStatusError>().map(|e| e.status)
}

#[derive(Clone, Default)]
pub struct LlmProviderClient {
    pub antigravity_accounts: Arc<RwLock<Vec<AntigravityAccount>>>,
    pub next_ag_idx: Arc<AtomicUsize>,
    pub siwc_accounts: Arc<RwLock<Vec<SiwcAccount>>>,
    pub next_siwc_idx: Arc<AtomicUsize>,
    pub direct_providers: Arc<RwLock<HashMap<String, DirectProviderAccount>>>,
    pub http: reqwest::Client,
    /// Direct provider model catalogues, keyed by provider and api key index.
    model_catalogues: Arc<RwLock<HashMap<CatalogueKey, CatalogueEntry>>>,
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
            model_catalogues: Arc::new(RwLock::new(HashMap::new())),
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
        session_header: Option<String>,
        user_agent: Option<String>,
    ) {
        // A blank or malformed header name is a configuration typo, and reqwest panics on an
        // invalid header name on every request, so an unusable name degrades to "no session
        // header" instead of taking the provider down at dispatch time.
        let session_header = session_header
            .map(|name| name.trim().to_string())
            .filter(|name| reqwest::header::HeaderName::from_bytes(name.as_bytes()).is_ok());
        // A blank user agent policy means "keep the gateway default", same as absent. A literal
        // that cannot be sent as a header value would panic per request, so it is treated as
        // absent too; the `{forward}` sentinel is not a header value and is kept as written.
        let user_agent = user_agent
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .filter(|value| {
                value == FORWARD_USER_AGENT
                    || reqwest::header::HeaderValue::try_from(value.as_str()).is_ok()
            });
        let mut map = self.direct_providers.write().await;
        map.insert(
            provider_name.clone(),
            DirectProviderAccount {
                provider_name,
                base_url,
                api_keys,
                next_key_idx: Arc::new(AtomicUsize::new(0)),
                session_header,
                user_agent,
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

    /// Model ids advertised by a direct provider, cached per provider and api key index.
    ///
    /// `None` means the catalogue is unavailable (transport error, timeout, non-200, or an
    /// empty body). Callers must keep dispatching in that case: a catalogue outage cannot be
    /// allowed to break working traffic. Failures are cached for the same short TTL so a
    /// broken catalogue costs at most one timeout per window.
    async fn direct_provider_models(
        &self,
        direct: &DirectProviderAccount,
        key_index: usize,
    ) -> Option<Arc<Vec<String>>> {
        let cache_key = CatalogueKey {
            provider: direct.provider_name.clone(),
            key_index,
        };

        {
            let cache = self.model_catalogues.read().await;
            if let Some(entry) = cache.get(&cache_key)
                && entry.fetched_at.elapsed() < CATALOGUE_TTL
            {
                return match &entry.state {
                    CatalogueState::Ready(ids) => Some(ids.clone()),
                    CatalogueState::Unavailable => None,
                };
            }
        }

        let url = format!("{}/models", direct.base_url.trim_end_matches('/'));
        let mut req = self.http.get(&url).timeout(CATALOGUE_TIMEOUT);
        if let Some(key) = direct.api_key_at(key_index) {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let state = match req.send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
                Ok(body) => {
                    let ids: Vec<String> = body
                        .get("data")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(|item| {
                                    item.get("id").and_then(Value::as_str).map(str::to_owned)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if ids.is_empty() {
                        tracing::debug!(
                            provider = %direct.provider_name,
                            key_index = key_index,
                            "direct provider model catalogue is empty; dispatching without validation"
                        );
                        CatalogueState::Unavailable
                    } else {
                        CatalogueState::Ready(Arc::new(ids))
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        provider = %direct.provider_name,
                        key_index = key_index,
                        error = %err,
                        "direct provider model catalogue is unreadable; dispatching without validation"
                    );
                    CatalogueState::Unavailable
                }
            },
            Ok(resp) => {
                tracing::warn!(
                    provider = %direct.provider_name,
                    key_index = key_index,
                    upstream_status = resp.status().as_u16(),
                    "direct provider model catalogue returned a non-200; dispatching without validation"
                );
                CatalogueState::Unavailable
            }
            Err(err) => {
                tracing::warn!(
                    provider = %direct.provider_name,
                    key_index = key_index,
                    error = %err,
                    "direct provider model catalogue is unreachable; dispatching without validation"
                );
                CatalogueState::Unavailable
            }
        };

        let resolved = match &state {
            CatalogueState::Ready(ids) => Some(ids.clone()),
            CatalogueState::Unavailable => None,
        };
        self.model_catalogues.write().await.insert(
            cache_key,
            CatalogueEntry {
                state,
                fetched_at: Instant::now(),
            },
        );
        resolved
    }

    /// Reject a sub-model that the direct provider's catalogue does not list.
    ///
    /// When the catalogue cannot be fetched the model is forwarded unchecked, so a catalogue
    /// outage degrades to the previous behaviour instead of failing requests.
    async fn validate_direct_provider_model(
        &self,
        direct: &DirectProviderAccount,
        key_index: usize,
        requested_model: &str,
        base_model: &str,
    ) -> Result<()> {
        // An empty or `default` sub-model means "let the provider choose".
        if base_model.is_empty() || base_model.eq_ignore_ascii_case("default") {
            return Ok(());
        }
        let Some(ids) = self.direct_provider_models(direct, key_index).await else {
            return Ok(());
        };
        if ids.iter().any(|id| id.eq_ignore_ascii_case(base_model)) {
            return Ok(());
        }
        // The prefix is a registered provider here, so report the sub-model, not the prefix.
        Err(ModelResolutionError::SubModelNotFound {
            provider: direct.provider_name.clone(),
            model: requested_model.to_string(),
        }
        .into())
    }

    /// Name the backend that serves `model`, without dispatching a request.
    ///
    /// The HTTP layer uses this so its logs name the same route the dispatch path takes.
    /// An unregistered prefix is reported verbatim, which is how a rejected model appears
    /// in the log line that explains the failure.
    pub async fn resolve_provider_name(&self, model: &str) -> String {
        let (prefix, _) = strip_model_prefix(model);
        if let Some(p) = prefix {
            let providers = self.direct_providers.read().await;
            if let Some((name, _)) = providers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(p))
            {
                return name.clone();
            }
            return p.to_string();
        }
        if is_siwc_model(model) {
            return "chatgpt".to_string();
        }
        if self.antigravity_accounts.read().await.is_empty() {
            // The dispatch path falls back to the first direct provider in this case.
            if let Some((name, _)) = self.direct_providers.read().await.iter().next() {
                return name.clone();
            }
        }
        "agy".to_string()
    }

    /// Dispatch a non-streaming chat completion and trace the routing decision.
    ///
    /// A model prefix that names neither a registered direct provider nor a built-in backend
    /// is rejected with [`ModelResolutionError::ModelNotFound`] instead of falling through to
    /// an unrelated upstream.
    pub async fn chat_completion(&self, input: ChatRequest) -> Result<ChatResponse> {
        let started = std::time::Instant::now();
        let requested_model = input.model.trim().to_string();
        let mut trace = ChatDispatchTrace {
            provider: self.resolve_provider_name(&requested_model).await,
            resolved_model: resolve_upstream_model_id(&requested_model),
            requested_model,
            upstream_status: None,
        };

        let result = self.dispatch_chat_completion(input, &mut trace).await;
        let elapsed_ms = started.elapsed().as_millis() as u64;

        match &result {
            Ok(resp) => tracing::info!(
                provider = %trace.provider,
                requested_model = %trace.requested_model,
                upstream_model = %trace.resolved_model,
                response_model = %resp.model,
                upstream_status = ?trace.upstream_status,
                elapsed_ms = elapsed_ms,
                "chat completion succeeded"
            ),
            Err(err) => tracing::warn!(
                provider = %trace.provider,
                requested_model = %trace.requested_model,
                upstream_model = %trace.resolved_model,
                upstream_status = ?trace.upstream_status,
                elapsed_ms = elapsed_ms,
                error = %err,
                "chat completion failed"
            ),
        }

        result
    }

    async fn dispatch_chat_completion(
        &self,
        mut input: ChatRequest,
        trace: &mut ChatDispatchTrace,
    ) -> Result<ChatResponse> {
        let requested_model = trace.requested_model.clone();
        let (prefix, base_model) = strip_model_prefix(&requested_model);
        input.model = trace.resolved_model.clone();

        // Check if explicitly matching a direct provider (e.g. opencode/gpt-5.6-luna)
        if let Some(p) = prefix {
            let direct = {
                let providers = self.direct_providers.read().await;
                find_direct_provider(&providers, p)
            };
            if let Some(direct) = direct {
                // Claim the key index first so the catalogue check and the dispatch agree,
                // and so rotating keys never share a catalogue entry.
                let key_index = direct.next_key_index();
                self.validate_direct_provider_model(
                    &direct,
                    key_index,
                    &requested_model,
                    &base_model,
                )
                .await?;
                input.model = base_model;
                trace.resolved_model = input.model.clone();
                return match execute_direct_provider_non_stream(
                    &self.http,
                    &direct,
                    key_index,
                    &input,
                )
                .await
                {
                    Ok(resp) => {
                        // A non-2xx status returns early as an error, so this is the success code.
                        trace.upstream_status = Some(200);
                        Ok(resp)
                    }
                    Err(err) => {
                        trace.upstream_status = upstream_status_of(&err);
                        Err(err)
                    }
                };
            }
            // Reject an unknown prefix before any backend is chosen. Falling through here is
            // what made a nonexistent model look like an unrelated provider failure.
            if !is_builtin_model_prefix(p) {
                return Err(ModelResolutionError::ModelNotFound {
                    prefix: p.to_string(),
                    model: requested_model.clone(),
                }
                .into());
            }
        }

        if is_siwc_model(&requested_model) {
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
                let key_index = direct.next_key_index();
                return execute_direct_provider_non_stream(
                    &self.http,
                    direct,
                    key_index,
                    &input,
                )
                .await;
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
        let (prefix, base_model) = strip_model_prefix(&original_model);
        input.model = resolve_upstream_model_id(&original_model);

        if let Some(p) = prefix {
            let direct = {
                let providers = self.direct_providers.read().await;
                find_direct_provider(&providers, p)
            };
            if let Some(direct) = direct {
                // Same catalogue check as the non-stream path: an unknown sub-model is a 404,
                // not an upstream 400 dressed up as a 502.
                let key_index = direct.next_key_index();
                self.validate_direct_provider_model(
                    &direct,
                    key_index,
                    &original_model,
                    &base_model,
                )
                .await?;
                input.model = base_model;
                return execute_direct_provider_stream(&self.http, &direct, key_index, &input)
                    .await;
            }
            if !is_builtin_model_prefix(p) {
                return Err(ModelResolutionError::ModelNotFound {
                    prefix: p.to_string(),
                    model: original_model.clone(),
                }
                .into());
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
                let key_index = direct.next_key_index();
                return execute_direct_provider_stream(&self.http, direct, key_index, &input).await;
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

/// Session id to send upstream for this request.
///
/// The inbound client value wins so a conversation keeps one id end to end; a request that
/// carried none gets a generated one, since OpenCode Go rejects a missing session header.
fn session_id_for(input: &ChatRequest) -> String {
    input
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(new_session_id)
}

/// Generate a session id for a request that carried none.
///
/// OpenCode Go accepts any opaque value, so a UUID v4 keeps callers that cannot send the
/// header (curl, internal callers) working without tracking conversation state.
fn new_session_id() -> String {
    let mut b = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

/// User agent the gateway sends to a direct provider unless the caller's own is forwarded.
const DEFAULT_USER_AGENT: &str = "opencode/1.0.0";

/// Policy value that passes the caller's own user agent through instead of a fixed string.
const FORWARD_USER_AGENT: &str = "{forward}";

/// Caller user agent, when it identifies an application rather than an HTTP library.
///
/// OpenCode Go asks clients to identify themselves and monitors traffic for abuse, so common
/// library agent strings (curl, python-requests, reqwest, and similar) are replaced by the
/// gateway's own default instead of being forwarded.
fn application_user_agent(value: &str) -> Option<&str> {
    const GENERIC: &[&str] = &[
        "curl/",
        "wget/",
        "python",
        "reqwest/",
        "node",
        "axios/",
        "okhttp/",
        "go-http-client/",
        "java/",
        "powershell/",
        "mozilla/",
        "httpie/",
        "postmanruntime/",
        "insomnia/",
        "undici/",
        "deno/",
        "bun/",
        "hyper/",
        "libwww-perl",
        "httpclient/",
        "guzzlehttp/",
        "faraday",
        "restsharp",
    ];
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lowered = trimmed.to_ascii_lowercase();
    if GENERIC.iter().any(|prefix| lowered.starts_with(prefix)) {
        return None;
    }
    Some(trimmed)
}

/// User-Agent header value for a direct provider request.
///
/// `None` keeps the gateway default, `{forward}` passes the caller's agent through when it
/// identifies an application, and any other configured value is sent as written.
fn user_agent_for<'a>(direct: &'a DirectProviderAccount, input: &'a ChatRequest) -> &'a str {
    let Some(policy) = direct.user_agent.as_deref() else {
        return DEFAULT_USER_AGENT;
    };
    if policy == FORWARD_USER_AGENT {
        return input
            .user_agent
            .as_deref()
            .and_then(application_user_agent)
            .unwrap_or(DEFAULT_USER_AGENT);
    }
    policy
}

/// Send a non-streaming completion to a direct provider using a pre-claimed key index.
///
/// The caller claims the index first so the catalogue check and this dispatch authenticate
/// with the same key.
pub async fn execute_direct_provider_non_stream(
    http: &reqwest::Client,
    direct: &DirectProviderAccount,
    key_index: usize,
    input: &ChatRequest,
) -> Result<ChatResponse> {
    let url = format!("{}/chat/completions", direct.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("User-Agent", user_agent_for(direct, input))
        .json(input);
    if let Some(key) = direct.api_key_at(key_index) {
        req = req.header("Authorization", format!("Bearer {}", key));
    }
    if let Some(header) = &direct.session_header {
        req = req.header(header.as_str(), session_id_for(input));
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let err_text = resp.text().await.unwrap_or_default();
        return Err(UpstreamStatusError {
            status,
            body: err_text,
        }
        .into());
    }
    let chat_resp = resp.json::<ChatResponse>().await?;
    Ok(chat_resp)
}

pub async fn execute_direct_provider_stream(
    http: &reqwest::Client,
    direct: &DirectProviderAccount,
    key_index: usize,
    input: &ChatRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
    let url = format!("{}/chat/completions", direct.base_url.trim_end_matches('/'));
    let mut req = http
        .post(&url)
        .header("User-Agent", user_agent_for(direct, input))
        .json(input);
    if let Some(key) = direct.api_key_at(key_index) {
        req = req.header("Authorization", format!("Bearer {}", key));
    }
    if let Some(header) = &direct.session_header {
        req = req.header(header.as_str(), session_id_for(input));
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let err_text = resp.text().await.unwrap_or_default();
        return Err(UpstreamStatusError {
            status,
            body: err_text,
        }
        .into());
    }
    let data_stream = sse_data_stream(resp.bytes_stream());
    let parsed = data_stream.filter_map(|payload_res| async move {
        match payload_res {
            Ok(data) => decode_direct_chunk(&data),
            Err(e) => Some(Err(e)),
        }
    });

    Ok(relay_direct_provider_stream(
        parsed,
        requests_usage(input),
        token_cap_of(input),
        input.model.clone(),
    ))
}

/// Largest upstream frame quoted in a decode-failure log line.
const MAX_LOGGED_FRAME: usize = 240;

/// Truncate an upstream frame to something a log line can carry.
fn preview_frame(payload: &str) -> String {
    match payload.char_indices().nth(MAX_LOGGED_FRAME) {
        Some((cut, _)) => format!("{}...", &payload[..cut]),
        None => payload.to_string(),
    }
}

/// Decode one upstream SSE payload into a completion chunk.
///
/// A payload that is not JSON is a keepalive and is skipped. A payload that is JSON but not a
/// chunk is skipped with a warning, so a stream that forwards nothing can be explained from the
/// logs instead of turning into a silent empty response. A payload that reports an error is
/// returned as an error: an upstream that failed mid-stream must not look like an empty answer.
fn decode_direct_chunk(payload: &str) -> Option<Result<ChatCompletionChunk>> {
    let value = match serde_json::from_str::<Value>(payload) {
        Ok(value) => value,
        Err(_) => {
            tracing::debug!(
                frame = %preview_frame(payload),
                "skipping a non-JSON direct provider stream frame"
            );
            return None;
        }
    };
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("upstream reported an error frame");
        return Some(Err(anyhow!("Direct provider upstream error: {}", message)));
    }
    match serde_json::from_value::<ChatCompletionChunk>(value) {
        Ok(chunk) => Some(Ok(chunk)),
        Err(err) => {
            tracing::warn!(
                error = %err,
                frame = %preview_frame(payload),
                "skipping a direct provider stream frame that is not a completion chunk"
            );
            None
        }
    }
}

/// Completion length bound the request placed on the upstream, when it set one.
fn token_cap_of(input: &ChatRequest) -> Option<i64> {
    [
        input.max_tokens,
        input.max_completion_tokens,
        input.max_output_tokens,
    ]
    .into_iter()
    .flatten()
    .find(|cap| *cap > 0)
}

/// Reason to report for a stream the upstream closed without naming one.
///
/// A stream that consumed the requested cap was truncated, so it reports `length`. Anything
/// else ended on its own, so it reports `stop`.
fn inferred_finish_reason(usage: Option<&OpenAIUsage>, token_cap: Option<i64>) -> String {
    match (usage, token_cap) {
        (Some(usage), Some(cap)) if usage.completion_tokens >= cap => "length".to_string(),
        _ => "stop".to_string(),
    }
}

/// Whether a delta carries anything a client has to see.
fn delta_carries_payload(delta: &ChatCompletionChunkDelta) -> bool {
    delta.role.is_some()
        || delta.content.is_some()
        || delta.reasoning_content.is_some()
        || delta.tool_calls.is_some()
}

/// State collected while one direct provider stream is relayed.
#[derive(Default)]
struct DirectRelayState {
    id: Option<String>,
    model: Option<String>,
    /// Last valid upstream usage block. Never forwarded verbatim.
    usage: Option<OpenAIUsage>,
    /// Last non-empty `finish_reason` the upstream supplied.
    finish_reason: Option<String>,
}

impl DirectRelayState {
    /// Fold one upstream chunk into the relay state.
    ///
    /// Returns the part of the chunk that still has to reach the client. Usage is collected,
    /// never forwarded raw, and the finish reason is held back so every stream is closed by
    /// exactly one terminal chunk. A frame that carried a finish reason and nothing else
    /// therefore yields nothing here; its reason is emitted when the stream ends.
    fn absorb(&mut self, mut chunk: ChatCompletionChunk) -> Option<ChatCompletionChunk> {
        if self.id.is_none() && !chunk.id.trim().is_empty() {
            self.id = Some(chunk.id.clone());
            self.model = Some(chunk.model.clone());
        }

        if let Some(usage) = chunk.usage.take()
            && is_valid_usage(&usage)
        {
            // The last valid upstream usage event wins; an invalid later one does not erase it.
            self.usage = Some(usage);
        }

        let mut reason = None;
        for choice in chunk.choices.iter_mut() {
            // An empty string is not a termination signal. Several upstreams send it on every
            // intermediate frame, and reading it as terminal dropped the real terminal frame.
            if let Some(value) = choice.finish_reason.take() {
                let value = value.trim();
                if !value.is_empty() {
                    reason = Some(value.to_string());
                }
            }
        }
        if reason.is_some() {
            self.finish_reason = reason;
        }

        let carries_payload = chunk
            .choices
            .iter()
            .any(|choice| delta_carries_payload(&choice.delta));
        if !carries_payload {
            return None;
        }
        if let Some(id) = &self.id {
            chunk.id = id.clone();
        }
        Some(chunk)
    }

    /// Terminal chunk for this stream.
    ///
    /// The upstream reason is forwarded when it named one, otherwise it is inferred from the
    /// last observed state. Usage rides on this chunk when the client asked for it, which keeps
    /// `finish_reason` on the final frame rather than behind a usage event.
    fn terminal(
        &self,
        include_usage: bool,
        token_cap: Option<i64>,
        model_fallback: &str,
    ) -> ChatCompletionChunk {
        let id = self
            .id
            .clone()
            .unwrap_or_else(|| format!("chatcmpl-{}", random_id()));
        let model = self
            .model
            .clone()
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| model_fallback.to_string());
        let reason = self
            .finish_reason
            .clone()
            .unwrap_or_else(|| inferred_finish_reason(self.usage.as_ref(), token_cap));
        let mut terminal = ChatCompletionChunk::finish(&id, &model, &reason);
        if include_usage {
            terminal.usage = self.usage.clone();
        }
        terminal
    }
}

/// Relay a direct provider stream to the client with a guaranteed terminal frame.
///
/// Every stream ends with one chunk whose `choices[0].finish_reason` is non-empty: the upstream
/// value when it supplied one, an inferred `length` or `stop` when it did not. Content frames,
/// including reasoning-only deltas, are forwarded as they arrive.
fn relay_direct_provider_stream<S>(
    stream: S,
    include_usage: bool,
    token_cap: Option<i64>,
    model_fallback: String,
) -> Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>
where
    S: Stream<Item = Result<ChatCompletionChunk>> + Send + 'static,
{
    let state = Arc::new(tokio::sync::RwLock::new(DirectRelayState::default()));
    let scan_state = state.clone();
    let forwarded = stream
        .scan((), move |_, item| {
            let state = scan_state.clone();
            async move {
                match item {
                    Ok(chunk) => {
                        let mut state = state.write().await;
                        Some(state.absorb(chunk).map(Ok))
                    }
                    Err(err) => Some(Some(Err(err))),
                }
            }
        })
        .filter_map(futures_util::future::ready);

    let trailer_state = state.clone();
    let trailer = futures_util::stream::once(async move {
        let state = trailer_state.read().await;
        Ok::<ChatCompletionChunk, anyhow::Error>(state.terminal(
            include_usage,
            token_cap,
            &model_fallback,
        ))
    });

    Box::pin(forwarded.chain(trailer))
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
        let output =
            relay_direct_provider_stream(stream::iter(chunks), true, None, "m".to_string())
                .collect::<Vec<_>>()
                .await;
        // The content frame survives, the two usage-only frames are folded away, and the
        // terminal frame carries the newest usage and the upstream reason.
        assert_eq!(output.len(), 2);
        let content = output[0].as_ref().unwrap();
        assert_eq!(content.id, "upstream-id");
        assert_eq!(
            content.choices[0].delta.content.as_deref(),
            Some("hello"),
            "content must not be lost while usage is collected"
        );
        let terminal = output[1].as_ref().unwrap();
        assert_eq!(terminal.id, "upstream-id");
        assert_eq!(
            terminal.choices[0].finish_reason.as_deref(),
            Some("stop"),
            "the stream must close with the upstream reason"
        );
        assert_eq!(terminal.usage.as_ref().unwrap().prompt_tokens, 9);
    }

    #[tokio::test]
    async fn direct_provider_suppresses_usage_when_not_requested_or_missing() {
        for (chunks, enabled) in [
            (
                vec![Ok(ChatCompletionChunk::usage("id", "m", usage(1)))],
                false,
            ),
            (
                vec![Ok(ChatCompletionChunk::finish("id", "m", "stop"))],
                true,
            ),
        ] {
            let output =
                relay_direct_provider_stream(stream::iter(chunks), enabled, None, "m".to_string())
                    .collect::<Vec<_>>()
                    .await;
            // One terminal frame either way, and usage only when the client asked for it.
            assert_eq!(output.len(), 1);
            let terminal = output[0].as_ref().unwrap();
            assert_eq!(terminal.choices[0].finish_reason.as_deref(), Some("stop"));
            assert!(terminal.usage.is_none());
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

#[cfg(test)]
mod direct_stream_relay_tests {
    use super::*;
    use tiny_http::{Header, Response, Server};

    /// Captured shape of an OpenCode Go reasoning model: reasoning-only deltas, an empty
    /// `finish_reason` string on every intermediate frame, and a terminal frame whose delta is
    /// empty but which carries both `finish_reason` and a `usage` block with provider extras.
    const REASONING_THEN_TERMINAL_USAGE: &str = r#"data: {"id":"gw-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":""}]}
data: {"id":"gw-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"reasoning_content":"The"},"finish_reason":""}]}
data: {"id":"gw-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"reasoning_content":" user wants a"},"finish_reason":""}]}
data: {"id":"gw-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":""}]}
data: {"id":"gw-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":36,"completion_tokens":13,"total_tokens":49,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":10},"cost":0,"opencode":{"billing_source":"go","allowance":0.00000792}}}
data: [DONE]
"#;

    /// A stream whose last frame is an ordinary content delta: no `finish_reason` anywhere.
    const NO_FINISH_REASON: &str = r#"data: {"id":"gw-2","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}
data: {"id":"gw-2","object":"chat.completion.chunk","created":1,"model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"content":"ok"}}]}
data: [DONE]
"#;

    /// A stream that ends without a reason while reporting usage that used the whole cap.
    const CAP_EXHAUSTED_WITHOUT_FINISH_REASON: &str = r#"data: {"id":"gw-3","object":"chat.completion.chunk","created":1,"model":"glm-5.3","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}
data: {"id":"gw-3","object":"chat.completion.chunk","created":1,"model":"glm-5.3","choices":[{"index":0,"delta":{"reasoning_content":"The"}}]}
data: {"id":"gw-3","object":"chat.completion.chunk","created":1,"model":"glm-5.3","choices":[{"index":0,"delta":{}}],"usage":{"prompt_tokens":18,"completion_tokens":16,"total_tokens":34,"opencode":{"billing_source":"go"}}}
data: [DONE]
"#;

    /// The shape that produced an empty client stream: nothing usable arrives at all.
    const EMPTY_STREAM: &str = "data: [DONE]\n";

    /// An upstream that fails in band instead of closing the stream.
    const ERROR_FRAME: &str = r#"data: {"error":{"message":"backend overloaded","type":"server_error"}}
data: [DONE]
"#;

    /// Start a stub direct provider that answers `/models` with 404 and replays `body` as the
    /// streaming completion, so the catalogue check falls through to dispatch.
    fn spawn_stream_provider(body: &'static str) -> String {
        let server = Server::http("127.0.0.1:0").expect("bind stub provider");
        let base = format!(
            "http://{}",
            server.server_addr().to_ip().expect("stub provider ip")
        );
        std::thread::spawn(move || {
            let sse_header = Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..])
                .expect("sse content type");
            let json_header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("json content type");
            for request in server.incoming_requests() {
                let response = if request.url().starts_with("/models") {
                    Response::from_string("{}")
                        .with_status_code(404)
                        .with_header(json_header.clone())
                } else {
                    Response::from_string(body).with_header(sse_header.clone())
                };
                let _ = request.respond(response);
            }
        });
        base
    }

    /// Drive one streaming completion through the client and collect the raw relay output.
    async fn collect(body: &'static str, request_body: Value) -> Vec<Result<ChatCompletionChunk>> {
        let base = spawn_stream_provider(body);
        let client = LlmProviderClient::new();
        client
            .add_direct_provider(
                "stub".to_string(),
                base,
                vec!["test-key".to_string()],
                None,
                None,
            )
            .await;
        let mut input: ChatRequest =
            serde_json::from_value(request_body).expect("build chat request");
        input.stream = Some(true);
        let stream = client
            .chat_completion_stream(input)
            .await
            .expect("stream dispatch must reach the stub provider");
        stream.collect().await
    }

    /// Same as [`collect`], for streams the relay must not fail.
    async fn collect_chunks(body: &'static str, request_body: Value) -> Vec<ChatCompletionChunk> {
        collect(body, request_body)
            .await
            .into_iter()
            .map(|item| item.expect("the relayed stream must not fail"))
            .collect()
    }

    fn request(model: &str, max_tokens: Option<i64>, include_usage: bool) -> Value {
        let mut body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": "reply with one word" }]
        });
        if let Some(cap) = max_tokens {
            body["max_tokens"] = json!(cap);
        }
        if include_usage {
            body["stream_options"] = json!({ "include_usage": true });
        }
        body
    }

    fn reasons(chunks: &[ChatCompletionChunk]) -> Vec<Option<String>> {
        chunks
            .iter()
            .map(|chunk| {
                chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.finish_reason.clone())
            })
            .collect()
    }

    fn reasoning(chunks: &[ChatCompletionChunk]) -> String {
        chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect()
    }

    #[tokio::test]
    async fn reasoning_deltas_and_terminal_usage_frame_reach_the_client() {
        let chunks = collect_chunks(
            REASONING_THEN_TERMINAL_USAGE,
            request("stub/deepseek-v4.1-flash", Some(16), false),
        )
        .await;

        // The role frame, two reasoning frames and the content frame, then one terminal frame:
        // the terminal frame used to be swallowed whole because it carried usage.
        assert_eq!(chunks.len(), 5, "reasons: {:?}", reasons(&chunks));
        assert_eq!(
            reasoning(&chunks),
            "The user wants a",
            "reasoning-only deltas must not be dropped"
        );
        assert_eq!(
            chunks[3].choices[0].delta.content.as_deref(),
            Some("ok"),
            "the content frame must survive the terminal frame"
        );
        assert_eq!(
            reasons(&chunks),
            vec![None, None, None, None, Some("stop".to_string())],
            "the upstream reason must be forwarded on the last frame"
        );
        assert_eq!(chunks[4].id, "gw-1", "the upstream id is kept");

        // The bytes a client receives: no intermediate frame may look terminal, and the last
        // frame must name a reason.
        for (index, chunk) in chunks.iter().enumerate().take(4) {
            let frame = serde_json::to_value(chunk).expect("serialize frame");
            assert!(
                frame["choices"][0].get("finish_reason").is_none(),
                "frame {index} must not carry a finish_reason: {frame}"
            );
        }
        let terminal = serde_json::to_value(&chunks[4]).expect("serialize terminal frame");
        assert_eq!(terminal["choices"][0]["finish_reason"], json!("stop"));
    }

    #[tokio::test]
    async fn terminal_usage_stays_closed() {
        let chunks = collect_chunks(
            REASONING_THEN_TERMINAL_USAGE,
            request("stub/deepseek-v4.1-flash", Some(16), true),
        )
        .await;

        let terminal = chunks.last().expect("terminal frame");
        let usage = terminal
            .usage
            .as_ref()
            .expect("the requested usage must ride the terminal frame");
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (36, 13, 49)
        );
        let serialized = serde_json::to_value(terminal).expect("serialize terminal frame");
        let mut keys: Vec<&str> = serialized["usage"]
            .as_object()
            .expect("usage object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["completion_tokens", "prompt_tokens", "total_tokens"],
            "provider-specific usage fields must not leak: {serialized}"
        );
    }

    #[tokio::test]
    async fn stream_without_finish_reason_gets_a_synthesized_terminal_chunk() {
        let chunks = collect_chunks(
            NO_FINISH_REASON,
            request("stub/deepseek-v4.1-flash", Some(16), false),
        )
        .await;

        assert_eq!(chunks.len(), 3, "reasons: {:?}", reasons(&chunks));
        assert_eq!(
            reasons(&chunks),
            vec![None, None, Some("stop".to_string())],
            "a stream that ended on its own closes with stop"
        );
        assert_eq!(chunks[2].id, "gw-2");
        assert_eq!(chunks[2].model, "deepseek-v4.1-flash");
    }

    #[tokio::test]
    async fn stream_that_used_the_token_cap_reports_length() {
        let chunks = collect_chunks(
            CAP_EXHAUSTED_WITHOUT_FINISH_REASON,
            request("stub/glm-5.3", Some(16), false),
        )
        .await;

        assert_eq!(
            reasons(&chunks),
            vec![None, None, Some("length".to_string())],
            "usage equal to the requested cap means the stream was truncated"
        );
    }

    #[tokio::test]
    async fn empty_upstream_stream_still_closes_with_a_terminal_chunk() {
        let chunks = collect_chunks(EMPTY_STREAM, request("stub/glm-5.3", Some(16), false)).await;

        assert_eq!(
            chunks.len(),
            1,
            "an empty upstream stream must not relay nothing"
        );
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            chunks[0].model, "glm-5.3",
            "the requested model names the frame"
        );
        assert!(
            chunks[0].id.starts_with("chatcmpl-"),
            "an id is minted when the upstream never supplied one: {}",
            chunks[0].id
        );
    }

    #[tokio::test]
    async fn upstream_error_frame_is_surfaced_instead_of_an_empty_stream() {
        let output = collect(ERROR_FRAME, request("stub/glm-5.3", Some(16), false)).await;

        let err = output
            .first()
            .expect("the error frame must produce an item")
            .as_ref()
            .expect_err("an in-band upstream error must not be dropped");
        assert!(
            err.to_string().contains("backend overloaded"),
            "the upstream message must survive: {err}"
        );
    }
}

#[cfg(test)]
mod direct_provider_catalogue_tests {
    use super::*;
    use tiny_http::{Header, Response, Server};

    const CATALOGUE_BODY: &str =
        r#"{"object":"list","data":[{"id":"known-model"},{"id":"second-model"}]}"#;

    const COMPLETION_BODY: &str = r#"{"id":"chatcmpl-stub","object":"chat.completion","created":0,"model":"known-model","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

    /// Start a stub direct provider and return its base url.
    ///
    /// `catalogue_status` lets a test simulate a catalogue outage while completions still
    /// answer, which is the case that must fall through to dispatch.
    fn spawn_provider(catalogue_status: u16) -> String {
        let server = Server::http("127.0.0.1:0").expect("bind stub provider");
        let base = format!(
            "http://{}",
            server.server_addr().to_ip().expect("stub provider ip")
        );
        std::thread::spawn(move || {
            let json_header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("content type header");
            for request in server.incoming_requests() {
                let url = request.url().to_string();
                let (status, body) = if url.starts_with("/models") {
                    (catalogue_status, CATALOGUE_BODY)
                } else if url.starts_with("/chat/completions") {
                    (200, COMPLETION_BODY)
                } else {
                    (404, "{}")
                };
                let response = Response::from_string(body)
                    .with_status_code(status)
                    .with_header(json_header.clone());
                let _ = request.respond(response);
            }
        });
        base
    }

    async fn client_with_provider(name: &str, base: String) -> LlmProviderClient {
        let client = LlmProviderClient::new();
        client
            .add_direct_provider(
                name.to_string(),
                base,
                vec!["test-key".to_string()],
                None,
                None,
            )
            .await;
        client
    }

    fn request(model: &str) -> ChatRequest {
        serde_json::from_value(json!({
            "model": model,
            "messages": [{ "role": "user", "content": "ping" }]
        }))
        .expect("build chat request")
    }

    #[tokio::test]
    async fn unknown_sub_model_under_registered_provider_is_reported_missing() {
        let base = spawn_provider(200);
        let client = client_with_provider("composync", base).await;

        let err = client
            .chat_completion(request("composync/does-not-exist-model"))
            .await
            .expect_err("an unlisted sub-model must be rejected locally");

        match err.downcast_ref::<ModelResolutionError>() {
            Some(ModelResolutionError::SubModelNotFound { provider, model }) => {
                assert_eq!(provider, "composync");
                assert_eq!(model, "composync/does-not-exist-model");
            }
            _ => panic!("expected SubModelNotFound, got: {err}"),
        }
    }

    #[tokio::test]
    async fn listed_sub_model_is_dispatched() {
        let base = spawn_provider(200);
        let client = client_with_provider("composync", base).await;

        let resp = client
            .chat_completion(request("composync/known-model"))
            .await
            .expect("a listed sub-model must reach the provider");
        assert_eq!(resp.model, "known-model");
    }

    #[tokio::test]
    async fn unavailable_catalogue_falls_back_to_dispatch() {
        let base = spawn_provider(500);
        let client = client_with_provider("composync", base).await;

        let resp = client
            .chat_completion(request("composync/does-not-exist-model"))
            .await
            .expect("a catalogue outage must not block dispatch");
        assert_eq!(resp.model, "known-model");
    }

    #[test]
    fn catalogue_cache_is_keyed_per_provider_and_key_index() {
        let key_a = CatalogueKey {
            provider: "composync".to_string(),
            key_index: 0,
        };
        let key_b = CatalogueKey {
            provider: "composync".to_string(),
            key_index: 1,
        };
        let key_c = CatalogueKey {
            provider: "opencode".to_string(),
            key_index: 0,
        };
        let mut entries: HashMap<CatalogueKey, CatalogueEntry> = HashMap::new();
        entries.insert(
            key_a.clone(),
            CatalogueEntry {
                state: CatalogueState::Ready(Arc::new(vec!["known-model".to_string()])),
                fetched_at: Instant::now(),
            },
        );
        assert!(matches!(
            entries.get(&key_a).map(|e| &e.state),
            Some(CatalogueState::Ready(_))
        ));
        // A rotated key or a different provider must not read the first catalogue.
        assert!(entries.get(&key_b).is_none());
        assert!(entries.get(&key_c).is_none());
    }

    #[test]
    fn upstream_model_faults_are_recognized() {
        let live_composync_body = r#"{"error":{"message":"Unable to determine provider for model 'does-not-exist-model'.","type":"invalid_request_error","code":"bad_request"}}"#;
        let cases = [
            (
                400,
                live_composync_body,
                true,
                "the live composync wording",
            ),
            (
                404,
                r#"{"error":{"message":"The model `nope` does not exist or you do not have access to it."}}"#,
                true,
                "the OpenAI wording",
            ),
            (
                400,
                r#"{"error":{"message":"messages: at least one message is required"}}"#,
                false,
                "a malformed messages array",
            ),
            (
                400,
                r#"{"error":{"message":"This model's maximum context length is 8192 tokens, however you requested 9000 tokens."}}"#,
                false,
                "a token-limit violation",
            ),
            (
                400,
                r#"{"error":{"message":"invalid model parameters: temperature must be <= 2"}}"#,
                false,
                "generic model parameters",
            ),
            (
                503,
                r#"{"error":{"message":"unknown model backend"}}"#,
                false,
                "a server fault that mentions a model",
            ),
        ];
        for (status, body, expected, label) in cases {
            let err = UpstreamStatusError {
                status,
                body: body.to_string(),
            };
            assert_eq!(err.reports_unknown_model(), expected, "case: {label}");
        }
    }
}

#[cfg(test)]
mod direct_provider_session_header_tests {
    use super::*;
    use std::sync::Mutex;
    use tiny_http::{Header, Response, Server};

    const SESSION_HEADER: &str = "x-opencode-session";

    const USER_AGENT_HEADER: &str = "user-agent";

    const COMPLETION_BODY: &str = r#"{"id":"chatcmpl-stub","object":"chat.completion","created":0,"model":"known-model","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

    /// Headers the stub provider saw on one chat completion.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct SeenHeaders {
        session: Option<String>,
        user_agent: Option<String>,
    }

    /// One capture per chat completion the stub provider answered.
    type Captured = Arc<Mutex<Vec<SeenHeaders>>>;

    fn header_value(request: &tiny_http::Request, name: &'static str) -> Option<String> {
        request
            .headers()
            .iter()
            .find(|h| h.field.equiv(name))
            .map(|h| h.value.as_str().to_string())
    }

    /// Start a stub direct provider that records the session and user-agent headers of every
    /// completion.
    ///
    /// `/models` answers 404 so the catalogue check falls through to dispatch, which is the
    /// path under test here.
    fn spawn_provider(captured: Captured) -> String {
        let server = Server::http("127.0.0.1:0").expect("bind stub provider");
        let base = format!(
            "http://{}",
            server.server_addr().to_ip().expect("stub provider ip")
        );
        std::thread::spawn(move || {
            let json_header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("content type header");
            for request in server.incoming_requests() {
                let url = request.url().to_string();
                if url.starts_with("/models") {
                    let _ = request.respond(
                        Response::from_string("{}")
                            .with_status_code(404)
                            .with_header(json_header.clone()),
                    );
                    continue;
                }
                let seen = SeenHeaders {
                    session: header_value(&request, SESSION_HEADER),
                    user_agent: header_value(&request, USER_AGENT_HEADER),
                };
                captured.lock().expect("capture lock").push(seen);
                let _ = request.respond(
                    Response::from_string(COMPLETION_BODY)
                        .with_status_code(200)
                        .with_header(json_header.clone()),
                );
            }
        });
        base
    }

    async fn client_with_provider(
        base: String,
        session_header: Option<String>,
        user_agent: Option<String>,
    ) -> LlmProviderClient {
        let client = LlmProviderClient::new();
        client
            .add_direct_provider(
                "session-provider".to_string(),
                base,
                vec!["test-key".to_string()],
                session_header,
                user_agent,
            )
            .await;
        client
    }

    fn request(model: &str) -> ChatRequest {
        serde_json::from_value(json!({
            "model": model,
            "messages": [{ "role": "user", "content": "ping" }]
        }))
        .expect("build chat request")
    }

    /// Shape check for a canonical UUID v4: 36 characters, dashes at the fixed offsets, a
    /// version nibble of 4 and a variant nibble of 8, 9, a or b.
    fn is_uuid_v4(value: &str) -> bool {
        let bytes = value.as_bytes();
        if bytes.len() != 36 {
            return false;
        }
        for offset in [8, 13, 18, 23] {
            if bytes[offset] != b'-' {
                return false;
            }
        }
        let hex_ok = value
            .char_indices()
            .all(|(i, c)| matches!(i, 8 | 13 | 18 | 23) || c.is_ascii_hexdigit());
        hex_ok && bytes[14] == b'4' && matches!(bytes[19], b'8' | b'9' | b'a' | b'b')
    }

    fn captured_values(captured: &Captured) -> Vec<SeenHeaders> {
        captured.lock().expect("capture lock").clone()
    }

    #[tokio::test]
    async fn inbound_session_id_reaches_the_provider() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, Some(SESSION_HEADER.to_string()), None).await;

        let mut input = request("session-provider/known-model");
        input.session_id = Some("abc-123".to_string());
        let resp = client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");
        assert_eq!(resp.model, "known-model");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1, "one completion must produce one capture");
        assert_eq!(
            seen[0].session.as_deref(),
            Some("abc-123"),
            "the inbound session id must be forwarded verbatim"
        );
    }

    #[tokio::test]
    async fn missing_session_id_is_generated_as_uuid_v4() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, Some(SESSION_HEADER.to_string()), None).await;

        let input = request("session-provider/known-model");
        assert!(input.session_id.is_none());
        let stream = client
            .chat_completion_stream(input)
            .await
            .expect("stream dispatch must reach the stub provider");
        drop(stream);

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1, "one completion must produce one capture");
        let value = seen[0]
            .session
            .as_deref()
            .expect("the session header must be present");
        assert!(
            is_uuid_v4(value),
            "a generated session id must be a canonical uuid v4, got: {value}"
        );
    }

    #[tokio::test]
    async fn provider_without_session_header_sends_none() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, None, None).await;

        let mut input = request("session-provider/known-model");
        input.session_id = Some("abc-123".to_string());
        client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert!(
            seen[0].session.is_none(),
            "a provider that does not ask for a session header must not receive one"
        );
    }

    #[tokio::test]
    async fn blank_session_header_name_degrades_to_no_header() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, Some("   ".to_string()), None).await;

        let mut input = request("session-provider/known-model");
        input.session_id = Some("abc-123".to_string());
        client
            .chat_completion(input)
            .await
            .expect("a blank header name must not panic the dispatch path");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert!(
            seen[0].session.is_none(),
            "a whitespace-only header name must be treated as absent"
        );
    }

    #[tokio::test]
    async fn forward_policy_sends_the_callers_application_agent() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, None, Some(FORWARD_USER_AGENT.to_string())).await;

        let mut input = request("session-provider/known-model");
        input.user_agent = Some("opencode/1.2.3".to_string());
        client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].user_agent.as_deref(),
            Some("opencode/1.2.3"),
            "the {{forward}} policy must pass an application agent through"
        );
    }

    #[tokio::test]
    async fn forward_policy_still_blocks_library_agents() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, None, Some(FORWARD_USER_AGENT.to_string())).await;

        let mut input = request("session-provider/known-model");
        input.user_agent = Some("curl/8.9.1".to_string());
        client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].user_agent.as_deref(),
            Some(DEFAULT_USER_AGENT),
            "a library user agent must never reach the upstream"
        );
    }

    #[tokio::test]
    async fn literal_policy_replaces_the_caller_agent() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, None, Some("avrig/0.3.0".to_string())).await;

        let mut input = request("session-provider/known-model");
        input.user_agent = Some("opencode/1.2.3".to_string());
        client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].user_agent.as_deref(),
            Some("avrig/0.3.0"),
            "a literal policy must be sent verbatim whatever the caller sends"
        );
    }

    #[tokio::test]
    async fn absent_policy_keeps_the_gateway_default() {
        let captured: Captured = Arc::new(Mutex::new(Vec::new()));
        let base = spawn_provider(captured.clone());
        let client = client_with_provider(base, None, None).await;

        let mut input = request("session-provider/known-model");
        input.user_agent = Some("opencode/1.2.3".to_string());
        client
            .chat_completion(input)
            .await
            .expect("completion must reach the stub provider");

        let seen = captured_values(&captured);
        assert_eq!(seen.len(), 1);
        assert_eq!(
            seen[0].user_agent.as_deref(),
            Some(DEFAULT_USER_AGENT),
            "a provider without a policy keeps the gateway default"
        );
    }
}
