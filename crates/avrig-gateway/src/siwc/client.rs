use super::oauth::{refresh_access_token, save_credentials};
use super::types::{DEFAULT_TOKEN_URL, StoredCredentials};
use crate::antigravity::converter::message_text;
use crate::antigravity::types::ChatRequest;
use anyhow::{Context, Result, anyhow};
use chrono::{Duration, Utc};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct SiwcClient {
    pub creds_path: PathBuf,
    pub token_url: String,
    pub creds: Arc<RwLock<StoredCredentials>>,
    pub http: reqwest::Client,
}

impl SiwcClient {
    pub fn new(creds_path: PathBuf, token_url: Option<String>, creds: StoredCredentials) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("build reqwest client");

        Self {
            creds_path,
            token_url: token_url.unwrap_or_else(|| DEFAULT_TOKEN_URL.to_string()),
            creds: Arc::new(RwLock::new(creds)),
            http,
        }
    }

    pub async fn get_valid_access_token(&self) -> Result<String> {
        let (needs_refresh, creds_clone) = {
            let creds = self.creds.read().await;
            let expired = (creds.expires_at - Utc::now()) < Duration::seconds(60);
            (expired, creds.clone())
        };

        if !needs_refresh {
            return Ok(creds_clone.access_token);
        }

        let new_creds = refresh_access_token(&self.http, &self.token_url, &creds_clone).await?;
        let token = new_creds.access_token.clone();
        {
            let mut creds = self.creds.write().await;
            *creds = new_creds.clone();
        }
        let _ = save_credentials(&self.creds_path, &new_creds);
        Ok(token)
    }

    pub async fn available_models(&self) -> Result<Vec<String>> {
        let access_token = self.get_valid_access_token().await?;
        let response = self
            .http
            .get("https://api.openai.com/v1/models")
            .bearer_auth(access_token)
            .send()
            .await
            .context("request SIWC model catalog")?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(anyhow!(
                "SIWC model catalog returned HTTP {}: {}",
                status,
                body
            ));
        }

        let catalog: Value =
            serde_json::from_str(&body).context("parse SIWC model catalog response")?;
        Ok(parse_model_slugs(&catalog))
    }

    pub fn to_responses_request(&self, input: &ChatRequest) -> Result<Value> {
        let mut input_items: Vec<Value> = Vec::new();

        for msg in &input.messages {
            match msg.role.as_str() {
                "system" | "developer" => {
                    let text = message_text(&msg.content)?;
                    if !text.is_empty() {
                        input_items.push(json!({
                            "role": "system",
                            "content": text
                        }));
                    }
                }
                "user" => {
                    let text = message_text(&msg.content)?;
                    input_items.push(json!({
                        "role": "user",
                        "content": text
                    }));
                }
                "assistant" => {
                    if let Ok(text) = message_text(&msg.content) {
                        if !text.is_empty() {
                            input_items.push(json!({
                                "role": "assistant",
                                "content": text
                            }));
                        }
                    }
                    if let Some(tool_calls) = &msg.tool_calls {
                        for call in tool_calls {
                            let parsed_args: Value =
                                serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));
                            input_items.push(json!({
                                "type": "function_call",
                                "call_id": call.id,
                                "name": call.function.name,
                                "arguments": parsed_args.to_string(),
                            }));
                        }
                    }
                }
                "tool" => {
                    let text = message_text(&msg.content)?;
                    input_items.push(json!({
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id.clone().unwrap_or_default(),
                        "output": text
                    }));
                }
                _ => {}
            }
        }

        let mut body = json!({
            "model": input.model,
            "input": input_items,
            "stream": true,
            "store": false
        });

        if let Some(tools) = &input.tools {
            let mut tool_defs = Vec::new();
            for t in tools {
                let mut def = json!({
                    "type": "function",
                    "name": t.function.name,
                });
                if let Some(desc) = &t.function.description {
                    def["description"] = json!(desc);
                }
                if let Some(params) = &t.function.parameters {
                    def["parameters"] = params.clone();
                }
                tool_defs.push(def);
            }
            if !tool_defs.is_empty() {
                body["tools"] = json!(tool_defs);
            }
        }

        Ok(body)
    }

    pub async fn chat_completions_raw(&self, input: &ChatRequest) -> Result<reqwest::Response> {
        let access_token = self.get_valid_access_token().await?;
        let responses_req = self.to_responses_request(input)?;

        let upstream = self
            .http
            .post("https://api.openai.com/v1/responses")
            .bearer_auth(access_token)
            .header("accept", "text/event-stream")
            .header("content-type", "application/json")
            .header("user-agent", "avrig-gateway/0.1.0")
            .json(&responses_req)
            .send()
            .await
            .context("send SIWC responses request")?;

        let status = upstream.status();
        if !status.is_success() {
            let err_text = upstream.text().await.unwrap_or_default();
            return Err(anyhow!("SIWC HTTP {}: {}", status, err_text));
        }

        Ok(upstream)
    }
}

pub fn parse_model_slugs(catalog: &Value) -> Vec<String> {
    let mut slugs = Vec::new();

    if let Some(data) = catalog.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(slug) = entry.get("id").and_then(|id| id.as_str()) {
                slugs.push(slug.to_string());
            }
        }
    } else if let Some(models) = catalog.get("models").and_then(|m| m.as_array()) {
        for entry in models {
            if let Some(slug) = entry.get("slug").and_then(|s| s.as_str()) {
                slugs.push(slug.to_string());
            }
        }
    }

    slugs.sort();
    slugs.dedup();
    slugs
}
