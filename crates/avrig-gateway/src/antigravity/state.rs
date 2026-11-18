use super::types::{
    AvailableModel, Credentials, DEFAULT_THOUGHT_SIGNATURE, GATEWAY_URL, HUB_USER_AGENT,
    UpstreamError,
};
use anyhow::{Context, Result, anyhow};
use chrono::{Duration, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub creds_path: PathBuf,
    pub client_id: String,
    pub secret: String,
    pub http: reqwest::Client,
    pub creds: Arc<RwLock<Credentials>>,
    pub models: Arc<RwLock<Option<Vec<String>>>>,
    pub model_info: Arc<RwLock<HashMap<String, AvailableModel>>>,
    pub call_name_cache: Arc<RwLock<HashMap<String, String>>>,
    pub thought_sig_cache: Arc<RwLock<HashMap<String, String>>>,
}

impl AppState {
    pub fn new(creds_path: PathBuf, client_id: String, secret: String, creds: Credentials) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .build()
            .expect("build reqwest client");

        Self {
            creds_path,
            client_id,
            secret,
            http,
            creds: Arc::new(RwLock::new(creds)),
            models: Arc::new(RwLock::new(None)),
            model_info: Arc::new(RwLock::new(HashMap::new())),
            call_name_cache: Arc::new(RwLock::new(HashMap::new())),
            thought_sig_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn set_call_info(&self, id: &str, name: &str, sig: &str) {
        if id.is_empty() {
            return;
        }
        if !name.is_empty() {
            let mut map = self.call_name_cache.write().await;
            map.insert(id.to_string(), name.to_string());
        }
        if !sig.is_empty() {
            let mut map = self.thought_sig_cache.write().await;
            map.insert(id.to_string(), sig.to_string());
        }
    }

    pub async fn get_call_info(&self, id: &str) -> (String, String) {
        let name = {
            let map = self.call_name_cache.read().await;
            map.get(id).cloned().unwrap_or_default()
        };
        let sig = {
            let map = self.thought_sig_cache.read().await;
            map.get(id)
                .cloned()
                .unwrap_or_else(|| DEFAULT_THOUGHT_SIGNATURE.to_string())
        };
        (name, sig)
    }

    pub async fn refresh_token(&self) -> Result<()> {
        let (needs_refresh, refresh_token_val, cid, csec) = {
            let creds = self.creds.read().await;
            let valid = !creds.access_token.is_empty()
                && (creds.expires_at - Utc::now()) > Duration::seconds(60);
            let cid = creds
                .client_id
                .clone()
                .unwrap_or_else(|| self.client_id.clone());
            let csec = creds
                .client_secret
                .clone()
                .unwrap_or_else(|| self.secret.clone());
            (!valid, creds.refresh_token.clone(), cid, csec)
        };

        if !needs_refresh {
            return Ok(());
        }

        let params = [
            ("client_id", cid.as_str()),
            ("client_secret", csec.as_str()),
            ("refresh_token", refresh_token_val.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let resp = self
            .http
            .post(super::types::TOKEN_URL)
            .form(&params)
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!("token refresh failed (HTTP {}): {}", status, body));
        }

        let token_resp: super::types::TokenResponse = serde_json::from_str(&body)
            .with_context(|| format!("parse refresh response: {}", body))?;

        let access_token = token_resp
            .access_token
            .ok_or_else(|| anyhow!("refresh response missing access_token"))?;
        let expires_in = token_resp.expires_in.unwrap_or(3600);

        {
            let mut creds = self.creds.write().await;
            creds.access_token = access_token;
            if let Some(rt) = token_resp.refresh_token {
                if !rt.is_empty() {
                    creds.refresh_token = rt;
                }
            }
            creds.expires_at = Utc::now() + Duration::seconds(expires_in);
        }

        let snapshot = { self.creds.read().await.clone() };
        let _ = super::oauth::save_credentials(&self.creds_path, &snapshot);

        Ok(())
    }

    pub async fn google_post(&self, path: &str, body: &Value) -> Result<Value> {
        let access_token = {
            let creds = self.creds.read().await;
            creds.access_token.clone()
        };

        let url = format!("{}{}", GATEWAY_URL, path);
        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("User-Agent", HUB_USER_AGENT)
            .json(body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            if let Ok(up_err) = serde_json::from_str::<UpstreamError>(&text) {
                return Err(anyhow!(
                    "Upstream HTTP {}: {}",
                    up_err.error.code,
                    up_err.error.message
                ));
            }
            return Err(anyhow!("Upstream HTTP {}: {}", status, text));
        }

        let val: Value = serde_json::from_str(&text)?;
        Ok(val)
    }
}
