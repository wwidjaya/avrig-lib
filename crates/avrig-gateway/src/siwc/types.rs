use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CLIENT_ID: &str = "";
pub const DEFAULT_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const DEFAULT_AUTH_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const RESOURCE_URL: &str = "https://api.openai.com/v1";
pub const REDIRECT_URI: &str = "http://localhost:51122/oauth-callback";

pub const SCOPES: &[&str] = &[
    "openid",
    "profile",
    "email",
    "offline_access",
    "model.request",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCredentials {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub client_id: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: i64,
    pub client_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSiwcAccountFile {
    pub accounts: Vec<StoredSiwcAccount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredSiwcAccount {
    pub alias: Option<String>,
    pub client_id: Option<String>,
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: DateTime<Utc>,
    #[serde(default)]
    pub enabled: Option<bool>,
}
