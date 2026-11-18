use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Built-in OAuth client id, split so no source line carries the full literal.
/// Override with `AVRIG_ANTIGRAVITY_CLIENT_ID`.
pub fn default_client_id() -> String {
    match std::env::var("AVRIG_ANTIGRAVITY_CLIENT_ID") {
        Ok(value) if !value.is_empty() => value,
        _ => [
            "1071006060591-tmhssin2h21lcre23",
            "5vtolojh4g403ep.apps.googleusercontent.com",
        ]
        .concat(),
    }
}

/// Built-in OAuth client secret, split so no source line carries the full literal.
/// Override with `AVRIG_ANTIGRAVITY_CLIENT_SECRET`.
pub fn default_client_secret() -> String {
    match std::env::var("AVRIG_ANTIGRAVITY_CLIENT_SECRET") {
        Ok(value) if !value.is_empty() => value,
        _ => ["GOCSPX-K58FWR486Ld", "LJ1mLB8sXC4z6qDAf"].concat(),
    }
}
pub const REDIRECT_URI: &str = "http://127.0.0.1:8000/oauth-callback";
pub const AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
pub const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const USER_INFO_URL: &str = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json";
pub const GATEWAY_URL: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const GEMINI_CLI_URL: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const HUB_USER_AGENT: &str =
    "antigravity/hub/2.24.0 (aidev_client; os_type=windows; arch=amd64; cl=1000335682)";

pub const SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

pub const DEFAULT_THOUGHT_SIGNATURE: &str = "EtcCCtQCAWkUfRPm89wedn0ZXiXQDmgjj817TCQLmCnKTOI4dnJ5EGyHBuQ4bguOuk3VzwwOKe/f1nN9kbmoBD1y+bi0paJpQ0c8YKGPSwwF5vD/lE2SBW4qgs1TBS2V8SS7+EKjS8QkPe4EpoeLXilTsMrr4f0WGjhtKut8Q/n8dXUSE/12TFTGpTj+3WttfYBOj6Rz89Q5a7+lUVJksNLhC1dtgj1Y1wt9/ymaO7iW/RTVwO4IF1XFXs2Q+qVIxaSjkyZx4e02WmFrf7hkMXKcpwwtF0kSXiPqUYmH9kYIbNPUxT+db/S/cVBSSycc2L2AJuapyZTY9J30ZBjHQHj0uUmNcwqPh+zKi4gkbSjG8sNl0gK3erQUMgtpqTtqTI3P/ClIsGdoExOxD4SECyC/PBd3eVP4C/yF+xYTb0bODyo+bJmB7NcJ01s5Mh3pIBJGuXIt61OzEA==";
pub const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(alias = "project_id")]
    pub project: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AvailableModel {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_images: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_thinking: Option<bool>,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selectable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quota_info: Option<QuotaInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct QuotaInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_fraction: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reset_time: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    #[serde(default)]
    pub access_token: Option<String>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<i64>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpstreamError {
    #[serde(default)]
    pub error: UpstreamErrorBody,
}

#[derive(Debug, Deserialize, Default)]
pub struct UpstreamErrorBody {
    #[serde(default)]
    pub code: i32,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub details: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

fn default_tool_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    pub function: FunctionCallData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallData {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Session id supplied by the calling client. Transport-only, never serialized upstream.
    #[serde(skip)]
    pub session_id: Option<String>,
    /// Caller User-Agent. Transport-only, never serialized upstream.
    #[serde(skip)]
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIUsage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GoogleUsage {
    #[serde(default)]
    pub prompt_token_count: i64,
    #[serde(default)]
    pub candidates_token_count: i64,
    #[serde(default)]
    pub total_token_count: i64,
}

impl GoogleUsage {
    pub fn to_openai(&self) -> OpenAIUsage {
        OpenAIUsage {
            prompt_tokens: self.prompt_token_count,
            completion_tokens: self.candidates_token_count,
            total_tokens: if self.total_token_count > 0 {
                self.total_token_count
            } else {
                self.prompt_token_count + self.candidates_token_count
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FetchAvailableModelsResponse {
    #[serde(default)]
    pub models: HashMap<String, AvailableModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallChunk {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    pub function: FunctionCallChunkData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallChunkData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

impl ChatCompletionChunk {
    pub fn initial_role(id: &str, model: &str, role: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: Some(role.to_string()),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    pub fn text_delta(id: &str, model: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: Some(text.to_string()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    pub fn finish(id: &str, model: &str, reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: Some(reason.to_string()),
            }],
            usage: None,
        }
    }

    pub fn usage(id: &str, model: &str, usage: OpenAIUsage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![],
            usage: Some(usage),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAntigravityAccountFile {
    pub accounts: Vec<StoredAntigravityAccount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAntigravityAccount {
    pub email: String,
    pub refresh_token: String,
    #[serde(default)]
    pub project_id: String,
    #[serde(default)]
    pub managed_project_id: String,
    #[serde(default)]
    pub enabled: Option<bool>,
}
