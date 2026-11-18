pub use crate::antigravity::types::{
    AvailableModel, ChatChoice, ChatChoiceMessage, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionChunkDelta, ChatMessage, ChatRequest, ChatResponse, FunctionCallChunkData,
    FunctionCallData, FunctionDefinition, GoogleUsage, OpenAIUsage, QuotaInfo, Tool, ToolCall,
    ToolCallChunk,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountStatus {
    pub provider: String,
    pub alias: String,
    pub email_or_id: String,
    pub enabled: bool,
    pub healthy: bool,
    pub expires_at: Option<String>,
}
