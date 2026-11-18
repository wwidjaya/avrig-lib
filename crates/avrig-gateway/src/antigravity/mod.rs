pub mod converter;
pub mod oauth;
pub mod quota;
pub mod state;
pub mod types;

pub use converter::*;
pub use oauth::*;
pub use quota::*;
pub use state::*;
pub use types::*;

pub async fn server_fetch_models(state: &AppState) -> Result<Vec<String>, String> {
    {
        let cached = state.models.read().await;
        if let Some(list) = &*cached {
            return Ok(list.clone());
        }
    }

    if let Err(e) = state.refresh_token().await {
        return Err(format!("refresh token: {}", e));
    }

    let project = {
        let creds = state.creds.read().await;
        creds.project.clone()
    };

    let payload = serde_json::json!({ "project": project });
    let resp_val = match state
        .google_post("/v1internal:fetchAvailableModels", &payload)
        .await
    {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };

    let parsed: FetchAvailableModelsResponse = match serde_json::from_value(resp_val) {
        Ok(p) => p,
        Err(e) => return Err(format!("parse fetchAvailableModels: {}", e)),
    };

    if parsed.models.is_empty() {
        return Err("fetchAvailableModels returned no models".to_string());
    }

    let mut model_ids: Vec<String> = parsed.models.keys().cloned().collect();
    model_ids.sort();

    {
        let mut m = state.models.write().await;
        *m = Some(model_ids.clone());
    }
    {
        let mut mi = state.model_info.write().await;
        *mi = parsed.models;
    }

    Ok(model_ids)
}
