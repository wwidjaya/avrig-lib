use super::types::{Credentials, GEMINI_CLI_URL, HUB_USER_AGENT};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuota {
    pub model: String,
    pub group: String,
    pub remaining_fraction: f64,
    pub reset_time: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaResponse {
    pub project: String,
    pub tier: String,
    pub models: Vec<ModelQuota>,
}

pub fn clamp_fraction(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

pub fn quota_group(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    if lower.contains("flash") {
        "gemini-flash".to_string()
    } else if lower.contains("pro") {
        "gemini-pro".to_string()
    } else if lower.contains("claude") {
        "claude".to_string()
    } else {
        "other".to_string()
    }
}

pub async fn fetch_gemini_cli_quota(
    client: &reqwest::Client,
    creds: &Credentials,
) -> Result<QuotaResponse> {
    let url = format!("{}/v1internal:fetchAvailableModels", GEMINI_CLI_URL);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", creds.access_token))
        .header("User-Agent", HUB_USER_AGENT)
        .json(&serde_json::json!({
            "project": creds.project
        }))
        .send()
        .await
        .context("fetchAvailableModels request")?;

    let status = resp.status();
    let body = resp.text().await.context("read response body")?;

    if !status.is_success() {
        return Err(anyhow!("Gemini CLI quota API error ({}): {}", status, body));
    }

    let val: Value = serde_json::from_str(&body).context("parse JSON")?;
    let (models, models_without_quota) = parse_models_response(&val)?;

    if !models_without_quota.is_empty() {
        let missing = models_without_quota.join(", ");
        warn!(
            project = creds.project.as_str(),
            models = missing.as_str(),
            "fetchAvailableModels returned {} model(s) without quotaInfo for project \"{}\": {}; treating them as no entitlement, which usually means the project has no Gemini Code Assist quota",
            models_without_quota.len(),
            creds.project,
            missing
        );
    }

    Ok(QuotaResponse {
        project: creds.project.clone(),
        tier: "standard".to_string(),
        models,
    })
}

/// Parses the `fetchAvailableModels` body into quotas plus the names of models
/// that came back without `quotaInfo`.
fn parse_models_response(val: &Value) -> Result<(Vec<ModelQuota>, Vec<String>)> {
    let models_map = val
        .get("models")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("no models object in response"))?;

    let mut models = Vec::new();
    let mut models_without_quota = Vec::new();
    for (name, model_val) in models_map {
        let (quota, has_quota_info) = parse_model_quota(name, model_val);
        if !has_quota_info {
            models_without_quota.push(name.clone());
        }
        models.push(quota);
    }

    models.sort_by(|a, b| a.model.cmp(&b.model));
    models_without_quota.sort();

    Ok((models, models_without_quota))
}

/// A missing or null `quotaInfo` means the project has no entitlement for that
/// model, so it is reported as an exhausted fraction rather than full quota.
fn parse_model_quota(name: &str, model_val: &Value) -> (ModelQuota, bool) {
    let quota_info = model_val.get("quotaInfo").filter(|q| !q.is_null());
    let (remaining_fraction, reset_time) = match quota_info {
        Some(qi) => match qi.get("remainingFraction").and_then(|f| f.as_f64()) {
            Some(frac) => (
                clamp_fraction(frac),
                qi.get("resetTime")
                    .and_then(|r| r.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            None => (0.0, String::new()),
        },
        None => (0.0, String::new()),
    };

    (
        ModelQuota {
            model: name.to_string(),
            group: quota_group(name),
            remaining_fraction,
            reset_time,
        },
        quota_info.is_some(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_quota_info_means_no_entitlement() {
        let body = json!({
            "models": {
                "gemini-2.5-pro": { "quotaInfo": null },
                "gemini-2.5-flash": { "quotaInfo": { "remainingFraction": 0.5 } }
            }
        });

        let (models, missing) = parse_models_response(&body).unwrap();

        assert_eq!(missing, vec!["gemini-2.5-pro".to_string()]);
        assert_eq!(models.len(), 2);
        let pro = models.iter().find(|m| m.model == "gemini-2.5-pro").unwrap();
        assert_eq!(pro.remaining_fraction, 0.0);
        assert_eq!(pro.reset_time, "");
        let flash = models
            .iter()
            .find(|m| m.model == "gemini-2.5-flash")
            .unwrap();
        assert_eq!(flash.remaining_fraction, 0.5);
    }

    #[test]
    fn present_quota_info_is_parsed_and_clamped() {
        let body = json!({
            "models": {
                "gemini-2.5-pro": {
                    "quotaInfo": {
                        "remainingFraction": 0.25,
                        "resetTime": "2026-11-17T04:04:11Z"
                    }
                },
                "gemini-2.5-flash": { "quotaInfo": { "remainingFraction": 1.0 } },
                "claude-sonnet-4.5": { "quotaInfo": { "remainingFraction": 2.5 } }
            }
        });

        let (models, missing) = parse_models_response(&body).unwrap();

        assert!(missing.is_empty());
        let by_name = |name: &str| models.iter().find(|m| m.model == name).unwrap().clone();
        assert_eq!(by_name("gemini-2.5-pro").remaining_fraction, 0.25);
        assert_eq!(by_name("gemini-2.5-pro").reset_time, "2026-11-17T04:04:11Z");
        assert_eq!(by_name("gemini-2.5-flash").remaining_fraction, 1.0);
        assert_eq!(by_name("claude-sonnet-4.5").remaining_fraction, 1.0);
        assert_eq!(by_name("gemini-2.5-pro").group, "gemini-pro");
    }

    #[test]
    fn missing_models_object_is_an_error() {
        assert!(parse_models_response(&json!({ "models": [] })).is_err());
        assert!(parse_models_response(&json!({})).is_err());
    }
}
