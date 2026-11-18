use super::types::{
    AUTH_URL, Credentials, GATEWAY_URL, HUB_USER_AGENT, REDIRECT_URI, SCOPES,
    StoredAntigravityAccountFile, TOKEN_URL, USER_INFO_URL,
};
use anyhow::{Context, Result, anyhow};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{Duration, Utc};
use rand::RngCore;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration as StdDuration;
use tiny_http::{Header, Response, Server};
use tracing::{info, warn};
use url::Url;

pub fn env_or(key: &str, fallback: &str) -> String {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => fallback.to_string(),
    }
}

pub fn generate_pkce_and_auth_url(
    client_id: &str,
    redirect_uri: &str,
) -> Result<(String, String, String)> {
    let mut verifier_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize());

    let mut state_bytes = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut state_bytes);
    let state = URL_SAFE_NO_PAD.encode(state_bytes);

    let mut auth = Url::parse(AUTH_URL)?;
    auth.query_pairs_mut()
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", &SCOPES.join(" "))
        .append_pair("state", &state)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");

    Ok((auth.to_string(), verifier, state))
}

pub fn load_or_login(
    creds_path: &Path,
    client_id: &str,
    client_secret: &str,
) -> Result<Credentials> {
    match load_credentials(creds_path) {
        Ok(creds) => Ok(creds),
        Err(err) => {
            info!(
                "Stored credentials not found or unreadable: {}. Starting OAuth login flow...",
                err
            );
            login(creds_path, client_id, client_secret)
        }
    }
}

pub fn load_credentials(creds_path: &Path) -> Result<Credentials> {
    if !creds_path.exists() {
        let account_file = env_or("ANTIGRAVITY_ACCOUNT_FILE", "");
        if account_file.is_empty() {
            return Err(anyhow!("credentials file not found: {:?}", creds_path));
        }
        let data = fs::read_to_string(&account_file)
            .with_context(|| format!("read Antigravity account file: {}", account_file))?;
        let stored: StoredAntigravityAccountFile = serde_json::from_str(&data)
            .with_context(|| format!("parse Antigravity account file: {}", account_file))?;

        for account in stored.accounts {
            if account.refresh_token.is_empty() || account.enabled == Some(false) {
                continue;
            }
            let parts: Vec<&str> = account.refresh_token.splitn(3, '|').collect();
            let mut project = account.project_id;
            if project.is_empty() && parts.len() > 1 {
                project = parts[1].to_string();
            }
            if project.is_empty() {
                project = account.managed_project_id;
            }
            if project.is_empty() && parts.len() > 2 {
                project = parts[2].to_string();
            }
            if project.is_empty() {
                return Err(anyhow!(
                    "configured Antigravity account has no companion project ID"
                ));
            }
            return Ok(Credentials {
                access_token: String::new(),
                refresh_token: parts[0].to_string(),
                project,
                client_id: None,
                client_secret: None,
                email: if account.email.is_empty() {
                    None
                } else {
                    Some(account.email)
                },
                expires_at: Utc::now() - Duration::hours(1), // trigger refresh
            });
        }
        return Err(anyhow!(
            "Antigravity account file contains no enabled account with a refresh token"
        ));
    }

    let data = fs::read_to_string(creds_path)
        .with_context(|| format!("read credentials file: {:?}", creds_path))?;
    let mut creds: Credentials = serde_json::from_str(&data)
        .with_context(|| format!("parse credentials file: {:?}", creds_path))?;

    if creds.refresh_token.is_empty() {
        return Err(anyhow!("credentials file has no refresh_token"));
    }

    let override_project = env_or("ANTIGRAVITY_PROJECT", "");
    if !override_project.is_empty() {
        creds.project = override_project;
    }

    warn_on_project_mismatch(&creds.project, &env_or("ANTIGRAVITY_ACCOUNT_FILE", ""));

    Ok(creds)
}

/// First `managedProjectId` from an enabled, usable account in the account file.
///
/// The hub account file uses camelCase keys, so this reads both spellings
/// instead of relying on the strict `StoredAntigravityAccountFile` deserializer.
fn first_managed_project(value: &serde_json::Value) -> Option<String> {
    let accounts = value.get("accounts")?.as_array()?;
    for account in accounts {
        if account.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
            continue;
        }
        let has_refresh_token = account_field(account, &["refreshToken", "refresh_token"])
            .is_some_and(|token| !token.trim().is_empty());
        if !has_refresh_token {
            continue;
        }
        let managed = account_field(account, &["managedProjectId", "managed_project_id"])
            .map(str::trim)
            .filter(|project| !project.is_empty());
        if let Some(project) = managed {
            return Some(project.to_string());
        }
    }
    None
}

fn account_field<'a>(account: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| account.get(*key).and_then(|v| v.as_str()))
}

fn managed_project_from_file(account_file: &str) -> Option<String> {
    if account_file.is_empty() {
        return None;
    }
    let data = fs::read_to_string(account_file).ok()?;
    let value: serde_json::Value = serde_json::from_str(&data).ok()?;
    first_managed_project(&value)
}

/// Returns both values when they are both set and differ.
fn project_mismatch<'a>(configured: &'a str, managed: &'a str) -> Option<(&'a str, &'a str)> {
    let configured = configured.trim();
    let managed = managed.trim();
    if configured.is_empty() || managed.is_empty() || configured == managed {
        None
    } else {
        Some((configured, managed))
    }
}

/// Warns when the configured project disagrees with the account file's managed
/// project. The configured value still wins: a project without Gemini Code Assist
/// quota shows up later as `quotaInfo: null` and HTTP 429 QUOTA_EXHAUSTED.
fn warn_on_project_mismatch(configured_project: &str, account_file: &str) {
    if configured_project.trim().is_empty() {
        return;
    }
    let managed_project = match managed_project_from_file(account_file) {
        Some(project) => project,
        None => return,
    };
    if let Some((configured, managed)) = project_mismatch(configured_project, &managed_project) {
        warn!(
            configured = configured,
            managed = managed,
            "Antigravity project mismatch: configured project \"{}\" differs from the enabled account's managedProjectId \"{}\"; using the configured value, which may have no Gemini Code Assist quota (expect quotaInfo: null and HTTP 429 QUOTA_EXHAUSTED)",
            configured,
            managed
        );
    }
}

pub fn save_credentials(creds_path: &Path, creds: &Credentials) -> Result<()> {
    if let Some(parent) = creds_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let data = serde_json::to_string_pretty(creds)?;
    fs::write(creds_path, data)?;
    Ok(())
}

pub fn login(creds_path: &Path, client_id: &str, client_secret: &str) -> Result<Credentials> {
    let mut verifier_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize());

    let mut state_bytes = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut state_bytes);
    let state = URL_SAFE_NO_PAD.encode(state_bytes);

    let mut auth = Url::parse(AUTH_URL)?;
    auth.query_pairs_mut()
        .append_pair("client_id", client_id)
        .append_pair("response_type", "code")
        .append_pair("redirect_uri", REDIRECT_URI)
        .append_pair("scope", &SCOPES.join(" "))
        .append_pair("state", &state)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");

    let server = Server::http("127.0.0.1:8085")
        .map_err(|e| anyhow!("listen on OAuth callback port 8085: {}", e))?;

    println!("Opening Google OAuth in your browser...");
    println!("{}", auth);
    open_browser(auth.as_str());

    let (tx, rx) = mpsc::channel();
    let state_clone = state.clone();

    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + StdDuration::from_secs(300);
        while std::time::Instant::now() < deadline {
            if let Ok(Some(req)) = server.recv_timeout(StdDuration::from_millis(500)) {
                let url_str = format!("http://localhost{}", req.url());
                let parsed = match Url::parse(&url_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if parsed.path() == "/oauth2callback" || parsed.path() == "/oauth-callback" {
                    let mut code = None;
                    let mut req_state = None;
                    let mut err = None;
                    for (k, v) in parsed.query_pairs() {
                        if k == "code" {
                            code = Some(v.to_string());
                        } else if k == "state" {
                            req_state = Some(v.to_string());
                        } else if k == "error" {
                            err = Some(v.to_string());
                        }
                    }

                    if req_state.as_deref() != Some(&state_clone) {
                        let _ = req.respond(
                            Response::from_string("OAuth state mismatch").with_status_code(400),
                        );
                        let _ = tx.send(Err(anyhow!("OAuth state mismatch")));
                        return;
                    }

                    if let Some(e) = err {
                        let _ = req.respond(
                            Response::from_string(
                                "Google authorization failed. Return to the terminal.",
                            )
                            .with_status_code(400),
                        );
                        let _ = tx.send(Err(anyhow!("Google OAuth: {}", e)));
                        return;
                    }

                    let code = match code {
                        Some(c) => c,
                        None => {
                            let _ = req.respond(
                                Response::from_string("Missing authorization code")
                                    .with_status_code(400),
                            );
                            let _ = tx
                                .send(Err(anyhow!("callback did not include authorization code")));
                            return;
                        }
                    };

                    let header =
                        Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
                            .unwrap();
                    let resp = Response::from_string(
                        "Google Antigravity authorization complete. You may close this tab.",
                    )
                    .with_header(header);
                    let _ = req.respond(resp);
                    let _ = tx.send(Ok(code));
                    return;
                }
            }
        }
        let _ = tx.send(Err(anyhow!("timed out waiting for OAuth callback")));
    });

    let code = rx
        .recv_timeout(StdDuration::from_secs(310))
        .map_err(|_| anyhow!("timed out waiting for OAuth callback"))??;

    exchange_code(creds_path, client_id, client_secret, &code, &verifier)
}

pub fn open_browser(target: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", target])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(target).spawn();
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(target).spawn();
    }
}

pub fn exchange_code(
    creds_path: &Path,
    client_id: &str,
    client_secret: &str,
    code: &str,
    verifier: &str,
) -> Result<Credentials> {
    let client = reqwest::blocking::Client::builder()
        .timeout(StdDuration::from_secs(30))
        .build()?;

    let params = [
        ("client_id", client_id),
        ("client_secret", client_secret),
        ("code", code),
        ("grant_type", "authorization_code"),
        ("redirect_uri", REDIRECT_URI),
        ("code_verifier", verifier),
    ];

    let resp = client.post(TOKEN_URL).form(&params).send()?;
    let status = resp.status();
    let body = resp.text()?;
    if !status.is_success() {
        return Err(anyhow!("token exchange failed (HTTP {}): {}", status, body));
    }

    let token_resp: super::types::TokenResponse = serde_json::from_str(&body)
        .with_context(|| format!("parse Google token response: {}", body))?;

    let access_token = token_resp
        .access_token
        .ok_or_else(|| anyhow!("Google token response missing access_token"))?;
    let refresh_token = token_resp
        .refresh_token
        .ok_or_else(|| anyhow!("Google token response missing refresh_token"))?;
    let expires_in = token_resp.expires_in.unwrap_or(3600);

    let mut creds = Credentials {
        access_token: access_token.clone(),
        refresh_token,
        project: String::new(),
        client_id: Some(client_id.to_string()),
        client_secret: Some(client_secret.to_string()),
        email: None,
        expires_at: Utc::now() + Duration::seconds(expires_in),
    };

    // Fetch email
    if let Ok(email_resp) = client
        .get(USER_INFO_URL)
        .header("Authorization", format!("Bearer {}", access_token))
        .send()
    {
        if let Ok(info) = email_resp.json::<serde_json::Value>() {
            if let Some(email) = info.get("email").and_then(|e| e.as_str()) {
                creds.email = Some(email.to_string());
            }
        }
    }

    // Resolve project via loadCodeAssist
    let load_assist_url = format!("{}/v1internal:loadCodeAssist", GATEWAY_URL);
    let payload = json!({
        "metadata": { "ideType": "ANTIGRAVITY" }
    });
    let assist_resp = client
        .post(&load_assist_url)
        .header("Authorization", format!("Bearer {}", access_token))
        .header("User-Agent", HUB_USER_AGENT)
        .json(&payload)
        .send()?;

    let assist_body = assist_resp.text()?;
    let assist_json: serde_json::Value = serde_json::from_str(&assist_body)
        .with_context(|| format!("parse loadCodeAssist response: {}", assist_body))?;

    let mut project_id = String::new();
    if let Some(proj_val) = assist_json.get("cloudaicompanionProject") {
        if let Some(s) = proj_val.as_str() {
            project_id = s.to_string();
        } else if let Some(id_val) = proj_val.get("id").and_then(|v| v.as_str()) {
            project_id = id_val.to_string();
        }
    }

    let override_project = env_or("ANTIGRAVITY_PROJECT", "");
    if !override_project.is_empty() {
        project_id = override_project;
    }

    if project_id.is_empty() {
        return Err(anyhow!(
            "loadCodeAssist returned no project; set ANTIGRAVITY_PROJECT if your account requires one"
        ));
    }
    creds.project = project_id;

    save_credentials(creds_path, &creds)?;
    println!(
        "OAuth connected{}",
        creds
            .email
            .as_ref()
            .map(|e| format!(" as {}", e))
            .unwrap_or_default()
    );
    Ok(creds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn project_mismatch_only_fires_on_two_different_values() {
        assert_eq!(
            project_mismatch("cloudshell-gca", "avrig-ai"),
            Some(("cloudshell-gca", "avrig-ai"))
        );
        assert_eq!(project_mismatch(" avrig-ai ", "avrig-ai"), None);
        assert_eq!(project_mismatch("avrig-ai", "avrig-ai"), None);
        assert_eq!(project_mismatch("", "avrig-ai"), None);
        assert_eq!(project_mismatch("cloudshell-gca", ""), None);
    }

    #[test]
    fn first_managed_project_skips_disabled_and_unusable_accounts() {
        let value = json!({
            "version": 4,
            "accounts": [
                {
                    "email": "stale@example.com",
                    "refreshToken": "1//old",
                    "managedProjectId": "cloudshell-gca",
                    "enabled": false
                },
                {
                    "email": "unusable@example.com",
                    "refreshToken": "",
                    "managedProjectId": "other-project"
                },
                {
                    "email": "active@example.com",
                    "refreshToken": "1//new",
                    "managedProjectId": "avrig-ai",
                    "enabled": true
                }
            ]
        });

        assert_eq!(first_managed_project(&value).as_deref(), Some("avrig-ai"));
    }

    #[test]
    fn managed_project_is_read_from_camel_case_account_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("antigravity-accounts.json");
        fs::write(
            &path,
            r#"{"version":4,"accounts":[{"refreshToken":"1//new","managedProjectId":"avrig-ai","enabled":true}]}"#,
        )
        .unwrap();

        let managed = managed_project_from_file(path.to_str().unwrap());
        assert_eq!(managed.as_deref(), Some("avrig-ai"));
    }

    #[test]
    fn managed_project_lookup_is_inert_without_account_file() {
        assert_eq!(managed_project_from_file(""), None);
        assert_eq!(
            managed_project_from_file("D:/definitely/not/a/real/account/file.json"),
            None
        );
    }
}
