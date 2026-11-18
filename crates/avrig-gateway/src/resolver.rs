pub fn strip_model_prefix(model: &str) -> (Option<&str>, String) {
    let m = model.trim();
    if let Some(idx) = m.rfind('/') {
        let prefix_candidate = &m[..idx];
        let rest = &m[idx + 1..];
        if prefix_candidate.eq_ignore_ascii_case("chatgpt") {
            return (Some("chatgpt"), rest.to_string());
        } else if prefix_candidate.eq_ignore_ascii_case("agy")
            || prefix_candidate.eq_ignore_ascii_case("antigravity")
        {
            return (Some("agy"), rest.to_string());
        } else if prefix_candidate.eq_ignore_ascii_case("opencode") {
            return (Some("opencode"), rest.to_string());
        } else {
            return (Some(prefix_candidate), rest.to_string());
        }
    }

    if let Some(rest) = m.strip_prefix("chatgpt/") {
        (Some("chatgpt"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("chatgpt-") {
        (Some("chatgpt"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("chatgpt:") {
        (Some("chatgpt"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("agy/") {
        (Some("agy"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("agy-") {
        (Some("agy"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("agy:") {
        (Some("agy"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("opencode/") {
        (Some("opencode"), rest.to_string())
    } else if let Some(rest) = m.strip_prefix("opencode:") {
        (Some("opencode"), rest.to_string())
    } else {
        (None, m.to_string())
    }
}

pub fn is_siwc_model(model: &str) -> bool {
    let (prefix, base) = strip_model_prefix(model);
    if let Some(p) = prefix {
        return p == "chatgpt";
    }

    let m = base.to_lowercase();
    if m == "gpt-oss-120b" {
        return false;
    }
    m.starts_with("gpt-")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("chatgpt")
        || m.starts_with("openai")
}

pub fn resolve_upstream_model_id(model: &str) -> String {
    let (prefix, base) = strip_model_prefix(model);
    match prefix {
        Some("chatgpt") => {
            if base.is_empty() || base == "default" {
                "gpt-5.6-luna".to_string()
            } else {
                base
            }
        }
        Some("agy") => {
            if base.is_empty() || base == "default" {
                "gemini-3.8-flash-tiered".to_string()
            } else {
                base
            }
        }
        _ => {
            if base.is_empty() || base == "default" {
                "gemini-3.8-flash-tiered".to_string()
            } else {
                base
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_prefix() {
        assert_eq!(
            strip_model_prefix("agy/gemini-2.5-pro"),
            (Some("agy"), "gemini-2.5-pro".to_string())
        );
        assert_eq!(
            strip_model_prefix("chatgpt/o3-mini"),
            (Some("chatgpt"), "o3-mini".to_string())
        );
        assert_eq!(
            strip_model_prefix("gemini-2.5-flash"),
            (None, "gemini-2.5-flash".to_string())
        );
    }

    #[test]
    fn test_is_siwc() {
        assert!(is_siwc_model("chatgpt/gpt-4o"));
        assert!(is_siwc_model("gpt-4o"));
        assert!(is_siwc_model("o1-mini"));
        assert!(!is_siwc_model("agy/gemini-2.5-flash"));
        assert!(!is_siwc_model("gemini-2.5-flash"));
        assert!(!is_siwc_model("gpt-oss-120b"));
    }

    #[test]
    fn test_resolve_upstream() {
        assert_eq!(
            resolve_upstream_model_id("agy/default"),
            "gemini-3.8-flash-tiered"
        );
        assert_eq!(resolve_upstream_model_id("chatgpt/default"), "gpt-5.6-luna");
        assert_eq!(resolve_upstream_model_id("chatgpt/gpt-4o"), "gpt-4o");
    }
}
