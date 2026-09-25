use litellm_http::request::string_headers as shared_string_headers;
pub(super) use litellm_http::request::truncate_error_body;

use serde_json::{Map, Value};

use super::Error;

const HEADER_CONTEXT: &str = "messages";

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{string_headers, truncate_error_body};
    use crate::messages::Error;

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(400);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, 256);
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({"x-count": 3}).as_object().unwrap().clone();
        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            Error::Headers(litellm_http::request::HeaderError {
                context: "messages",
                name: "x-count".to_string(),
                actual: "number",
            })
        );
    }
}
