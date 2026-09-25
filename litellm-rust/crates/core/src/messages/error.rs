use std::sync::Arc;

use litellm_llms::base_llm::chat::transformation::Error as LlmError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    HostFault(#[from] litellm_host::MachineFault),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("messages request was already projected")]
    AlreadyProjected,
    #[error("invalid Anthropic messages request: {0}")]
    RequestDecoding(#[source] JsonError),
    #[error("failed to serialize Anthropic messages request: {0}")]
    RequestEncoding(#[source] JsonError),
    #[error("invalid messages response JSON: {0}")]
    ResponseDecoding(#[source] JsonError),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("unsupported by the Rust messages route: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Client(#[from] litellm_http::Error),
    #[error(transparent)]
    Transport(#[from] litellm_http::transport::Error),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Secret(#[from] SecretError),
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct SecretError(Arc<litellm_secrets::Error>);

impl SecretError {
    pub fn source_error(&self) -> &litellm_secrets::Error {
        &self.0
    }
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct JsonError(Arc<serde_json::Error>);

impl JsonError {
    pub fn source_error(&self) -> &serde_json::Error {
        &self.0
    }
}

impl From<serde_json::Error> for JsonError {
    fn from(error: serde_json::Error) -> Self {
        Self(Arc::new(error))
    }
}

impl PartialEq for JsonError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for JsonError {}

impl From<litellm_secrets::Error> for Error {
    fn from(error: litellm_secrets::Error) -> Self {
        Self::Secret(SecretError(Arc::new(error)))
    }
}

impl PartialEq for SecretError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SecretError {}

impl From<LlmError> for Error {
    fn from(error: LlmError) -> Self {
        match error {
            error @ LlmError::InvalidType { .. } => Self::InvalidRequest(error.to_string()),
            LlmError::MissingField(field) => Self::MissingField(field),
            LlmError::InvalidRequest(message) => Self::InvalidRequest(message),
            LlmError::InvalidResponse(message) => Self::InvalidResponse(message),
            LlmError::Unsupported(reason) => Self::Unsupported(reason),
            LlmError::Auth(error) => Self::Auth(error),
        }
    }
}

impl Error {
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidProvider(_)
            | Self::MissingField(_)
            | Self::InvalidRequest(_)
            | Self::AlreadyProjected
            | Self::RequestDecoding(_)
            | Self::RequestEncoding(_)
            | Self::Unsupported(_)
            | Self::Headers(_) => true,
            Self::Auth(error) => !matches!(error, litellm_auth::Error::MissingApiKey { .. }),
            _ => false,
        }
    }

    pub fn is_response(&self) -> bool {
        matches!(self, Self::InvalidResponse(_) | Self::ResponseDecoding(_))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn json_error() -> JsonError {
        serde_json::from_str::<()>("{").unwrap_err().into()
    }

    #[rstest]
    #[case::invalid_provider(Error::InvalidProvider("x".into()), true, false)]
    #[case::invalid_request(Error::InvalidRequest("x".into()), true, false)]
    #[case::already_projected(Error::AlreadyProjected, true, false)]
    #[case::request_decoding(Error::RequestDecoding(json_error()), true, false)]
    #[case::request_encoding(Error::RequestEncoding(json_error()), true, false)]
    #[case::invalid_response(Error::InvalidResponse("x".into()), false, true)]
    #[case::response_decoding(Error::ResponseDecoding(json_error()), false, true)]
    #[case::unsupported(Error::Unsupported("x"), true, false)]
    fn error_classification(#[case] error: Error, #[case] request: bool, #[case] response: bool) {
        assert_eq!(
            (error.is_request(), error.is_response()),
            (request, response)
        );
    }

    #[test]
    fn json_error_equality_is_identity() {
        let first = json_error();
        assert_eq!(first, first.clone());
        assert_ne!(first, json_error());
    }
}
