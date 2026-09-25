use std::time::Duration;

use litellm_http::{request::http_request, transport::Error as TransportError};
use litellm_llms::anthropic::experimental_pass_through::messages::fake_stream_iterator::fake_anthropic_messages_stream;
use litellm_llms::base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;

use super::{Error, common_utils::truncate_error_body, types::MessagesProvider};
use crate::constants::MESSAGES_TIMEOUT_SECS;

pub(super) fn network(error: reqwest::Error) -> Error {
    Error::Transport(TransportError::Network(error.to_string()))
}

pub(super) async fn send(
    http: &litellm_http::Client,
    url: &str,
    headers: &[(String, String)],
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, Error> {
    let encoded = serde_json::to_vec(body).map_err(|e| Error::RequestEncoding(e.into()))?;
    let builder = headers.iter().fold(
        http.post(url)
            .body(encoded)
            .timeout(timeout.unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS))),
        |builder, (key, value)| builder.header(key, value),
    );
    http_request(builder).await.map_err(network)
}

pub(super) async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match response.text().await {
        Ok(text) => Error::Transport(TransportError::Http {
            status,
            body: truncate_error_body(&text),
        }),
        Err(error) => network(error),
    }
}

pub(super) fn decode_response(
    config: &dyn BaseAnthropicMessagesConfig,
    model: &str,
    text: &str,
) -> Result<AnthropicMessagesResponse, Error> {
    let response = serde_json::from_str(text).map_err(|e| Error::ResponseDecoding(e.into()))?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}

pub(super) async fn synthesize(
    host: &super::route::MessagesHost,
    message: AnthropicMessagesResponse,
) -> Result<super::route::MessagesOutput, Error> {
    use super::route::{MessagesOutput, MessagesStreamHead};
    use litellm_host::host::Demand;

    let head = MessagesStreamHead {
        headers: HeaderMap::from_iter([(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        )]),
    };
    if host.open(head).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    for event in fake_anthropic_messages_stream(&message) {
        if host.deliver(event.sse_frame()).await? == Demand::Detached {
            return Ok(MessagesOutput::Streamed);
        }
    }
    Ok(MessagesOutput::Streamed)
}

pub(super) fn recover_thinking(
    error: &Error,
    provider: MessagesProvider,
    body: &Value,
) -> Result<Option<serde_json::Map<String, Value>>, Error> {
    use litellm_llms::anthropic::common_utils::{
        is_anthropic_invalid_thinking_block_error, strip_thinking_blocks_from_anthropic_messages,
    };
    let Error::Transport(TransportError::Http {
        status: 400,
        body: error_text,
    }) = error
    else {
        return Ok(None);
    };
    if provider != MessagesProvider::Anthropic
        || !is_anthropic_invalid_thinking_block_error(error_text)
    {
        return Ok(None);
    }
    let messages = serde_json::from_value(body["messages"].clone())
        .map_err(|e| Error::RequestDecoding(e.into()))?;
    let stripped = serde_json::to_value(strip_thinking_blocks_from_anthropic_messages(messages))
        .map_err(|e| Error::RequestEncoding(e.into()))?;
    Ok(Some(
        body.as_object()
            .into_iter()
            .flatten()
            .filter(|(name, _)| !matches!(name.as_str(), "messages" | "thinking"))
            .map(|(name, value)| (name.clone(), value.clone()))
            .chain([("messages".into(), stripped)])
            .collect(),
    ))
}
