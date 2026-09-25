use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use litellm_auth::SecretValue;
use litellm_host::{
    event::{MachineEvent, PublicRequest, RawResponse, RequestContext, WireRequest},
    host::{Demand, Host, Verdict},
    machine::{CallMachine, HostChannel},
    protocol::Protocol,
};
use litellm_http::{Client, ClientVariant, HttpClientConfig, HttpClientPool};
use litellm_secrets::source::SecretSource;
use litellm_types::{
    llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse,
    utils::ProviderSpecificHeaders,
};
use serde_json::{Map, Value};

use super::{
    Error,
    handler::{decode_response, network, provider_error, send},
    prepare::{prepare_provider_request, resolve_provider},
    types::{MessagesRequest, MessagesShaping},
};
use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;

pub const BODY_FIELDS: [&str; 23] = [
    "messages",
    "max_tokens",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "inference_geo",
    "top_p",
    "mcp_servers",
    "context_management",
    "compaction",
    "container",
    "output_format",
    "speed",
    "output_config",
    "cache_control",
    "reasoning_effort",
    "safeguards",
];

/// The caller's request as the host projects it.
pub struct MessagesCall {
    pub model: String,
    pub body: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub provider_specific_header: Option<ProviderSpecificHeaders>,
    pub timeout: Option<Duration>,
    pub shaping: MessagesShaping,
}

impl MessagesCall {
    fn streams(&self) -> bool {
        self.body.get("stream").and_then(Value::as_bool) == Some(true)
    }
}

pub enum MessagesOutput {
    Message(Box<AnthropicMessagesResponse>),
    /// Every chunk already reached the host through `Deliver`.
    Streamed,
}

/// The upstream response as the caller sees it at stream hand-off, before any chunk.
pub struct MessagesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub struct Messages;

impl Protocol for Messages {
    type Response = MessagesOutput;
    type Error = Error;
    type Projection = MessagesCall;
    type Op = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesHost = HostChannel<Messages>;
pub type MessagesMachine = CallMachine<Messages>;

/// The in-process host for a request already in hand. It answers projection once and
/// observes nothing.
pub struct LocalMessagesHost {
    call: Mutex<Option<MessagesCall>>,
}

impl LocalMessagesHost {
    pub fn new(call: MessagesCall) -> Self {
        Self {
            call: Mutex::new(Some(call)),
        }
    }
}

impl Host<Messages> for LocalMessagesHost {
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.call
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .ok_or_else(|| Error::InvalidRequest("messages request was already projected".into()))
    }

    async fn open(&self, _: MessagesStreamHead) -> Result<Demand, Error> {
        Err(Error::Unsupported(
            "streamed responses need a streaming host",
        ))
    }

    async fn custom_op(&self, op: Infallible) -> Result<(), Error> {
        match op {}
    }
}

pub fn messages_machine(
    pool: &HttpClientPool,
    config: &HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesMachine, litellm_http::Error> {
    let http = pool.client(config, ClientVariant::Provider)?;
    Ok(CallMachine::new(move |host| {
        Box::pin(execute(host, http.clone(), secrets.clone()))
    }))
}

async fn execute(
    host: MessagesHost,
    http: Client,
    secrets: Arc<dyn SecretSource>,
) -> Result<MessagesOutput, Error> {
    let call = host.project().await?;
    let caller_streams = call.streams();
    let resolved = resolve_provider(&call.model, call.custom_llm_provider.as_deref())?;
    let params = host
        .pre_request(PublicRequest {
            model: call.model.clone(),
            custom_llm_provider: resolved.provider.to_string(),
            messages: call.body.get("messages").cloned().unwrap_or(Value::Null),
            params: call
                .body
                .iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            fields: &BODY_FIELDS,
        })
        .await?;
    let patch = params
        .into_iter()
        .filter(|(name, _)| BODY_FIELDS.contains(&name.as_str()))
        .collect();
    let mut body = patched(&call.body, patch);
    let secrets = secrets.resolve(resolved.config.secret_names()).await?;
    let mut recovered_thinking = false;
    loop {
        let resolved = resolve_provider(&call.model, call.custom_llm_provider.as_deref())?;
        let request = prepare_provider_request(
            MessagesRequest {
                model: &call.model,
                body: Value::Object(body.clone()),
                api_key: call.api_key.as_deref(),
                api_base: call.api_base.as_deref(),
                custom_llm_provider: call.custom_llm_provider.as_deref(),
                extra_headers: call.extra_headers.clone(),
                provider_specific_header: call.provider_specific_header.clone(),
                timeout: call.timeout,
                shaping: call.shaping.clone(),
            },
            resolved,
            secrets.as_ref(),
        )?;
        let stream = request.body.get("stream").and_then(Value::as_bool) == Some(true);
        if stream && request.provider != ANTHROPIC_MESSAGES_PROVIDER {
            return Err(Error::Unsupported("streaming messages for this provider"));
        }
        let context = RequestContext {
            model: request.model.clone(),
            custom_llm_provider: request.provider.clone(),
            optional_params: Value::Object(
                request
                    .body
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            ),
            secret_fields: Vec::new(),
            api_key: call.api_key.clone().map(SecretValue::new),
        };
        let wire = host
            .before_send(
                WireRequest {
                    url: request.url,
                    headers: request.upstream_headers,
                    body: request.body,
                },
                context,
            )
            .await?;
        let response = send(&http, &wire.url, &wire.headers, &wire.body, request.timeout).await?;
        if !response.status().is_success() {
            let error = provider_error(response).await;
            if !recovered_thinking
                && let Some(recovered) =
                    super::handler::recover_thinking(&error, &request.provider, &wire.body)?
            {
                body = recovered;
                recovered_thinking = true;
                continue;
            }
            return Err(error);
        }
        if stream {
            return relay(&host, response).await;
        }
        let text = response.text().await.map_err(network)?;
        host.emit(MachineEvent::ResponseReceived {
            raw: RawResponse { body: text.clone() },
        })
        .await?;
        let message = decode_response(request.config, &request.model, &text)?;
        match host
            .after_response(MessagesOutput::Message(Box::new(message)))
            .await?
        {
            Verdict::Return(MessagesOutput::Message(message)) if caller_streams => {
                return super::handler::synthesize(&host, *message).await;
            }
            Verdict::Return(response) => return Ok(response),
            Verdict::Resend(patch) => body = patched(&body, patch),
        }
    }
}

fn patched(body: &Map<String, Value>, patch: Map<String, Value>) -> Map<String, Value> {
    body.iter()
        .filter(|(name, _)| !patch.contains_key(*name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .chain(
            patch
                .iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(name, value)| (name.clone(), value.clone())),
        )
        .collect()
}

/// Hands each upstream chunk to the caller as it arrives. A caller that stops reading
/// ends the upstream read, and the call completes with what it delivered.
async fn relay(
    host: &MessagesHost,
    mut response: reqwest::Response,
) -> Result<MessagesOutput, Error> {
    let head = MessagesStreamHead {
        headers: response
            .headers()
            .iter()
            .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
            .collect(),
    };
    if host.open(head).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    while let Some(chunk) = response.chunk().await.map_err(network)? {
        if host.deliver(chunk).await? == Demand::Detached {
            break;
        }
    }
    Ok(MessagesOutput::Streamed)
}
