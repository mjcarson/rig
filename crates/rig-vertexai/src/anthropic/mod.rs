//! Support for Claude models on Vertex AI
//!
//! Vertex AI serves Anthropic's models over the native Anthropic Messages API
//! at `.../publishers/anthropic/models/{model}:rawPredict` (unary) and
//! `:streamRawPredict` (SSE streaming), authenticated with Google Cloud OAuth2
//! bearer tokens instead of an API key. The model id rides in the URL and the
//! API version rides in the body as `anthropic_version` — both handled here,
//! on top of Rig's generic Anthropic machinery.

mod client;
mod http;

pub use client::{
    ANTHROPIC_VERSION_VERTEX, AnthropicVertexClient, AnthropicVertexClientBuilder,
    CLAUDE_FABLE_5_1, CLAUDE_HAIKU_4_5, CLAUDE_HAIKU_5_5, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5_5,
    CLAUDE_SONNET_4_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5_5, CompletionModel, VertexAnthropicExt,
};
pub use http::VertexHttpClient;
