//! Support for the AWS Bedrock Mantle endpoint
//!
//! Mantle is Bedrock's Anthropic Messages-API endpoint: the native Anthropic
//! wire protocol, authenticated with AWS sigv4 (service `bedrock-mantle`)
//! instead of an API key. Some Bedrock model ids are only served here and are
//! rejected by the legacy `bedrock-runtime` Converse API.

mod client;
mod http;

pub use client::{CompletionModel, MantleClient, MantleClientBuilder, MantleExt};
pub use http::MantleHttpClient;
