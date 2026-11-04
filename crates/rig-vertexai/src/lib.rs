#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Google Cloud Vertex AI provider integration for Rig.
//!
//! This crate exposes Vertex AI hosted model completions through Rig's
//! completion traits: Gemini models through [`Client`] (the `google` publisher,
//! unary only), and Claude models through
//! [`anthropic::AnthropicVertexClient`] (the `anthropic` publisher over the
//! native Anthropic Messages API, with tools, prompt caching, and streaming).
//! Configure Google Cloud Application Default Credentials or provide
//! credentials through Google Cloud's standard environment before constructing
//! a client.
//!
//! The Gemini client sits behind the default-on `gemini` feature, which pulls
//! in the generated `google-cloud-aiplatform-v1` crate; consumers that only
//! need Claude can disable default features and keep just the `anthropic`
//! module.
//!
//! The root `rig` facade re-exports this crate as `rig::vertexai` when the
//! `vertexai` (Gemini and Claude) or `vertexai-anthropic` (Claude only)
//! feature is enabled.

pub mod anthropic;
#[cfg(feature = "gemini")]
pub mod client;
mod common;
#[cfg(feature = "gemini")]
pub mod completion;
#[cfg(feature = "gemini")]
pub(crate) mod types;

#[cfg(feature = "gemini")]
pub use client::{Client, ClientBuilder};
pub use common::{DEFAULT_LOCATION, VertexAiClientError};
