//! A client for Claude models on Vertex AI
//!
//! Vertex AI serves Anthropic's models over the native Messages API at
//! `.../publishers/anthropic/models/{model}:rawPredict` (and
//! `:streamRawPredict` for streaming), authenticated with Google Cloud OAuth2
//! bearer tokens. The client reuses Rig's generic Anthropic completion
//! machinery with a [`VertexHttpClient`] authenticating backend, so tools,
//! prompts, prompt caching, and streaming behave exactly like the Anthropic
//! provider.

use google_cloud_auth::credentials::Credentials;
use rig_core::client::{Nothing, Provider, ProviderBuilder};
use rig_core::http_client::{self, HttpClientExt};
use rig_core::prelude::*;
use rig_core::providers::anthropic::completion::{
    AnthropicCompatibleProvider, GenericCompletionModel, default_max_tokens_for_model,
};

use super::http::{CredentialSource, VertexHttpClient};
use crate::client::{DEFAULT_LOCATION, VertexAiClientError, vertex_host};

/// The Anthropic API version Vertex AI expects in every request body
///
/// Vertex AI reads the API version from the body instead of the
/// `anthropic-version` header; this sentinel is the value Google documents.
/// A newer date-style version set through `additional_params` wins over it.
pub const ANTHROPIC_VERSION_VERTEX: &str = "vertex-2023-10-16";

/// `claude-sonnet-5-5` on Vertex AI
pub const CLAUDE_SONNET_5_5: &str = "claude-sonnet-5-5";
/// `claude-opus-5-5` on Vertex AI
pub const CLAUDE_OPUS_5_5: &str = "claude-opus-5-5";
/// `claude-fable-5-1` on Vertex AI
pub const CLAUDE_FABLE_5_1: &str = "claude-fable-5-1";
/// `claude-haiku-5-5` on Vertex AI
pub const CLAUDE_HAIKU_5_5: &str = "claude-haiku-5-5";
/// `claude-opus-4-8` on Vertex AI
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// `claude-sonnet-4-6` on Vertex AI
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
/// `claude-sonnet-4-5` on Vertex AI
pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5";
/// `claude-haiku-4-5` on Vertex AI
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

/// The base URL serving the Anthropic publisher for a project and location
///
/// # Arguments
///
/// * `host` - The Vertex AI host for the location
/// * `project` - The Google Cloud project to bill against
/// * `location` - The location the model is served from
fn anthropic_base_url(host: &str, project: &str, location: &str) -> String {
    format!("{host}/v1/projects/{project}/locations/{location}/publishers/anthropic")
}

/// The provider extension marking a generic client as Claude on Vertex AI
#[derive(Debug, Default, Clone, Copy)]
pub struct VertexAnthropicExt;

impl Provider for VertexAnthropicExt {
    type Builder = VertexAnthropicProviderBuilder;

    /// Vertex AI has no models listing under the Anthropic publisher; this
    /// const only satisfies the trait and is never requested
    const VERIFY_PATH: &'static str = "/models";
}

impl rig_core::client::DebugExt for VertexAnthropicExt {}

impl AnthropicCompatibleProvider for VertexAnthropicExt {
    const PROVIDER_NAME: &'static str = "vertex-anthropic";

    /// Vertex AI reports no Anthropic-style request id header
    const REQUEST_ID_HEADER: Option<&'static str> = None;

    /// Vertex AI rejects a body containing `model`; it rides in the URL
    const MODEL_IN_BODY: bool = false;

    fn completion_path(model: &str) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(format!("/models/{model}:rawPredict"))
    }

    fn streaming_completion_path(model: &str) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Owned(format!("/models/{model}:streamRawPredict"))
    }

    fn required_body_params(_model: &str) -> Option<serde_json::Value> {
        // the API version Vertex AI reads from the body instead of a header
        Some(serde_json::json!({ "anthropic_version": ANTHROPIC_VERSION_VERTEX }))
    }

    fn default_max_tokens(model: &str) -> Option<u64> {
        // Vertex AI serves bare first-party model ids, so Anthropic's own
        // defaults apply; unknown ids get a safe floor instead of an error
        Some(default_max_tokens_for_model(model).unwrap_or(4096))
    }
}

/// The [`ProviderBuilder`] plumbing for [`VertexAnthropicExt`]
///
/// Only exists to satisfy the generic client's builder typestate; the real
/// configuration surface is [`AnthropicVertexClientBuilder`].
#[derive(Debug, Default, Clone, Copy)]
pub struct VertexAnthropicProviderBuilder;

impl ProviderBuilder for VertexAnthropicProviderBuilder {
    type Extension<H>
        = VertexAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = "https://aiplatform.googleapis.com";

    fn build<H>(
        _builder: &rig_core::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(VertexAnthropicExt)
    }
}

/// The completion model type Claude-on-Vertex clients hand out
pub type CompletionModel = GenericCompletionModel<VertexAnthropicExt, VertexHttpClient>;

/// Builds an [`AnthropicVertexClient`] from a project, location, and credentials
///
/// The project is required (explicitly or via `GOOGLE_CLOUD_PROJECT`); the
/// location falls back to `GOOGLE_CLOUD_LOCATION` and then to
/// [`DEFAULT_LOCATION`]. Credential precedence mirrors the Gemini client:
/// explicit credentials, then a service account key, then the Application
/// Default Credentials chain (honoring `GOOGLE_CLOUD_SERVICE_ACCOUNT`
/// impersonation) — all resolved lazily on the first request.
#[derive(Clone, Default)]
pub struct AnthropicVertexClientBuilder {
    /// The Google Cloud project to bill against
    project: Option<String>,
    /// The location the model is served from
    location: Option<String>,
    /// Pre-built credentials, outranking every other source
    credentials: Option<Credentials>,
    /// A service account key to build credentials from
    service_account_key: Option<serde_json::Value>,
    /// An explicit host overriding the location-derived one
    endpoint: Option<String>,
}

impl AnthropicVertexClientBuilder {
    /// Set the Google Cloud project ID explicitly
    ///
    /// Falls back to `GOOGLE_CLOUD_PROJECT` when unset.
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the Google Cloud location explicitly
    ///
    /// Falls back to `GOOGLE_CLOUD_LOCATION`, then [`DEFAULT_LOCATION`]. The
    /// location steers both the request host and the URL path, so the two can
    /// never disagree.
    pub fn location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Authenticate with pre-built Google Cloud credentials
    pub fn credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Authenticate with a service account key instead of the chain
    ///
    /// Ignored when explicit credentials were also set. The key is the parsed
    /// contents of a service account JSON file; it is held redacted and only
    /// turned into credentials on the first request.
    ///
    /// # Arguments
    ///
    /// * `key` - The parsed service account key JSON
    pub fn service_account_key(mut self, key: serde_json::Value) -> Self {
        self.service_account_key = Some(key);
        self
    }

    /// Override the host requests are sent to (e.g. a test server)
    ///
    /// The URL path still carries the configured project and location.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Build the client, resolving the project and location from the
    /// environment when unset
    pub fn build(self) -> Result<AnthropicVertexClient, VertexAiClientError> {
        // a project is required since it is baked into every request URL
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or(VertexAiClientError::MissingProject)?;
        // fall back to the env var and then the crate-wide default location
        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        // an explicit endpoint outranks the location-derived host
        let host = self.endpoint.unwrap_or_else(|| vertex_host(&location));
        // explicit credentials outrank a service account key, which outranks
        // the Application Default Credentials chain
        let source = match (self.credentials, self.service_account_key) {
            (Some(credentials), _) => CredentialSource::Explicit(credentials),
            (None, Some(key)) => CredentialSource::ServiceAccountKey(key),
            (None, None) => CredentialSource::Chain,
        };
        let http_client = VertexHttpClient::new(source);
        AnthropicVertexClient::assemble(&host, &project, &location, http_client)
    }
}

/// A Vertex AI client speaking the Anthropic Messages API
///
/// The direct analog of Bedrock's Mantle client: Rig's generic Anthropic
/// machinery behind cloud-provider auth, so Claude models on Vertex AI get
/// tools, prompt caching, and streaming without a bespoke wire implementation.
#[derive(Clone, Debug)]
pub struct AnthropicVertexClient {
    /// The generic Anthropic-compatible client backed by Google OAuth
    inner: rig_core::client::Client<VertexAnthropicExt, VertexHttpClient>,
}

impl AnthropicVertexClient {
    /// Start building a Claude-on-Vertex client
    pub fn builder() -> AnthropicVertexClientBuilder {
        AnthropicVertexClientBuilder::default()
    }

    /// Assemble the generic client around an authenticating backend
    ///
    /// # Arguments
    ///
    /// * `host` - The Vertex AI host requests are sent to
    /// * `project` - The Google Cloud project baked into the URL
    /// * `location` - The location baked into the URL
    /// * `http_client` - The authenticating backend
    fn assemble(
        host: &str,
        project: &str,
        location: &str,
        http_client: VertexHttpClient,
    ) -> Result<AnthropicVertexClient, VertexAiClientError> {
        // no `anthropic-version` header: Vertex AI reads it from the body
        let inner = rig_core::client::Client::<VertexAnthropicExt>::builder()
            .base_url(anthropic_base_url(host, project, location))
            .api_key(Nothing)
            .http_client(http_client)
            .build()
            .map_err(|err| VertexAiClientError::AnthropicClient(err.to_string()))?;
        Ok(AnthropicVertexClient { inner })
    }
}

impl ProviderClient for AnthropicVertexClient {
    type Input = Nothing;
    type Error = VertexAiClientError;

    /// Create a client from the environment and the default credentials chain
    ///
    /// Reads `GOOGLE_CLOUD_PROJECT` (required), `GOOGLE_CLOUD_LOCATION`
    /// (optional, defaults to [`DEFAULT_LOCATION`]), and resolves credentials
    /// through the Application Default Credentials chain on first use
    /// (honoring `GOOGLE_CLOUD_SERVICE_ACCOUNT` impersonation).
    fn from_env() -> Result<Self, Self::Error> {
        AnthropicVertexClient::builder().build()
    }

    fn from_val(_: Nothing) -> Result<Self, Self::Error> {
        Err(VertexAiClientError::AnthropicInvalidInput)
    }
}

impl CompletionClient for AnthropicVertexClient {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        GenericCompletionModel::new(self.inner.clone(), model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_bakes_the_project_and_location_in() {
        assert_eq!(
            anthropic_base_url(
                "https://us-east5-aiplatform.googleapis.com",
                "corn",
                "us-east5"
            ),
            "https://us-east5-aiplatform.googleapis.com/v1/projects/corn/locations/us-east5/publishers/anthropic"
        );
    }

    #[test]
    fn paths_address_the_model_with_the_vertex_verbs() {
        assert_eq!(
            VertexAnthropicExt::completion_path(CLAUDE_SONNET_4_5),
            "/models/claude-sonnet-4-5:rawPredict"
        );
        assert_eq!(
            VertexAnthropicExt::streaming_completion_path(CLAUDE_SONNET_4_5),
            "/models/claude-sonnet-4-5:streamRawPredict"
        );
    }

    #[test]
    fn required_body_params_carry_the_vertex_api_version() {
        assert_eq!(
            VertexAnthropicExt::required_body_params(CLAUDE_SONNET_4_5),
            Some(serde_json::json!({ "anthropic_version": "vertex-2023-10-16" }))
        );
    }

    #[test]
    fn known_models_get_anthropic_defaults_and_unknown_a_floor() {
        assert_eq!(
            VertexAnthropicExt::default_max_tokens(CLAUDE_SONNET_4_5),
            Some(64_000)
        );
        assert_eq!(
            VertexAnthropicExt::default_max_tokens("claude-unknown"),
            Some(4096)
        );
    }

    #[test]
    fn builder_requires_a_project() {
        // the project env fallback would mask the error, so only assert on
        // machines without one rather than mutating the process environment
        if std::env::var_os("GOOGLE_CLOUD_PROJECT").is_some() {
            return;
        }
        let err = AnthropicVertexClientBuilder::default()
            .location("us-east5")
            .endpoint("https://example.invalid")
            .build()
            .err();
        assert!(matches!(err, Some(VertexAiClientError::MissingProject)));
    }
}
