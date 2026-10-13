//! A client for the AWS Bedrock Mantle endpoint
//!
//! Mantle serves the native Anthropic Messages API over sigv4-signed HTTP at
//! `https://bedrock-mantle.{region}.api.aws/anthropic`, carrying models that
//! the legacy `bedrock-runtime` Converse API does not (e.g. Bedrock model ids
//! only published on the Messages API). The client reuses Rig's generic
//! Anthropic completion machinery with a [`MantleHttpClient`] signing backend,
//! so tools, prompts, and streaming behave exactly like the Anthropic provider.

use aws_config::BehaviorVersion;
use rig_core::client::{Nothing, Provider, ProviderBuilder, ProviderClientError};
use rig_core::http_client::{self, HeaderValue, HttpClientExt};
use rig_core::prelude::*;
use rig_core::providers::anthropic::completion::{
    ANTHROPIC_VERSION_LATEST, AnthropicCompatibleProvider, GenericCompletionModel,
};

use super::http::MantleHttpClient;
use crate::client::{DEFAULT_AWS_REGION, assemble_credentials};

/// The Mantle URL for a region
///
/// # Arguments
///
/// * `region` - The AWS region to talk to Mantle in
fn mantle_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/anthropic")
}

/// The region from the process environment, or the crate default
fn env_region() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|_| DEFAULT_AWS_REGION.to_owned())
}

/// The provider extension marking a generic client as Bedrock Mantle
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleExt;

impl Provider for MantleExt {
    type Builder = MantleProviderBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl rig_core::client::DebugExt for MantleExt {}

impl AnthropicCompatibleProvider for MantleExt {
    const PROVIDER_NAME: &'static str = "bedrock-mantle";

    /// Mantle reports its transport request id the AWS way
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-amzn-requestid");

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

/// The [`ProviderBuilder`] plumbing for [`MantleExt`]
///
/// Only exists to satisfy the generic client's builder typestate; the real
/// configuration surface is [`MantleClientBuilder`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MantleProviderBuilder;

impl ProviderBuilder for MantleProviderBuilder {
    type Extension<H>
        = MantleExt
    where
        H: HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = "https://bedrock-mantle.us-east-1.api.aws/anthropic";

    fn build<H>(
        _builder: &rig_core::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MantleExt)
    }
}

/// The completion model type Mantle clients hand out
pub type CompletionModel = GenericCompletionModel<MantleExt, MantleHttpClient>;

/// Builds a [`MantleClient`] from a region and credential source
///
/// Credential precedence mirrors the legacy Bedrock client: explicit
/// credentials, then the configured profile, then the environment chain. The
/// region steers both the Mantle URL and the sigv4 signing region, so setting
/// it explicitly is recommended; otherwise it resolves from the AWS config
/// chain (falling back to [`DEFAULT_AWS_REGION`]).
#[derive(Clone, Default)]
pub struct MantleClientBuilder {
    region: Option<String>,
    profile_name: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
}

impl MantleClientBuilder {
    /// Set the AWS region for both the Mantle URL and request signing
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Resolve credentials (and the region, when unset) from an AWS profile
    pub fn profile_name(mut self, profile_name: impl Into<String>) -> Self {
        self.profile_name = Some(profile_name.into());
        self
    }

    /// Authenticate with an explicit AWS access key pair instead of the chain
    pub fn credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Add a session token to the explicit credentials set with [`MantleClientBuilder::credentials`]
    ///
    /// Ignored unless an access key pair is also set.
    pub fn session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }

    /// Build the client, resolving the region when none was set
    ///
    /// Async because an unset region is read from the AWS config chain (the
    /// profile when one was given, the environment otherwise).
    pub async fn build(self) -> http_client::Result<MantleClient> {
        // resolve the region once so the URL and the signature always agree
        let region = match self.region {
            Some(region) => region,
            None => {
                let config = if let Some(profile_name) = &self.profile_name {
                    aws_config::defaults(BehaviorVersion::latest())
                        .profile_name(profile_name)
                        .load()
                        .await
                } else {
                    aws_config::load_from_env().await
                };
                config
                    .region()
                    .map(ToString::to_string)
                    .unwrap_or_else(env_region)
            }
        };
        let credentials = assemble_credentials(
            self.access_key_id,
            self.secret_access_key,
            self.session_token,
        );
        let http_client = MantleHttpClient::new(&region, self.profile_name, credentials);
        MantleClient::assemble(&region, http_client)
    }
}

/// An AWS Bedrock Mantle client speaking the Anthropic Messages API
#[derive(Clone, Debug)]
pub struct MantleClient {
    /// The generic Anthropic-compatible client backed by sigv4 signing
    inner: rig_core::client::Client<MantleExt, MantleHttpClient>,
}

impl MantleClient {
    /// Start building a Mantle client
    pub fn builder() -> MantleClientBuilder {
        MantleClientBuilder::default()
    }

    /// Assemble the generic client around a signing backend
    ///
    /// # Arguments
    ///
    /// * `region` - The region baked into the Mantle URL
    /// * `http_client` - The signing backend, already pinned to that region
    fn assemble(region: &str, http_client: MantleHttpClient) -> http_client::Result<MantleClient> {
        // the Messages API requires its version header on every request
        let mut headers = http_client::HeaderMap::new();
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION_LATEST),
        );
        let inner = rig_core::client::Client::<MantleExt>::builder()
            .base_url(mantle_base_url(region))
            .http_headers(headers)
            .api_key(Nothing)
            .http_client(http_client)
            .build()?;
        Ok(MantleClient { inner })
    }
}

impl ProviderClient for MantleClient {
    type Input = Nothing;
    type Error = ProviderClientError;

    /// Create a Mantle client from the AWS environment credential chain
    fn from_env() -> Result<Self, Self::Error> {
        // from_env is sync, so the region comes straight from the env vars
        let region = env_region();
        let http_client = MantleHttpClient::new(&region, None, None);
        Ok(MantleClient::assemble(&region, http_client)?)
    }

    fn from_val(_: Nothing) -> Result<Self, Self::Error> {
        Err(ProviderClientError::InvalidConfiguration(
            "use `MantleClient::from_env()` or `MantleClient::builder()` instead",
        ))
    }
}

impl CompletionClient for MantleClient {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        GenericCompletionModel::new(self.inner.clone(), model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_bakes_the_region_in() {
        assert_eq!(
            mantle_base_url("us-west-2"),
            "https://bedrock-mantle.us-west-2.api.aws/anthropic"
        );
    }
}
