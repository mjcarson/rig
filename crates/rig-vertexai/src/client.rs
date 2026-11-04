use crate::completion::CompletionModel;
use google_cloud_aiplatform_v1 as vertexai;
use google_cloud_auth::credentials::Credentials;
use rig_core::client::{CompletionClient, Nothing};
use rig_core::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub use crate::common::{DEFAULT_LOCATION, VertexAiClientError};
use crate::common::{build_credentials, vertex_host};

#[derive(Clone, Debug)]
pub struct ClientBuilder {
    project: Option<String>,
    location: Option<String>,
    credentials: Option<Credentials>,
}

impl ClientBuilder {
    pub fn new() -> Self {
        Self {
            project: None,
            location: None,
            credentials: None,
        }
    }

    /// Set the Google Cloud project ID explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_PROJECT` environment variable.
    pub fn with_project(mut self, project: &str) -> Self {
        self.project = Some(project.to_string());
        self
    }

    /// Set the Google Cloud location explicitly.
    ///
    /// If not set, will fall back to `GOOGLE_CLOUD_LOCATION` environment variable,
    /// or default to "global" if the env var is also not set.
    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    /// Set credentials explicitly.
    ///
    /// If not set, will build credentials from Application Default Credentials (ADC),
    /// with optional service account impersonation if `GOOGLE_CLOUD_SERVICE_ACCOUNT` is set.
    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Build the client with the configured values, falling back to environment variables where not set.
    ///
    /// The Vertex AI client is built lazily on first use via `get_inner()`.
    pub fn build(self) -> Result<Client, VertexAiClientError> {
        let project = self
            .project
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or(VertexAiClientError::MissingProject)?;

        let location = self
            .location
            .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());

        let credentials = build_credentials(self.credentials)?;

        Ok(Client {
            project,
            location,
            credentials,
            vertex_client: Arc::new(OnceCell::new()),
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    project: String,
    location: String,
    credentials: Credentials,
    pub(crate) vertex_client:
        Arc<OnceCell<Result<vertexai::client::PredictionService, VertexAiClientError>>>,
}

impl Client {
    /// Create a new client builder that uses environment variables as defaults.
    ///
    /// You can override any values using the builder methods:
    /// - `.with_project()` - override project
    /// - `.with_location()` - override location
    /// - `.with_credentials()` - override credentials
    ///
    /// Example:
    /// ```no_run
    /// # use rig_vertexai::Client;
    /// # fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
    /// // Use all env vars
    /// let client = Client::builder().build()?;
    ///
    /// // Override just the location
    /// let client = Client::builder().with_location("us-central1").build()?;
    ///
    /// // Override project and location
    /// let client = Client::builder()
    ///     .with_project("my-project")
    ///     .with_location("us-central1")
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a new client using environment variables for project, location, and credentials.
    ///
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    ///
    pub fn new() -> Result<Self, VertexAiClientError> {
        ClientBuilder::new().build()
    }

    /// Create a client using environment variables for project, location, and credentials.
    ///
    /// This is a convenience method that calls the `ProviderClient::from_env()` trait method.
    /// Reads from:
    /// - `GOOGLE_CLOUD_PROJECT` (required)
    /// - `GOOGLE_CLOUD_LOCATION` (optional, defaults to "global")
    /// - `GOOGLE_CLOUD_SERVICE_ACCOUNT` (optional, for service account impersonation)
    pub fn from_env() -> Result<Self, VertexAiClientError> {
        <Self as ProviderClient>::from_env()
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn location(&self) -> &str {
        &self.location
    }

    pub async fn get_inner(
        &self,
    ) -> Result<&vertexai::client::PredictionService, VertexAiClientError> {
        let credentials = self.credentials.clone();
        // non-global locations are only served from their own host, so the
        // endpoint must follow the location or regional requests 404; the
        // global default stays unset so the SDK's universe-domain resolution
        // (sovereign/partner clouds) keeps working
        let endpoint = (self.location != DEFAULT_LOCATION).then(|| vertex_host(&self.location));
        self.vertex_client
            .get_or_init(|| async {
                let mut builder = vertexai::client::PredictionService::builder();
                builder = builder.with_credentials(credentials);
                // only pin the endpoint when the location demands one
                if let Some(endpoint) = endpoint {
                    builder = builder.with_endpoint(endpoint);
                }
                builder
                    .build()
                    .await
                    .map_err(|error| VertexAiClientError::PredictionService(error.to_string()))
            })
            .await
            .as_ref()
            .map_err(Clone::clone)
    }
}

impl ProviderClient for Client {
    type Input = Nothing;
    type Error = VertexAiClientError;

    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Client::new()
    }

    fn from_val(_: Self::Input) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Err(VertexAiClientError::InvalidInput)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model.into())
    }
}

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify credentials - they're validated on first use
        Ok(())
    }
}
