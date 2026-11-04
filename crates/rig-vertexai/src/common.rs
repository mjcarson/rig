//! Configuration shared by the Gemini and Claude Vertex AI clients
//!
//! Locations, hosts, errors, and the Application Default Credentials chain
//! are the same for every publisher, so they live here rather than behind the
//! `gemini` feature.

use google_cloud_auth::credentials;
use google_cloud_auth::credentials::Credentials;
use thiserror::Error;

// Env vars and terminology (location, project) chosen to match google genai client
// https://googleapis.github.io/python-genai/genai.html#genai.client.Client

/// Default location for Vertex AI Gemini models.
///
/// The `global` endpoint is recommended for Gemini models as it provides higher availability
/// and reduces resource exhaustion errors. Regional endpoints (e.g., `us-central1`, `europe-west4`)
/// are also supported and can be specified via `ClientBuilder::with_location()`.
/// Regional endpoints may be preferred for data residency requirements or to use regional quotas.
pub const DEFAULT_LOCATION: &str = "global";

#[derive(Clone, Debug, Error)]
pub enum VertexAiClientError {
    #[error(
        "Google Cloud project is required. Set it via `ClientBuilder::with_project()` or `GOOGLE_CLOUD_PROJECT`"
    )]
    MissingProject,
    #[error("failed to build source credentials: {0}")]
    SourceCredentials(String),
    #[error("failed to build impersonated credentials: {0}")]
    ImpersonatedCredentials(String),
    #[error("failed to build Vertex AI prediction service: {0}")]
    PredictionService(String),
    #[error(
        "Vertex AI uses Application Default Credentials (ADC). Use `Client::from_env()` for default credentials or `Client::builder().with_credentials(...).build()` for explicit credentials."
    )]
    InvalidInput,
    #[error("failed to build the Claude on Vertex AI client: {0}")]
    AnthropicClient(String),
    #[error(
        "Vertex AI uses Google Cloud credentials. Use `AnthropicVertexClient::from_env()` for default credentials or `AnthropicVertexClient::builder()` for explicit configuration."
    )]
    AnthropicInvalidInput,
}

/// The Vertex AI host serving a location
///
/// The `global` endpoint lives on the bare host, the `us`/`eu` multi-region
/// endpoints live on dedicated `rep` hosts, and every other location gets a
/// regional host. The host and the location in the request path must always
/// agree, so both come from the same resolved location.
///
/// # Arguments
///
/// * `location` - The location requests are served from
pub(crate) fn vertex_host(location: &str) -> String {
    match location {
        // the global endpoint lives on the bare host
        "global" => "https://aiplatform.googleapis.com".to_string(),
        // the multi-region endpoints live on dedicated rep hosts
        "us" | "eu" => format!("https://aiplatform.{location}.rep.googleapis.com"),
        // everything else is a regional host
        region => format!("https://{region}-aiplatform.googleapis.com"),
    }
}

/// Helper function to build credentials with optional service account impersonation.
///
/// # Arguments
///
/// * `explicit_creds` - Credentials to use as is, or `None` to build them from
///   the Application Default Credentials chain
pub(crate) fn build_credentials(
    explicit_creds: Option<Credentials>,
) -> Result<Credentials, VertexAiClientError> {
    if let Some(creds) = explicit_creds {
        Ok(creds)
    } else {
        // Build default credentials
        let source_credentials = credentials::Builder::default()
            .build()
            .map_err(|e| VertexAiClientError::SourceCredentials(e.to_string()))?;
        // Check for service account impersonation
        if let Ok(service_account) = std::env::var("GOOGLE_CLOUD_SERVICE_ACCOUNT") {
            credentials::impersonated::Builder::from_source_credentials(source_credentials)
                .with_target_principal(service_account)
                .build()
                .map_err(|e| VertexAiClientError::ImpersonatedCredentials(e.to_string()))
        } else {
            Ok(source_credentials)
        }
    }
}
