use crate::image::ImageGenerationModel;
use crate::{completion::CompletionModel, embedding::EmbeddingModel};
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::config::{Credentials, SharedCredentialsProvider};
use rig_core::client::Nothing;
use rig_core::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

pub const DEFAULT_AWS_REGION: &str = "us-east-1";

/// The provider name stamped onto explicitly supplied credentials
const EXPLICIT_CREDENTIALS_PROVIDER: &str = "rig-bedrock-explicit";

/// Assemble explicit static credentials from their parts, if both keys are set
///
/// A session token on its own is not a usable credential, so it only rides
/// along when the access key pair is present.
pub(crate) fn assemble_credentials(
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
) -> Option<Credentials> {
    Some(Credentials::new(
        access_key_id?,
        secret_access_key?,
        session_token,
        None,
        EXPLICIT_CREDENTIALS_PROVIDER,
    ))
}

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    region: &'a str,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
}

impl<'a> ClientBuilder<'a> {
    /// Make sure to verify model and region [compatibility]
    ///
    /// [compatibility]: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
    pub fn region(mut self, region: &'a str) -> Self {
        self.region = region;
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

    /// Add a session token to the explicit credentials set with [`ClientBuilder::credentials`]
    ///
    /// Ignored unless an access key pair is also set.
    pub fn session_token(mut self, session_token: impl Into<String>) -> Self {
        self.session_token = Some(session_token.into());
        self
    }

    /// Make sure you have permissions to access [Amazon Bedrock foundation model]
    ///
    /// [ Amazon Bedrock foundation model]: <https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html>
    pub async fn build(self) -> Client {
        let mut loader = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(String::from(self.region)));
        // explicit credentials take the place of the chain when supplied
        if let Some(credentials) = assemble_credentials(
            self.access_key_id,
            self.secret_access_key,
            self.session_token,
        ) {
            loader = loader.credentials_provider(SharedCredentialsProvider::new(credentials));
        }
        let sdk_config = loader.load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);
        Client {
            profile_name: None,
            credentials: None,
            aws_client: Arc::new(OnceCell::from(client)),
        }
    }
}

impl Default for ClientBuilder<'_> {
    fn default() -> Self {
        Self {
            region: DEFAULT_AWS_REGION,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    profile_name: Option<String>,
    credentials: Option<Credentials>,
    pub(crate) aws_client: Arc<OnceCell<aws_sdk_bedrockruntime::Client>>,
}

impl From<aws_sdk_bedrockruntime::Client> for Client {
    fn from(aws_client: aws_sdk_bedrockruntime::Client) -> Self {
        Client {
            profile_name: None,
            credentials: None,
            aws_client: Arc::new(OnceCell::from(aws_client)),
        }
    }
}

impl Client {
    fn new() -> Self {
        Self {
            profile_name: None,
            credentials: None,
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// Create an AWS Bedrock client using AWS profile name
    pub fn with_profile_name(profile_name: &str) -> Self {
        Self {
            profile_name: Some(profile_name.into()),
            credentials: None,
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    /// Create an AWS Bedrock client using an explicit AWS access key pair
    ///
    /// The rest of the AWS config (e.g. the region) still comes from the
    /// environment-driven chain; only the credentials are pinned.
    pub fn with_credentials(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            profile_name: None,
            credentials: assemble_credentials(
                Some(access_key_id.into()),
                Some(secret_access_key.into()),
                session_token,
            ),
            aws_client: Arc::new(OnceCell::new()),
        }
    }

    pub async fn get_inner(&self) -> &aws_sdk_bedrockruntime::Client {
        self.aws_client
            .get_or_init(|| async {
                // explicit credentials outrank the profile, which outranks the
                // bare env chain; a profile still steers non-credential config
                let config = match (&self.credentials, &self.profile_name) {
                    (Some(credentials), profile_name) => {
                        let mut loader = aws_config::defaults(BehaviorVersion::latest())
                            .credentials_provider(SharedCredentialsProvider::new(
                                credentials.clone(),
                            ));
                        if let Some(profile_name) = profile_name {
                            loader = loader.profile_name(profile_name);
                        }
                        loader.load().await
                    }
                    (None, Some(profile_name)) => {
                        aws_config::defaults(BehaviorVersion::latest())
                            .profile_name(profile_name)
                            .load()
                            .await
                    }
                    (None, None) => aws_config::load_from_env().await,
                };
                aws_sdk_bedrockruntime::Client::new(&config)
            })
            .await
    }
}

impl ProviderClient for Client {
    type Input = Nothing;
    type Error = rig_core::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Ok(Client::new())
    }

    fn from_val(_: Nothing) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Err(rig_core::client::ProviderClientError::InvalidConfiguration(
            "use `Client::from_env()` or `Client::with_profile_name(\"aws_profile\")` instead",
        ))
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = EmbeddingModel;

    fn embedding_model(&self, model: impl Into<String>) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, None)
    }

    fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self::EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, Some(ndims))
    }
}

impl ImageGenerationClient for Client {
    type ImageGenerationModel = ImageGenerationModel;

    fn image_generation_model(&self, model: impl Into<String>) -> Self::ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client {
    async fn verify(&self) -> Result<(), VerifyError> {
        // No API endpoint to verify the API key
        Ok(())
    }
}
