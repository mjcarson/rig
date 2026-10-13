//! A sigv4-signing HTTP backend for the Bedrock Mantle endpoint
//!
//! Mantle speaks the native Anthropic Messages API but authenticates with
//! AWS sigv4 instead of an `x-api-key` header, so this backend signs every
//! request with the resolved AWS credentials and then delegates the actual
//! transport to the bundled `reqwest` backend.

use std::sync::Arc;
use std::time::SystemTime;

use aws_config::BehaviorVersion;
use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use bytes::Bytes;
use rig_core::http_client::{
    Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use tokio::sync::OnceCell;

use crate::client::DEFAULT_AWS_REGION;

/// The sigv4 service name the Mantle endpoint is signed for
const MANTLE_SIGNING_SERVICE: &str = "bedrock-mantle";

/// A `HttpClientExt` backend that sigv4-signs requests for Bedrock Mantle
///
/// Credentials resolve lazily on the first request: explicit credentials win,
/// then a configured AWS profile, then the environment-driven chain. The
/// signing region must match the region in the Mantle URL, so both come from
/// the same builder.
#[derive(Clone, Debug)]
pub struct MantleHttpClient {
    /// The transport the signed request is delegated to
    inner: reqwest::Client,
    /// The region requests are signed for
    region: String,
    /// The AWS profile to resolve credentials from, when no explicit ones exist
    profile_name: Option<String>,
    /// Explicit static credentials, outranking every other source
    credentials: Option<Credentials>,
    /// The lazily resolved credentials provider requests are signed with
    provider: Arc<OnceCell<SharedCredentialsProvider>>,
}

impl MantleHttpClient {
    /// Build a signing backend for a region and credential source
    ///
    /// # Arguments
    ///
    /// * `region` - The region requests are signed for
    /// * `profile_name` - The AWS profile to resolve credentials from, if any
    /// * `credentials` - Explicit static credentials, overriding the chain
    pub fn new(
        region: impl Into<String>,
        profile_name: Option<String>,
        credentials: Option<Credentials>,
    ) -> Self {
        Self {
            inner: reqwest::Client::default(),
            region: region.into(),
            profile_name,
            credentials,
            provider: Arc::new(OnceCell::new()),
        }
    }

    /// Resolve the credentials provider requests are signed with
    ///
    /// Resolved once and cached: explicit credentials, then the configured
    /// profile's chain, then the environment-driven chain.
    async fn credentials_provider(&self) -> Result<&SharedCredentialsProvider> {
        self.provider
            .get_or_try_init(|| async {
                match (&self.credentials, &self.profile_name) {
                    (Some(credentials), _) => {
                        Ok(SharedCredentialsProvider::new(credentials.clone()))
                    }
                    (None, profile_name) => {
                        // let the AWS chain find credentials, steered by the
                        // profile when one was configured
                        let config = if let Some(profile_name) = profile_name {
                            aws_config::defaults(BehaviorVersion::latest())
                                .profile_name(profile_name)
                                .load()
                                .await
                        } else {
                            aws_config::load_from_env().await
                        };
                        config.credentials_provider().ok_or_else(|| {
                            Error::Instance(Box::from(
                                "the AWS credential chain yielded no credentials provider",
                            ))
                        })
                    }
                }
            })
            .await
    }

    /// Sigv4-sign a request's parts and body for the Mantle endpoint
    ///
    /// # Arguments
    ///
    /// * `parts` - The request's method, URI, and headers
    /// * `body` - The request's final body bytes
    async fn signed_request(
        &self,
        parts: http::request::Parts,
        body: Bytes,
    ) -> Result<http::Request<Bytes>> {
        // resolve the actual credentials to sign with
        let credentials = self
            .credentials_provider()
            .await?
            .provide_credentials()
            .await
            .map_err(|err| Error::Instance(Box::new(err)))?;
        let identity: Identity = credentials.into();
        let params: SigningParams = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(MANTLE_SIGNING_SERVICE)
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|err| Error::Instance(Box::new(err)))?
            .into();
        // sign over the final headers and body bytes
        let signable = SignableRequest::new(
            parts.method.as_str(),
            parts.uri.to_string(),
            parts
                .headers
                .iter()
                .filter_map(|(key, value)| value.to_str().ok().map(|value| (key.as_str(), value))),
            SignableBody::Bytes(&body),
        )
        .map_err(|err| Error::Instance(Box::new(err)))?;
        let (instructions, _signature) = sign(signable, &params)
            .map_err(|err| Error::Instance(Box::new(err)))?
            .into_parts();
        // stamp the authorization/x-amz-* headers onto the real request
        let mut request = http::Request::from_parts(parts, body);
        instructions.apply_to_request_http1x(&mut request);
        Ok(request)
    }
}

impl Default for MantleHttpClient {
    /// An env-chain backend: region from `AWS_REGION`/`AWS_DEFAULT_REGION`
    /// (falling back to [`DEFAULT_AWS_REGION`]), credentials from the chain
    fn default() -> Self {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| DEFAULT_AWS_REGION.to_owned());
        Self::new(region, None, None)
    }
}

impl HttpClientExt for MantleHttpClient {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        // take the final body bytes now; signing covers them
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let this = self.clone();
        async move {
            let signed = this.signed_request(parts, body).await?;
            this.inner.send(signed).await
        }
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        // multipart bodies cannot be buffered for sigv4 signing here, and the
        // Messages API never sends them
        std::future::ready(Err(Error::Instance(Box::from(
            "multipart requests are not supported on the Bedrock Mantle transport",
        ))))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        // the request body is still plain bytes; only the response streams
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let this = self.clone();
        async move {
            let signed = this.signed_request(parts, body).await?;
            this.inner.send_streaming(signed).await
        }
    }
}
