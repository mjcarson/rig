//! A Google-OAuth-authenticating HTTP backend for Claude on Vertex AI
//!
//! Vertex AI serves the Anthropic Messages API but authenticates with a Google
//! Cloud OAuth2 bearer token instead of an `x-api-key` header, so this backend
//! stamps the resolved credentials' auth headers onto every request and then
//! delegates the actual transport to the bundled `reqwest` backend.

use std::sync::Arc;

use bytes::Bytes;
use google_cloud_auth::credentials::{CacheableResource, Credentials};
use rig_core::http_client::{
    Error, HttpClientExt, LazyBody, MultipartForm, Request, Response, Result, StreamingResponse,
};
use rig_core::wasm_compat::WasmCompatSend;
use tokio::sync::OnceCell;

use crate::client::build_credentials;

/// Where the backend gets its Google Cloud credentials from
#[derive(Clone, Default)]
pub(crate) enum CredentialSource {
    /// Pre-built credentials handed in by the caller
    Explicit(Credentials),
    /// A service account key to build credentials from on first use
    ServiceAccountKey(serde_json::Value),
    /// The Application Default Credentials chain
    #[default]
    Chain,
}

impl std::fmt::Debug for CredentialSource {
    /// Redacted by hand: a service account key carries a private key
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredentialSource::Explicit(_) => f.write_str("Explicit"),
            CredentialSource::ServiceAccountKey(_) => f.write_str("ServiceAccountKey(redacted)"),
            CredentialSource::Chain => f.write_str("Chain"),
        }
    }
}

/// A `HttpClientExt` backend that bearer-authenticates requests for Vertex AI
///
/// Credentials resolve lazily on the first request: explicit credentials win,
/// then a configured service account key, then the Application Default
/// Credentials chain (honoring `GOOGLE_CLOUD_SERVICE_ACCOUNT` impersonation).
/// The resolved credentials cache and refresh their token in the background,
/// so stamping headers per request is cheap.
#[derive(Clone, Debug, Default)]
pub struct VertexHttpClient {
    /// The transport the authenticated request is delegated to
    inner: reqwest::Client,
    /// The credential source to resolve on first use
    source: CredentialSource,
    /// The lazily resolved credentials requests are authenticated with
    credentials: Arc<OnceCell<Credentials>>,
}

impl VertexHttpClient {
    /// Build an authenticating backend for a credential source
    ///
    /// # Arguments
    ///
    /// * `source` - Where to get Google Cloud credentials from on first use
    pub(crate) fn new(source: CredentialSource) -> Self {
        Self {
            inner: reqwest::Client::default(),
            source,
            credentials: Arc::new(OnceCell::new()),
        }
    }

    /// Resolve the credentials requests are authenticated with
    ///
    /// Resolved once and cached: explicit credentials, then a service account
    /// key, then the Application Default Credentials chain.
    async fn credentials(&self) -> Result<&Credentials> {
        // the "failed to load credentials" prose is deliberate: it is the
        // credential-shaped marker provider-agnostic retry layers key off for
        // status-less transport failures, so a dead key pauses instead of
        // retrying forever
        self.credentials
            .get_or_try_init(|| async {
                match &self.source {
                    CredentialSource::Explicit(credentials) => Ok(credentials.clone()),
                    CredentialSource::ServiceAccountKey(key) => {
                        // build credentials from the configured key json
                        google_cloud_auth::credentials::service_account::Builder::new(key.clone())
                            .build()
                            .map_err(|err| {
                                Error::Instance(Box::from(format!(
                                    "failed to load credentials from the service account key: {err}"
                                )))
                            })
                    }
                    CredentialSource::Chain => {
                        // let the shared chain helper find credentials
                        build_credentials(None).map_err(|err| {
                            Error::Instance(Box::from(format!(
                                "failed to load credentials from the Google Cloud chain: {err}"
                            )))
                        })
                    }
                }
            })
            .await
    }

    /// Stamp the credentials' auth headers onto a request
    ///
    /// # Arguments
    ///
    /// * `req` - The request to authenticate
    async fn authorize<T>(&self, mut req: Request<T>) -> Result<Request<T>>
    where
        T: WasmCompatSend,
    {
        // resolve the actual credentials to authenticate with
        let credentials = self.credentials().await?;
        // mint the auth headers; the credentials cache the underlying token
        let headers = credentials
            .headers(http::Extensions::new())
            .await
            .map_err(|err| Error::Instance(Box::new(err)))?;
        // without an entity tag the credentials always yield fresh headers
        let CacheableResource::New { data, .. } = headers else {
            return Err(Error::Instance(Box::from(
                "the Google Cloud credentials returned no auth headers",
            )));
        };
        // stamp the authorization (and any project) headers onto the request
        req.headers_mut().extend(data);
        Ok(req)
    }
}

impl HttpClientExt for VertexHttpClient {
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
        // pin the body down to bytes now so the future owns no borrowed input
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let this = self.clone();
        async move {
            let req = http::Request::from_parts(parts, body);
            let authorized = this.authorize(req).await?;
            this.inner.send(authorized).await
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
        // the Messages API never sends multipart bodies
        std::future::ready(Err(Error::Instance(Box::from(
            "multipart requests are not supported on the Vertex AI Anthropic transport",
        ))))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        // the request body is still plain bytes; only the response streams,
        // and each SSE (re)connect re-stamps freshly minted auth headers
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let this = self.clone();
        async move {
            let req = http::Request::from_parts(parts, body);
            let authorized = this.authorize(req).await?;
            this.inner.send_streaming(authorized).await
        }
    }
}
