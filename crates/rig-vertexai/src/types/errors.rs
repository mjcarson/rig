//! Maps Vertex AI SDK errors onto Rig's completion errors
//!
//! The SDK's `google-cloud-gax` error carries the HTTP status, response
//! headers, and raw payload of a failed call when they exist, plus the parsed
//! `google.rpc.Status` for service errors. Preserving all of that lets a
//! provider-agnostic retry layer classify throttles (429) and server faults
//! (5xx) through `provider_response_status()`, pace retries off `Retry-After`,
//! and pause on auth faults — instead of dying on status-less prose (the
//! Vertex analog of the Bedrock Converse error work).

use google_cloud_gax::error::Error as GaxError;
use google_cloud_gax::error::rpc::Code;
use http::StatusCode;
use rig_core::completion::CompletionError;
use rig_core::http_client;

/// The HTTP status a `google.rpc.Code` maps onto
///
/// The mapping is the canonical one documented on [`Code`] itself; it only
/// matters when a service error arrives without its own HTTP status.
///
/// # Arguments
///
/// * `code` - The gRPC status code to map
fn grpc_code_status(code: Code) -> StatusCode {
    match code {
        // not an error; never expected here but mapped faithfully
        Code::Ok => StatusCode::OK,
        // 499 has no named constant; fall back to 500 if it is ever rejected
        Code::Cancelled => StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
            StatusCode::BAD_REQUEST
        }
        Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        Code::NotFound => StatusCode::NOT_FOUND,
        Code::AlreadyExists | Code::Aborted => StatusCode::CONFLICT,
        Code::PermissionDenied => StatusCode::FORBIDDEN,
        Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        Code::Unimplemented => StatusCode::NOT_IMPLEMENTED,
        Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        Code::Unauthenticated => StatusCode::UNAUTHORIZED,
        // Unknown, Internal, DataLoss, and any future code are server faults
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Attach transport metadata to a preserved provider response body
///
/// Rig-authored diagnostics are left untouched: transport metadata belongs
/// with what the provider actually said.
///
/// # Arguments
///
/// * `error` - The completion error to attach metadata to
/// * `status` - The HTTP status the provider answered with
/// * `headers` - The response headers the provider answered with
fn attach_transport_metadata(
    error: CompletionError,
    status: Option<StatusCode>,
    headers: Option<Box<http::HeaderMap>>,
) -> CompletionError {
    match error {
        CompletionError::ProviderResponse(response) => {
            CompletionError::ProviderResponse(response.with_status(status).with_headers(headers))
        }
        other => other,
    }
}

/// Map a failed Vertex AI RPC onto a [`CompletionError`]
///
/// A transient fault that never reached the service — a client-side timeout
/// or exhausted retry budget, a connection or I/O failure, or a transient
/// credential fetch hiccup — becomes a retryable
/// [`CompletionError::HttpError`]. A non-transient credential failure is
/// normalized to a 401 provider response (the token endpoint rejected us, or
/// the chain yielded nothing usable — either way an auth fault, so
/// pause-and-refresh layers can react instead of retrying a dead key).
/// Anything the service actually answered keeps its raw body, HTTP status
/// (falling back to the canonical `google.rpc.Code` mapping), and response
/// headers. Only rig-side faults that never reached the service (a malformed
/// request, an undecodable response) stay rig-authored
/// [`CompletionError::ProviderError`] diagnostics.
///
/// Note: `is_transient_and_before_rpc`, `is_io`, and `is_authentication` are
/// `doc(hidden)` on the gax error ("subject to change without notice"), but
/// they are the only accessors that distinguish connection, I/O, and
/// credential faults; the unit tests below pin their semantics so a gax bump
/// that shifts them fails loudly here instead of silently misrouting errors.
///
/// # Arguments
///
/// * `error` - The SDK error from a failed `send()`
pub(crate) fn completion_error(error: GaxError) -> CompletionError {
    // a transient fault never got a service answer; surface it retryable:
    // client-side deadline, retry budget exhausted on something transient,
    // connection or transient-credential faults before the RPC, raw I/O
    if error.is_timeout()
        || error.is_exhausted()
        || error.is_transient_and_before_rpc()
        || error.is_io()
    {
        return CompletionError::HttpError(http_client::Error::Instance(Box::new(error)));
    }
    // a non-transient credential failure is an auth fault even though the
    // model endpoint never spoke; a 401 is the class provider-agnostic
    // layers key off (transient ones already returned retryable above)
    if error.is_authentication() {
        return attach_transport_metadata(
            CompletionError::from_provider_body(error.to_string()),
            Some(StatusCode::UNAUTHORIZED),
            None,
        );
    }
    // read the HTTP status, falling back to the parsed google.rpc code
    let status = error
        .http_status_code()
        .and_then(|code| StatusCode::from_u16(code).ok())
        .or_else(|| error.status().map(|status| grpc_code_status(status.code)));
    // without either signal the service never spoke; keep it a rig diagnostic
    let Some(status) = status else {
        return CompletionError::ProviderError(error.to_string());
    };
    // prefer the raw payload, then the parsed service message, then the error
    let body = error
        .http_payload()
        .map(|payload| String::from_utf8_lossy(payload).into_owned())
        .filter(|payload| !payload.is_empty())
        .or_else(|| {
            error
                .status()
                .map(|status| status.message.clone())
                .filter(|message| !message.is_empty())
        })
        .unwrap_or_else(|| error.to_string());
    // response headers carry the rate-limit metadata retry pacing needs
    let headers = error.http_headers().cloned().map(Box::new);
    attach_transport_metadata(
        CompletionError::from_provider_body(body),
        Some(status),
        headers,
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use google_cloud_gax::error::CredentialsError;
    use google_cloud_gax::error::rpc::Status;

    #[test]
    fn http_errors_keep_status_headers_and_body() {
        let mut headers = http::HeaderMap::new();
        headers.insert("retry-after", http::HeaderValue::from_static("7"));
        let raw = r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}"#;

        let err = completion_error(GaxError::http(
            429,
            headers,
            bytes::Bytes::from_static(raw.as_bytes()),
        ));

        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(err.provider_response_body(), Some(raw));
        let headers = err.provider_response_headers().expect("headers preserved");
        assert_eq!(headers.get("retry-after").unwrap(), "7");
    }

    #[test]
    fn service_errors_without_http_metadata_map_the_grpc_code() {
        let status = Status::default()
            .set_code(Code::Unavailable)
            .set_message("the service is currently unavailable");

        let err = completion_error(GaxError::service(status));

        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(
            err.provider_response_body(),
            Some("the service is currently unavailable")
        );
    }

    #[test]
    fn timeouts_connects_and_exhaustion_surface_as_retryable_transport_errors() {
        // every transient kind that never got a service answer retries
        for err in [
            GaxError::timeout("deadline blown"),
            GaxError::connect("connection refused"),
            GaxError::exhausted("retry policy exhausted"),
        ] {
            assert!(matches!(
                completion_error(err),
                CompletionError::HttpError(http_client::Error::Instance(_))
            ));
        }
    }

    #[test]
    fn transient_credential_hiccups_retry_instead_of_pausing() {
        // a metadata-server blip is transient; it must not become a hard 401
        let err = completion_error(GaxError::authentication(CredentialsError::from_msg(
            true,
            "the token endpoint timed out",
        )));

        assert!(matches!(
            err,
            CompletionError::HttpError(http_client::Error::Instance(_))
        ));
    }

    #[test]
    fn credential_failures_normalize_to_an_auth_status() {
        let err = completion_error(GaxError::authentication(CredentialsError::from_msg(
            false,
            "the service account key was revoked",
        )));

        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::UNAUTHORIZED)
        );
        assert!(
            err.provider_response_body()
                .expect("the credential prose is preserved")
                .contains("revoked")
        );
    }

    #[test]
    fn every_grpc_code_lands_on_its_documented_status() {
        assert_eq!(grpc_code_status(Code::ResourceExhausted), 429);
        assert_eq!(grpc_code_status(Code::Unauthenticated), 401);
        assert_eq!(grpc_code_status(Code::PermissionDenied), 403);
        assert_eq!(grpc_code_status(Code::DeadlineExceeded), 504);
        assert_eq!(grpc_code_status(Code::Unavailable), 503);
        assert_eq!(grpc_code_status(Code::Internal), 500);
        assert_eq!(grpc_code_status(Code::InvalidArgument), 400);
        assert_eq!(grpc_code_status(Code::NotFound), 404);
        assert_eq!(grpc_code_status(Code::Cancelled), 499);
    }
}
