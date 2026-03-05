use std::time::Duration;

/// Client-side errors for the HEBBS SDK.
///
/// Stable error taxonomy mapping gRPC status codes to actionable categories.
/// Each variant carries structured context to aid debugging.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Server is unreachable (DNS failure, connection refused, TLS handshake).
    #[error("cannot connect to {endpoint}: {reason}")]
    ConnectionFailed { endpoint: String, reason: String },

    /// Operation exceeded the configured timeout.
    #[error("operation '{operation}' timed out after {elapsed:?}")]
    Timeout {
        operation: &'static str,
        elapsed: Duration,
    },

    /// Requested resource does not exist.
    #[error("not found: {message}")]
    NotFound { message: String },

    /// Client sent invalid input (should rarely happen if using typed API).
    #[error("invalid input: {message}")]
    InvalidInput { message: String },

    /// Server returned an internal error (likely a bug).
    #[error("server error: {message}")]
    ServerError { message: String },

    /// Server is temporarily unavailable after all retries exhausted.
    #[error("server unavailable after {attempts} attempt(s): {message}")]
    Unavailable { message: String, attempts: u32 },

    /// Rate limited after all retries exhausted.
    #[error("rate limited after {attempts} attempt(s)")]
    RateLimited { attempts: u32 },

    /// Subscribe stream was closed unexpectedly.
    #[error("subscription closed: {reason}")]
    SubscriptionClosed { reason: String },

    /// Proto encoding/decoding failure (version mismatch between client and server).
    #[error("serialization error: {message}")]
    Serialization { message: String },

    /// Invalid configuration passed to the builder.
    #[error("invalid configuration: {message}")]
    InvalidConfig { message: String },
}

impl ClientError {
    /// Whether this error is likely transient and the operation could be retried.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            ClientError::Unavailable { .. }
                | ClientError::RateLimited { .. }
                | ClientError::Timeout { .. }
        )
    }
}

/// Map a tonic status to a ClientError.
pub(crate) fn from_status(status: tonic::Status, operation: &'static str) -> ClientError {
    match status.code() {
        tonic::Code::NotFound => ClientError::NotFound {
            message: status.message().to_string(),
        },
        tonic::Code::InvalidArgument => ClientError::InvalidInput {
            message: status.message().to_string(),
        },
        tonic::Code::Unavailable => ClientError::Unavailable {
            message: status.message().to_string(),
            attempts: 1,
        },
        tonic::Code::ResourceExhausted => ClientError::RateLimited { attempts: 1 },
        tonic::Code::DeadlineExceeded => ClientError::Timeout {
            operation,
            elapsed: Duration::ZERO,
        },
        tonic::Code::Internal => ClientError::ServerError {
            message: status.message().to_string(),
        },
        _ => ClientError::ServerError {
            message: format!("{}: {}", status.code(), status.message()),
        },
    }
}

/// Classify whether a gRPC status code is retryable.
pub(crate) fn is_retryable_status(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unavailable
            | tonic::Code::ResourceExhausted
            | tonic::Code::Aborted
            | tonic::Code::DeadlineExceeded
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_errors_detected() {
        let err = ClientError::Unavailable {
            message: "down".into(),
            attempts: 3,
        };
        assert!(err.is_transient());

        let err = ClientError::RateLimited { attempts: 2 };
        assert!(err.is_transient());

        let err = ClientError::NotFound {
            message: "gone".into(),
        };
        assert!(!err.is_transient());
    }

    #[test]
    fn from_status_not_found() {
        let s = tonic::Status::not_found("memory 123 not found");
        let e = from_status(s, "get");
        assert!(matches!(e, ClientError::NotFound { .. }));
    }

    #[test]
    fn from_status_invalid_argument() {
        let s = tonic::Status::invalid_argument("bad input");
        let e = from_status(s, "remember");
        assert!(matches!(e, ClientError::InvalidInput { .. }));
    }

    #[test]
    fn from_status_unavailable() {
        let s = tonic::Status::unavailable("server starting");
        let e = from_status(s, "recall");
        assert!(matches!(e, ClientError::Unavailable { .. }));
    }

    #[test]
    fn from_status_internal() {
        let s = tonic::Status::internal("panic");
        let e = from_status(s, "reflect");
        assert!(matches!(e, ClientError::ServerError { .. }));
    }

    #[test]
    fn from_status_resource_exhausted() {
        let s = tonic::Status::resource_exhausted("rate limit");
        let e = from_status(s, "remember");
        assert!(matches!(e, ClientError::RateLimited { .. }));
    }

    #[test]
    fn from_status_deadline_exceeded() {
        let s = tonic::Status::deadline_exceeded("timeout");
        let e = from_status(s, "recall");
        assert!(matches!(e, ClientError::Timeout { .. }));
    }

    #[test]
    fn retryable_status_codes() {
        assert!(is_retryable_status(tonic::Code::Unavailable));
        assert!(is_retryable_status(tonic::Code::ResourceExhausted));
        assert!(is_retryable_status(tonic::Code::Aborted));
        assert!(is_retryable_status(tonic::Code::DeadlineExceeded));

        assert!(!is_retryable_status(tonic::Code::NotFound));
        assert!(!is_retryable_status(tonic::Code::InvalidArgument));
        assert!(!is_retryable_status(tonic::Code::Internal));
        assert!(!is_retryable_status(tonic::Code::Ok));
    }

    #[test]
    fn error_display_messages() {
        let err = ClientError::ConnectionFailed {
            endpoint: "localhost:6380".into(),
            reason: "connection refused".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("localhost:6380"));
        assert!(msg.contains("connection refused"));

        let err = ClientError::Timeout {
            operation: "recall",
            elapsed: Duration::from_millis(500),
        };
        assert!(err.to_string().contains("recall"));
    }
}
