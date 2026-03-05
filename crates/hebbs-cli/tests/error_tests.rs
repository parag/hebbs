use hebbs_cli::error::*;

#[test]
fn connection_failed_exit_code() {
    let err = CliError::ConnectionFailed {
        endpoint: "http://localhost:6380".to_string(),
        source: "connection refused".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_CONNECTION_ERROR);
    assert!(err.to_string().contains("localhost:6380"));
    assert!(err.to_string().contains("Is hebbs-server running"));
}

#[test]
fn server_unavailable_exit_code() {
    let err = CliError::ServerUnavailable {
        endpoint: "http://localhost:6380".to_string(),
        message: "service unavailable".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_CONNECTION_ERROR);
}

#[test]
fn not_found_exit_code() {
    let err = CliError::NotFound {
        message: "Memory not found".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_NOT_FOUND);
    assert_eq!(err.to_string(), "Memory not found");
}

#[test]
fn invalid_argument_exit_code() {
    let err = CliError::InvalidArgument {
        message: "bad input".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_USAGE_ERROR);
}

#[test]
fn server_error_exit_code() {
    let err = CliError::ServerError {
        message: "internal error".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_SERVER_ERROR);
    assert!(err.to_string().contains("may be a bug"));
}

#[test]
fn deadline_exceeded_exit_code() {
    let err = CliError::DeadlineExceeded { timeout_ms: 5000 };
    assert_eq!(err.exit_code(), EXIT_SERVER_ERROR);
    assert!(err.to_string().contains("5000ms"));
    assert!(err.to_string().contains("--timeout"));
}

#[test]
fn internal_error_exit_code() {
    let err = CliError::Internal {
        message: "unexpected".to_string(),
    };
    assert_eq!(err.exit_code(), EXIT_GENERAL_ERROR);
}

#[test]
fn from_status_unavailable() {
    let status = tonic::Status::unavailable("server down");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_CONNECTION_ERROR);
}

#[test]
fn from_status_not_found() {
    let status = tonic::Status::not_found("memory not found");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_NOT_FOUND);
}

#[test]
fn from_status_invalid_argument() {
    let status = tonic::Status::invalid_argument("bad request");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_USAGE_ERROR);
}

#[test]
fn from_status_deadline() {
    let status = tonic::Status::deadline_exceeded("timeout");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_SERVER_ERROR);
}

#[test]
fn from_status_internal() {
    let status = tonic::Status::internal("crash");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_SERVER_ERROR);
}

#[test]
fn from_status_unknown() {
    let status = tonic::Status::cancelled("cancelled");
    let err = CliError::from_status(status, "http://localhost:6380");
    assert_eq!(err.exit_code(), EXIT_SERVER_ERROR);
}

#[test]
fn exit_code_constants_are_distinct() {
    let codes = [
        EXIT_SUCCESS,
        EXIT_GENERAL_ERROR,
        EXIT_USAGE_ERROR,
        EXIT_CONNECTION_ERROR,
        EXIT_NOT_FOUND,
        EXIT_SERVER_ERROR,
    ];
    for i in 0..codes.len() {
        for j in (i + 1)..codes.len() {
            assert_ne!(codes[i], codes[j], "Exit codes must be distinct");
        }
    }
}
