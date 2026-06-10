use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// Errors returned by the inference API.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Internal server error: {0}")]
    Internal(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Service overloaded")]
    Overloaded,
}

/// JSON error response body.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Detailed error information in API responses.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, code) = match &self {
            ApiError::BadRequest(_) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("bad_request".to_string()),
            ),
            ApiError::NotFound(_) => (
                StatusCode::NOT_FOUND,
                "not_found_error",
                Some("not_found".to_string()),
            ),
            ApiError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                None,
            ),
            ApiError::ModelNotFound(_) => (
                StatusCode::NOT_FOUND,
                "not_found_error",
                Some("model_not_found".to_string()),
            ),
            ApiError::RateLimit => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("rate_limit_exceeded".to_string()),
            ),
            ApiError::Overloaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                Some("overloaded".to_string()),
            ),
        };

        let body = ErrorResponse {
            error: ErrorDetail {
                message: self.to_string(),
                error_type: error_type.to_string(),
                code,
            },
        };

        (status, axum::Json(body)).into_response()
    }
}
