use axum::Json;
use infers_api::{Model, ModelList};
use crate::state::SharedState;

pub async fn list_models(
    axum::extract::State(state): axum::extract::State<SharedState>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![Model {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "infers".to_string(),
        }],
    })
}
