use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use hebbs_core::engine::Engine;
use hebbs_proto::generated as pb;
use pb::health_check_response::ServingStatus;
use pb::health_service_server::HealthService;

pub struct HealthServiceImpl {
    pub engine: Arc<Engine>,
    pub start_time: Instant,
    pub version: String,
}

#[tonic::async_trait]
impl HealthService for HealthServiceImpl {
    async fn check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        let engine = self.engine.clone();
        let count = tokio::task::spawn_blocking(move || engine.count())
            .await
            .map_err(|e| Status::internal(format!("task join error: {}", e)))?
            .unwrap_or(0);

        let uptime = self.start_time.elapsed().as_secs();

        Ok(Response::new(pb::HealthCheckResponse {
            status: ServingStatus::Serving as i32,
            version: self.version.clone(),
            memory_count: count as u64,
            uptime_seconds: uptime,
        }))
    }
}
