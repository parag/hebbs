use std::time::Duration;

use hebbs_proto::generated::{
    health_service_client::HealthServiceClient, memory_service_client::MemoryServiceClient,
    reflect_service_client::ReflectServiceClient, subscribe_service_client::SubscribeServiceClient,
};
use tonic::transport::Channel;

use crate::error::CliError;

/// Manages the gRPC connection to a HEBBS server.
/// Lazily connects on first use and supports reconnection.
pub struct ConnectionManager {
    endpoint: String,
    timeout: Duration,
    channel: Option<Channel>,
    api_key: Option<String>,
}

impl ConnectionManager {
    pub fn new(endpoint: String, timeout_ms: u64) -> Self {
        Self {
            endpoint,
            timeout: Duration::from_millis(timeout_ms),
            channel: None,
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn is_connected(&self) -> bool {
        self.channel.is_some()
    }

    pub fn set_endpoint(&mut self, endpoint: String) {
        self.endpoint = endpoint;
        self.channel = None;
    }

    pub fn disconnect(&mut self) {
        self.channel = None;
    }

    async fn ensure_channel(&mut self) -> Result<Channel, CliError> {
        if let Some(ref ch) = self.channel {
            return Ok(ch.clone());
        }

        let channel = Channel::from_shared(self.endpoint.clone())
            .map_err(|e| CliError::ConnectionFailed {
                endpoint: self.endpoint.clone(),
                source: e.to_string(),
            })?
            .timeout(self.timeout)
            .connect_timeout(Duration::from_secs(5))
            .connect()
            .await
            .map_err(|e| CliError::ConnectionFailed {
                endpoint: self.endpoint.clone(),
                source: e.to_string(),
            })?;

        self.channel = Some(channel.clone());
        Ok(channel)
    }

    pub async fn memory_client(&mut self) -> Result<MemoryServiceClient<Channel>, CliError> {
        let ch = self.ensure_channel().await?;
        Ok(MemoryServiceClient::new(ch))
    }

    pub async fn subscribe_client(&mut self) -> Result<SubscribeServiceClient<Channel>, CliError> {
        let ch = self.ensure_channel().await?;
        Ok(SubscribeServiceClient::new(ch))
    }

    pub async fn reflect_client(&mut self) -> Result<ReflectServiceClient<Channel>, CliError> {
        let ch = self.ensure_channel().await?;
        Ok(ReflectServiceClient::new(ch))
    }

    pub async fn health_client(&mut self) -> Result<HealthServiceClient<Channel>, CliError> {
        let ch = self.ensure_channel().await?;
        Ok(HealthServiceClient::new(ch))
    }

    /// Mark connection as potentially broken (e.g., after an Unavailable error).
    /// Next call will reconnect.
    pub fn mark_disconnected(&mut self) {
        self.channel = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_connection_is_disconnected() {
        let cm = ConnectionManager::new("http://localhost:6380".to_string(), 30000);
        assert!(!cm.is_connected());
        assert_eq!(cm.endpoint(), "http://localhost:6380");
        assert!(cm.api_key().is_none());
    }

    #[test]
    fn set_endpoint_clears_channel() {
        let mut cm = ConnectionManager::new("http://localhost:6380".to_string(), 30000);
        cm.set_endpoint("http://localhost:7000".to_string());
        assert!(!cm.is_connected());
        assert_eq!(cm.endpoint(), "http://localhost:7000");
    }

    #[test]
    fn disconnect_clears_channel() {
        let mut cm = ConnectionManager::new("http://localhost:6380".to_string(), 30000);
        cm.disconnect();
        assert!(!cm.is_connected());
    }
}
