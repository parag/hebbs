use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

pub struct HebbsMetrics {
    pub registry: Registry,
    pub operation_duration: HistogramVec,
    pub memory_count: IntGauge,
    pub active_subscriptions: IntGauge,
    pub reflect_runs: IntCounterVec,
    pub errors_total: IntCounterVec,
    pub grpc_requests: IntCounterVec,
    pub http_requests: IntCounterVec,
}

impl Default for HebbsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl HebbsMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0,
        ];

        let operation_duration = HistogramVec::new(
            HistogramOpts::new(
                "hebbs_operation_duration_seconds",
                "Latency of HEBBS operations",
            )
            .buckets(buckets),
            &["operation", "status"],
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(operation_duration.clone()))
            .expect("metric registration must succeed");

        let memory_count = IntGauge::new("hebbs_memory_count", "Total stored memories")
            .expect("metric creation must succeed");
        registry
            .register(Box::new(memory_count.clone()))
            .expect("metric registration must succeed");

        let active_subscriptions = IntGauge::new(
            "hebbs_active_subscriptions",
            "Currently active subscribe streams",
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(active_subscriptions.clone()))
            .expect("metric registration must succeed");

        let reflect_runs = IntCounterVec::new(
            Opts::new("hebbs_reflect_runs_total", "Reflect run count"),
            &["status"],
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(reflect_runs.clone()))
            .expect("metric registration must succeed");

        let errors_total = IntCounterVec::new(
            Opts::new("hebbs_errors_total", "Error count by type"),
            &["operation", "error_type"],
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(errors_total.clone()))
            .expect("metric registration must succeed");

        let grpc_requests = IntCounterVec::new(
            Opts::new("hebbs_grpc_requests_total", "gRPC requests by method"),
            &["method", "status"],
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(grpc_requests.clone()))
            .expect("metric registration must succeed");

        let http_requests = IntCounterVec::new(
            Opts::new("hebbs_http_requests_total", "HTTP requests"),
            &["method", "path", "status_code"],
        )
        .expect("metric creation must succeed");
        registry
            .register(Box::new(http_requests.clone()))
            .expect("metric registration must succeed");

        Self {
            registry,
            operation_duration,
            memory_count,
            active_subscriptions,
            reflect_runs,
            errors_total,
            grpc_requests,
            http_requests,
        }
    }

    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap_or(());
        String::from_utf8(buffer).unwrap_or_default()
    }

    pub fn observe_operation(&self, operation: &str, status: &str, duration_secs: f64) {
        self.operation_duration
            .with_label_values(&[operation, status])
            .observe(duration_secs);
    }

    pub fn record_error(&self, operation: &str, error_type: &str) {
        self.errors_total
            .with_label_values(&[operation, error_type])
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_render_not_empty() {
        let m = HebbsMetrics::new();
        m.observe_operation("remember", "ok", 0.001);
        let rendered = m.render();
        assert!(rendered.contains("hebbs_operation_duration_seconds"));
    }

    #[test]
    fn metrics_counter_increments() {
        let m = HebbsMetrics::new();
        m.record_error("recall", "invalid_input");
        m.record_error("recall", "invalid_input");
        let rendered = m.render();
        assert!(rendered.contains("hebbs_errors_total"));
    }
}
