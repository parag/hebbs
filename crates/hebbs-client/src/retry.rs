use std::time::Duration;

use rand::Rng;

/// Configuration for retry behavior on idempotent operations.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 disables retry).
    pub max_retries: u32,
    /// Initial backoff duration before first retry.
    pub initial_backoff: Duration,
    /// Backoff multiplier applied after each attempt.
    pub backoff_multiplier: f64,
    /// Maximum backoff duration cap.
    pub max_backoff: Duration,
    /// Jitter factor in [0.0, 1.0]. Applied as ±jitter to the computed delay.
    pub jitter: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(50),
            backoff_multiplier: 4.0,
            max_backoff: Duration::from_secs(5),
            jitter: 0.25,
        }
    }
}

impl RetryPolicy {
    /// No retries at all.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Compute the delay before the nth retry (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base =
            self.initial_backoff.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        let capped = base.min(self.max_backoff.as_secs_f64());

        let mut rng = rand::thread_rng();
        let jitter_range = capped * self.jitter;
        let jitter_offset = rng.gen_range(-jitter_range..=jitter_range);
        let final_delay = (capped + jitter_offset).max(0.001);

        Duration::from_secs_f64(final_delay)
    }
}

/// Whether a given operation name is safe to retry.
pub fn is_idempotent(operation: &str) -> bool {
    matches!(
        operation,
        "get" | "recall" | "prime" | "insights" | "health"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_values() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.initial_backoff, Duration::from_millis(50));
        assert!((p.backoff_multiplier - 4.0).abs() < f64::EPSILON);
        assert_eq!(p.max_backoff, Duration::from_secs(5));
    }

    #[test]
    fn no_retry_policy() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_retries, 0);
    }

    #[test]
    fn delay_increases_with_attempts() {
        let p = RetryPolicy {
            jitter: 0.0,
            ..Default::default()
        };
        let d0 = p.delay_for_attempt(0);
        let d1 = p.delay_for_attempt(1);
        let d2 = p.delay_for_attempt(2);
        assert!(d1 > d0, "d1={:?} should be > d0={:?}", d1, d0);
        assert!(d2 > d1, "d2={:?} should be > d1={:?}", d2, d1);
    }

    #[test]
    fn delay_capped_at_max() {
        let p = RetryPolicy {
            max_backoff: Duration::from_millis(100),
            jitter: 0.0,
            ..Default::default()
        };
        let d = p.delay_for_attempt(10);
        assert!(d <= Duration::from_millis(101));
    }

    #[test]
    fn delay_with_jitter_bounded() {
        let p = RetryPolicy::default();
        for attempt in 0..5 {
            let d = p.delay_for_attempt(attempt);
            assert!(d <= p.max_backoff + Duration::from_secs(2));
            assert!(d >= Duration::from_micros(1));
        }
    }

    #[test]
    fn idempotent_operations() {
        assert!(is_idempotent("get"));
        assert!(is_idempotent("recall"));
        assert!(is_idempotent("prime"));
        assert!(is_idempotent("insights"));
        assert!(is_idempotent("health"));

        assert!(!is_idempotent("remember"));
        assert!(!is_idempotent("revise"));
        assert!(!is_idempotent("forget"));
        assert!(!is_idempotent("reflect"));
        assert!(!is_idempotent("subscribe"));
    }
}
