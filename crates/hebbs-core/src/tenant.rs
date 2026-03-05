/// Tenant context propagated through every engine operation.
///
/// In single-tenant mode (embedded, `--no-auth`), the default tenant `"default"` is used.
/// In multi-tenant mode, the server middleware resolves the tenant from the API key.
/// Every deployment is multi-tenant; single-tenant deployments have exactly one tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantContext {
    tenant_id: String,
}

/// The implicit tenant for single-tenant deployments and `--no-auth` mode.
pub const DEFAULT_TENANT_ID: &str = "default";

/// Maximum length for tenant IDs (alphanumeric + hyphens, 1-128 chars).
pub const MAX_TENANT_ID_LENGTH: usize = 128;

impl TenantContext {
    pub fn new(tenant_id: impl Into<String>) -> Result<Self, String> {
        let id = tenant_id.into();
        validate_tenant_id(&id)?;
        Ok(Self { tenant_id: id })
    }

    /// Create a TenantContext without validation. For internal use where
    /// the tenant_id is known to be valid (e.g., loaded from storage).
    pub fn new_unchecked(tenant_id: String) -> Self {
        Self { tenant_id }
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }
}

impl Default for TenantContext {
    fn default() -> Self {
        Self {
            tenant_id: DEFAULT_TENANT_ID.to_string(),
        }
    }
}

fn validate_tenant_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("tenant_id must not be empty".to_string());
    }
    if id.len() > MAX_TENANT_ID_LENGTH {
        return Err(format!(
            "tenant_id length {} exceeds maximum {}",
            id.len(),
            MAX_TENANT_ID_LENGTH
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!(
            "tenant_id '{}' contains invalid characters (allowed: alphanumeric, hyphens, underscores)",
            id
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tenant_context() {
        let ctx = TenantContext::default();
        assert_eq!(ctx.tenant_id(), DEFAULT_TENANT_ID);
    }

    #[test]
    fn valid_tenant_ids() {
        assert!(TenantContext::new("acme").is_ok());
        assert!(TenantContext::new("tenant-123").is_ok());
        assert!(TenantContext::new("my_org_42").is_ok());
        assert!(TenantContext::new("a").is_ok());
    }

    #[test]
    fn invalid_tenant_ids() {
        assert!(TenantContext::new("").is_err());
        assert!(TenantContext::new("a".repeat(129)).is_err());
        assert!(TenantContext::new("has spaces").is_err());
        assert!(TenantContext::new("has.dots").is_err());
        assert!(TenantContext::new("has/slash").is_err());
    }
}
