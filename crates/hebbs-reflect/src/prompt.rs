use crate::types::{CandidateInsight, MemoryEntry};

const CHARS_PER_TOKEN: usize = 4;

/// Build the proposal prompt for a single cluster.
///
/// Returns `(system_message, user_message)`.
pub fn build_proposal_prompt(
    cluster_memories: &[&MemoryEntry],
    centroid: &[f32],
    max_tokens: usize,
) -> (String, String) {
    let system = "\
You are an expert analyst. You receive a cluster of related memories \
from an AI agent's experience log. Your task is to identify recurring \
patterns, consolidated knowledge, or actionable principles across \
these memories.\n\n\
Respond with valid JSON in this exact format:\n\
{\"insights\": [{\"content\": \"...\", \"confidence\": 0.0-1.0, \
\"source_memory_ids\": [\"hex_id\", ...], \"tags\": [\"tag\", ...]}]}\n\n\
Rules:\n\
- confidence must be between 0.0 and 1.0\n\
- source_memory_ids must be a subset of the provided memory IDs\n\
- If no meaningful insight can be derived, return {\"insights\": []}\n\
- Each insight should be a concise, actionable statement"
        .to_string();

    let budget_chars = max_tokens * CHARS_PER_TOKEN;
    let mut user_parts: Vec<String> = Vec::new();
    let header = format!("Cluster of {} related memories:\n", cluster_memories.len());
    user_parts.push(header);

    let mut total_chars = user_parts[0].len();

    let mut scored: Vec<(usize, f64)> = cluster_memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let sim: f64 = m
                .embedding
                .iter()
                .zip(centroid.iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            (i, sim)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, _sim) in scored {
        let m = cluster_memories[idx];
        let id_hex = hex::encode(m.id);
        let entry = format!(
            "\n---\nID: {}\nImportance: {:.2}\nEntity: {}\nContent: {}\n",
            id_hex,
            m.importance,
            m.entity_id.as_deref().unwrap_or("(none)"),
            &m.content,
        );
        if total_chars + entry.len() > budget_chars {
            break;
        }
        total_chars += entry.len();
        user_parts.push(entry);
    }

    (system, user_parts.join(""))
}

/// Build the validation prompt for a set of candidate insights.
///
/// Returns `(system_message, user_message)`.
pub fn build_validation_prompt(
    candidates: &[CandidateInsight],
    source_memories: &[&MemoryEntry],
    existing_insights: &[&MemoryEntry],
    max_tokens: usize,
) -> (String, String) {
    let system = "\
You are a rigorous evaluator. You receive candidate insights derived from \
an agent's memories, the source memories they were derived from, and any \
existing insights for the same scope.\n\n\
For each candidate, evaluate accuracy against source memories and check \
for contradiction or duplication with existing insights.\n\n\
Respond with valid JSON:\n\
{\"results\": [{\"candidate_index\": 0, \"verdict\": \"accepted\"|\"rejected\"|\"revised\"|\"merged\", \
\"confidence\": 0.0-1.0, ...}]}\n\n\
Verdict details:\n\
- \"accepted\": the insight is accurate and non-redundant\n\
- \"rejected\": include \"reason\" field\n\
- \"revised\": include \"revised_content\" field with corrected text\n\
- \"merged\": include \"existing_id\" field (hex) of the existing insight it duplicates"
        .to_string();

    let budget_chars = max_tokens * CHARS_PER_TOKEN;
    let mut parts: Vec<String> = Vec::new();

    parts.push(format!("\n## {} Candidate Insights:\n", candidates.len()));
    for (i, c) in candidates.iter().enumerate() {
        parts.push(format!(
            "\n[Candidate {}]\nContent: {}\nConfidence: {:.2}\nSources: {}\nTags: {}\n",
            i,
            c.content,
            c.confidence,
            c.source_memory_ids.join(", "),
            c.tags.join(", "),
        ));
    }

    let mut char_count: usize = parts.iter().map(|p| p.len()).sum();

    if !source_memories.is_empty() {
        parts.push(format!(
            "\n## Source Memories ({}):\n",
            source_memories.len()
        ));
        for m in source_memories {
            let entry = format!("\nID: {}\nContent: {}\n", hex::encode(m.id), &m.content,);
            if char_count + entry.len() > budget_chars {
                parts.push("\n[... truncated for token budget ...]\n".into());
                break;
            }
            char_count += entry.len();
            parts.push(entry);
        }
    }

    if !existing_insights.is_empty() {
        parts.push(format!(
            "\n## Existing Insights ({}):\n",
            existing_insights.len()
        ));
        for m in existing_insights {
            let entry = format!("\nID: {}\nContent: {}\n", hex::encode(m.id), &m.content,);
            if char_count + entry.len() > budget_chars {
                parts.push("\n[... truncated for token budget ...]\n".into());
                break;
            }
            char_count += entry.len();
            parts.push(entry);
        }
    }

    (system, parts.join(""))
}

/// Estimate token count from character count.
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / CHARS_PER_TOKEN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(idx: u8, content: &str) -> MemoryEntry {
        let mut id = [0u8; 16];
        id[0] = idx;
        MemoryEntry {
            id,
            content: content.into(),
            importance: 0.5 + idx as f32 * 0.1,
            entity_id: Some("test".into()),
            embedding: vec![0.0; 16],
            created_at: 1_000_000 * idx as u64,
        }
    }

    #[test]
    fn proposal_prompt_includes_all_memories() {
        let entries: Vec<MemoryEntry> = (0..5)
            .map(|i| make_entry(i, &format!("memory {i}")))
            .collect();
        let refs: Vec<&MemoryEntry> = entries.iter().collect();
        let centroid = vec![0.0f32; 16];
        let (sys, user) = build_proposal_prompt(&refs, &centroid, 4000);
        assert!(sys.contains("JSON"));
        for i in 0..5 {
            assert!(user.contains(&format!("memory {i}")));
        }
    }

    #[test]
    fn proposal_prompt_respects_token_budget() {
        let long_content = "x".repeat(5000);
        let entries: Vec<MemoryEntry> = (0..10).map(|i| make_entry(i, &long_content)).collect();
        let refs: Vec<&MemoryEntry> = entries.iter().collect();
        let centroid = vec![0.0f32; 16];
        let (_sys, user) = build_proposal_prompt(&refs, &centroid, 500);
        assert!(user.len() < 500 * 4 + 200);
    }

    #[test]
    fn validation_prompt_includes_candidates_and_sources() {
        let candidates = vec![CandidateInsight {
            content: "test insight".into(),
            confidence: 0.8,
            source_memory_ids: vec!["aa".into()],
            tags: vec!["tag1".into()],
        }];
        let source = make_entry(0, "source memory");
        let existing = make_entry(1, "existing insight");
        let (sys, user) = build_validation_prompt(&candidates, &[&source], &[&existing], 6000);
        assert!(sys.contains("evaluator"));
        assert!(user.contains("test insight"));
        assert!(user.contains("source memory"));
        assert!(user.contains("existing insight"));
    }

    #[test]
    fn estimate_tokens_reasonable() {
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens(""), 0);
    }
}
