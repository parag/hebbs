use proptest::prelude::*;

use hebbs_cli::format::{
    format_elapsed, format_kind, format_relative_time, parse_context_json, parse_memory_id,
    proto_memory_to_json, ulid_to_string,
};
use hebbs_cli::tokenizer;

// ═══════════════════════════════════════════════════════════════════════
//  ULID Round-Trip Property Tests
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn ulid_roundtrip_arbitrary_16_bytes(bytes in proptest::collection::vec(any::<u8>(), 16..=16)) {
        let s = ulid_to_string(&bytes);
        prop_assert_eq!(s.len(), 26, "ULID string should be exactly 26 characters");

        let parsed = parse_memory_id(&s).expect("Should parse valid ULID");
        prop_assert_eq!(parsed, bytes, "ULID round-trip should be identity");
    }

    #[test]
    fn hex_roundtrip_arbitrary_16_bytes(bytes in proptest::collection::vec(any::<u8>(), 16..=16)) {
        let hex_str = hex::encode(&bytes);
        let parsed = parse_memory_id(&hex_str).expect("Should parse valid hex");
        prop_assert_eq!(parsed, bytes, "Hex round-trip should be identity");
    }

    #[test]
    fn parse_memory_id_rejects_wrong_length(
        s in "[0-9a-fA-F]{1,31}|[0-9a-fA-F]{33,64}"
    ) {
        // Strings that are not 26 (ULID) or 32 (hex) chars should fail
        if s.len() != 26 && s.len() != 32 {
            prop_assert!(parse_memory_id(&s).is_err());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tokenizer Property Tests
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn tokenize_simple_words_no_crash(input in "[a-zA-Z0-9_ ]{0,200}") {
        let result = tokenizer::tokenize(&input);
        prop_assert!(result.is_ok());
        let tokens = result.unwrap();
        // Token count should be at most word count
        let non_empty_words: Vec<&str> = input.split_whitespace().collect();
        prop_assert_eq!(tokens.len(), non_empty_words.len());
    }

    #[test]
    fn tokenize_preserves_double_quoted_content(content in "[a-zA-Z0-9 ]{1,100}") {
        let input = format!(r#"cmd "{}""#, content);
        let result = tokenizer::tokenize(&input);
        prop_assert!(result.is_ok());
        let tokens = result.unwrap();
        prop_assert!(tokens.len() >= 2);
        prop_assert_eq!(&tokens[1], &content);
    }

    #[test]
    fn tokenize_preserves_single_quoted_content(content in "[a-zA-Z0-9 ]{1,100}") {
        let input = format!("cmd '{}'", content);
        let result = tokenizer::tokenize(&input);
        prop_assert!(result.is_ok());
        let tokens = result.unwrap();
        prop_assert!(tokens.len() >= 2);
        prop_assert_eq!(&tokens[1], &content);
    }

    #[test]
    fn tokenize_balanced_quotes_always_ok(
        words in proptest::collection::vec("[a-zA-Z0-9]{1,20}", 1..10)
    ) {
        let input = words.join(" ");
        let result = tokenizer::tokenize(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn tokenize_unbalanced_double_quote_errors(word in "[a-zA-Z]{1,20}") {
        let input = format!(r#"cmd "{}"#, word);
        let result = tokenizer::tokenize(&input);
        prop_assert!(result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Format Property Tests
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn format_elapsed_never_panics(micros in 0u64..1_000_000_000) {
        let d = std::time::Duration::from_micros(micros);
        let s = format_elapsed(d);
        prop_assert!(!s.is_empty());
    }

    #[test]
    fn format_relative_time_never_panics(us in 0u64..u64::MAX) {
        let s = format_relative_time(us);
        prop_assert!(!s.is_empty());
    }

    #[test]
    fn format_kind_never_panics(kind in -10i32..20) {
        let s = format_kind(kind);
        prop_assert!(!s.is_empty());
    }

    #[test]
    fn parse_context_json_valid_objects(
        key in "[a-zA-Z]{1,20}",
        value in "[a-zA-Z0-9 ]{1,50}"
    ) {
        let json = format!(r#"{{"{}": "{}"}}"#, key, value);
        let result = parse_context_json(&json);
        prop_assert!(result.is_ok());
        let s = result.unwrap();
        prop_assert_eq!(s.fields.len(), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Proto Memory JSON Conversion Property Tests
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn proto_memory_to_json_roundtrip_serialization(
        content in "[a-zA-Z0-9 ]{1,100}",
        importance in 0.0f32..1.0,
        kind in 1i32..4,
    ) {
        use hebbs_proto::generated as pb;

        let ulid = ulid::Ulid::new();
        let memory = pb::Memory {
            memory_id: ulid.0.to_be_bytes().to_vec(),
            content: content.clone(),
            importance,
            context: None,
            entity_id: None,
            embedding: Vec::new(),
            created_at: 0,
            updated_at: 0,
            last_accessed_at: 0,
            access_count: 0,
            decay_score: 0.0,
            kind,
            device_id: None,
            logical_clock: 0,
        };

        let json = proto_memory_to_json(&memory);

        // Verify serialization succeeds
        let serialized = serde_json::to_string(&json);
        prop_assert!(serialized.is_ok());

        // Verify key fields
        prop_assert_eq!(&json.content, &content);
        prop_assert_eq!(json.importance, importance);
        prop_assert_eq!(json.memory_id.len(), 26);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Renderer Property Tests
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn renderer_memory_never_panics(
        content in "[a-zA-Z0-9 ]{0,200}",
        importance in 0.0f32..1.0,
        kind in 0i32..5,
    ) {
        use hebbs_cli::config::OutputFormat;
        use hebbs_cli::format::Renderer;
        use hebbs_proto::generated as pb;

        let memory = pb::Memory {
            memory_id: vec![0u8; 16],
            content,
            importance,
            context: None,
            entity_id: None,
            embedding: Vec::new(),
            created_at: 0,
            updated_at: 0,
            last_accessed_at: 0,
            access_count: 0,
            decay_score: 0.0,
            kind,
            device_id: None,
            logical_clock: 0,
        };

        for fmt in [OutputFormat::Human, OutputFormat::Json, OutputFormat::Raw] {
            for color in [true, false] {
                let renderer = Renderer::new(fmt, color);
                let mut buf = Vec::new();
                let result = renderer.render_memory(&memory, &mut buf);
                prop_assert!(result.is_ok());
                prop_assert!(!buf.is_empty());
            }
        }
    }

    #[test]
    fn renderer_recall_results_never_panics(
        count in 0usize..20,
        score in 0.0f32..1.0,
    ) {
        use hebbs_cli::config::OutputFormat;
        use hebbs_cli::format::Renderer;
        use hebbs_proto::generated as pb;

        let results: Vec<pb::RecallResult> = (0..count).map(|_| {
            pb::RecallResult {
                memory: Some(pb::Memory {
                    memory_id: vec![0u8; 16],
                    content: "test".to_string(),
                    importance: 0.5,
                    context: None,
                    entity_id: None,
                    embedding: Vec::new(),
                    created_at: 0,
                    updated_at: 0,
                    last_accessed_at: 0,
                    access_count: 0,
                    decay_score: 0.0,
                    kind: 1,
                    device_id: None,
                    logical_clock: 0,
                }),
                score,
                strategy_details: Vec::new(),
            }
        }).collect();

        for fmt in [OutputFormat::Human, OutputFormat::Json, OutputFormat::Raw] {
            let renderer = Renderer::new(fmt, false);
            let mut buf = Vec::new();
            let result = renderer.render_recall_results(&results, &mut buf);
            prop_assert!(result.is_ok());
        }
    }
}
