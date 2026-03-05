use clap::Parser;
use hebbs_cli::cli::{Cli, Commands};

fn parse_args(args: &[&str]) -> Result<Cli, clap::Error> {
    let mut full_args = vec!["hebbs-cli"];
    full_args.extend_from_slice(args);
    Cli::try_parse_from(full_args)
}

// ═══════════════════════════════════════════════════════════════════════
//  Remember Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn remember_with_content() {
    let cli = parse_args(&["remember", "Hello world"]).unwrap();
    match cli.command {
        Some(Commands::Remember { content, .. }) => {
            assert_eq!(content.unwrap(), "Hello world");
        }
        _ => panic!("Expected Remember command"),
    }
}

#[test]
fn remember_with_all_flags() {
    let cli = parse_args(&[
        "remember",
        "Hello world",
        "--importance",
        "0.8",
        "--context",
        r#"{"key":"val"}"#,
        "--entity-id",
        "user-123",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Remember {
            content,
            importance,
            context,
            entity_id,
            edge,
        }) => {
            assert_eq!(content.unwrap(), "Hello world");
            assert_eq!(importance.unwrap(), 0.8);
            assert_eq!(context.unwrap(), r#"{"key":"val"}"#);
            assert_eq!(entity_id.unwrap(), "user-123");
            assert!(edge.is_empty());
        }
        _ => panic!("Expected Remember command"),
    }
}

#[test]
fn remember_with_single_edge() {
    let cli = parse_args(&[
        "remember",
        "Hello world",
        "--edge",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV:caused_by:0.9",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Remember { edge, .. }) => {
            assert_eq!(edge.len(), 1);
            assert_eq!(edge[0], "01ARZ3NDEKTSV4RRFFQ69G5FAV:caused_by:0.9");
        }
        _ => panic!("Expected Remember command"),
    }
}

#[test]
fn remember_with_multiple_edges() {
    let cli = parse_args(&[
        "remember",
        "Hello world",
        "--edge",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV:caused_by:0.9",
        "--edge",
        "01ARZ3NDEKTSV4RRFFQ69G5FAW:related_to",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Remember { edge, .. }) => {
            assert_eq!(edge.len(), 2);
        }
        _ => panic!("Expected Remember command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Get Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn get_with_id() {
    let cli = parse_args(&["get", "01ARZ3NDEKTSV4RRFFQ69G5FAV"]).unwrap();
    match cli.command {
        Some(Commands::Get { id }) => {
            assert_eq!(id, "01ARZ3NDEKTSV4RRFFQ69G5FAV");
        }
        _ => panic!("Expected Get command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Recall Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn recall_with_cue() {
    let cli = parse_args(&["recall", "what happened yesterday"]).unwrap();
    match cli.command {
        Some(Commands::Recall { cue, top_k, .. }) => {
            assert_eq!(cue.unwrap(), "what happened yesterday");
            assert_eq!(top_k, 10); // default
        }
        _ => panic!("Expected Recall command"),
    }
}

#[test]
fn recall_with_strategy() {
    let cli = parse_args(&["recall", "test", "--strategy", "temporal", "--top-k", "5"]).unwrap();
    match cli.command {
        Some(Commands::Recall {
            strategy, top_k, ..
        }) => {
            assert!(strategy.is_some());
            assert_eq!(top_k, 5);
        }
        _ => panic!("Expected Recall command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Revise Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn revise_with_content() {
    let cli = parse_args(&[
        "revise",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "--content",
        "updated text",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Revise { id, content, .. }) => {
            assert_eq!(id, "01ARZ3NDEKTSV4RRFFQ69G5FAV");
            assert_eq!(content.unwrap(), "updated text");
        }
        _ => panic!("Expected Revise command"),
    }
}

#[test]
fn revise_with_edges() {
    let cli = parse_args(&[
        "revise",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "--content",
        "updated",
        "--edge",
        "01ARZ3NDEKTSV4RRFFQ69G5FAW:followed_by:0.7",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Revise { id, edge, .. }) => {
            assert_eq!(id, "01ARZ3NDEKTSV4RRFFQ69G5FAV");
            assert_eq!(edge.len(), 1);
        }
        _ => panic!("Expected Revise command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Forget Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn forget_by_entity() {
    let cli = parse_args(&["forget", "--entity-id", "user-123"]).unwrap();
    match cli.command {
        Some(Commands::Forget { entity_id, .. }) => {
            assert_eq!(entity_id.unwrap(), "user-123");
        }
        _ => panic!("Expected Forget command"),
    }
}

#[test]
fn forget_by_ids() {
    let cli = parse_args(&[
        "forget",
        "--ids",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "--ids",
        "01ARZ3NDEKTSV4RRFFQ69G5FAW",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Forget { ids, .. }) => {
            assert_eq!(ids.len(), 2);
        }
        _ => panic!("Expected Forget command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Prime Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn prime_with_entity() {
    let cli = parse_args(&["prime", "user-123"]).unwrap();
    match cli.command {
        Some(Commands::Prime { entity_id, .. }) => {
            assert_eq!(entity_id, "user-123");
        }
        _ => panic!("Expected Prime command"),
    }
}

#[test]
fn prime_with_all_flags() {
    let cli = parse_args(&[
        "prime",
        "user-123",
        "--context",
        r#"{"topic":"sales"}"#,
        "--max-memories",
        "50",
        "--recency-us",
        "1000000",
        "--similarity-cue",
        "recent deals",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Prime {
            entity_id,
            context,
            max_memories,
            recency_us,
            similarity_cue,
        }) => {
            assert_eq!(entity_id, "user-123");
            assert!(context.is_some());
            assert_eq!(max_memories.unwrap(), 50);
            assert_eq!(recency_us.unwrap(), 1000000);
            assert_eq!(similarity_cue.unwrap(), "recent deals");
        }
        _ => panic!("Expected Prime command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Subscribe Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn subscribe_defaults() {
    let cli = parse_args(&["subscribe"]).unwrap();
    match cli.command {
        Some(Commands::Subscribe {
            entity_id,
            confidence,
        }) => {
            assert!(entity_id.is_none());
            assert_eq!(confidence, 0.5);
        }
        _ => panic!("Expected Subscribe command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Feed Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn feed_parsing() {
    let cli = parse_args(&["feed", "42", "some text to feed"]).unwrap();
    match cli.command {
        Some(Commands::Feed {
            subscription_id,
            text,
        }) => {
            assert_eq!(subscription_id, 42);
            assert_eq!(text, "some text to feed");
        }
        _ => panic!("Expected Feed command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Reflect Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn reflect_global() {
    let cli = parse_args(&["reflect"]).unwrap();
    match cli.command {
        Some(Commands::Reflect {
            entity_id,
            since_us,
        }) => {
            assert!(entity_id.is_none());
            assert!(since_us.is_none());
        }
        _ => panic!("Expected Reflect command"),
    }
}

#[test]
fn reflect_entity() {
    let cli = parse_args(&[
        "reflect",
        "--entity-id",
        "user-123",
        "--since-us",
        "1000000",
    ])
    .unwrap();
    match cli.command {
        Some(Commands::Reflect {
            entity_id,
            since_us,
        }) => {
            assert_eq!(entity_id.unwrap(), "user-123");
            assert_eq!(since_us.unwrap(), 1000000);
        }
        _ => panic!("Expected Reflect command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Insights Command Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn insights_all_flags() {
    let cli = parse_args(&[
        "insights",
        "--entity-id",
        "user-123",
        "--min-confidence",
        "0.7",
        "--max-results",
        "20",
    ])
    .unwrap();

    match cli.command {
        Some(Commands::Insights {
            entity_id,
            min_confidence,
            max_results,
        }) => {
            assert_eq!(entity_id.unwrap(), "user-123");
            assert_eq!(min_confidence.unwrap(), 0.7);
            assert_eq!(max_results.unwrap(), 20);
        }
        _ => panic!("Expected Insights command"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Diagnostic Commands Parsing
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn status_command() {
    let cli = parse_args(&["status"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Status)));
}

#[test]
fn inspect_command() {
    let cli = parse_args(&["inspect", "01ARZ3NDEKTSV4RRFFQ69G5FAV"]).unwrap();
    match cli.command {
        Some(Commands::Inspect { id }) => {
            assert_eq!(id, "01ARZ3NDEKTSV4RRFFQ69G5FAV");
        }
        _ => panic!("Expected Inspect command"),
    }
}

#[test]
fn export_defaults() {
    let cli = parse_args(&["export"]).unwrap();
    match cli.command {
        Some(Commands::Export { entity_id, limit }) => {
            assert!(entity_id.is_none());
            assert_eq!(limit, 1000);
        }
        _ => panic!("Expected Export command"),
    }
}

#[test]
fn export_with_flags() {
    let cli = parse_args(&["export", "--entity-id", "user-1", "--limit", "500"]).unwrap();
    match cli.command {
        Some(Commands::Export { entity_id, limit }) => {
            assert_eq!(entity_id.unwrap(), "user-1");
            assert_eq!(limit, 500);
        }
        _ => panic!("Expected Export command"),
    }
}

#[test]
fn metrics_command() {
    let cli = parse_args(&["metrics"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Metrics)));
}

#[test]
fn version_command() {
    let cli = parse_args(&["version"]).unwrap();
    assert!(matches!(cli.command, Some(Commands::Version)));
}

// ═══════════════════════════════════════════════════════════════════════
//  Global Flags
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn global_endpoint_flag() {
    let cli = parse_args(&["--endpoint", "http://remote:9000", "status"]).unwrap();
    assert_eq!(cli.endpoint.unwrap(), "http://remote:9000");
}

#[test]
fn global_format_flag() {
    let cli = parse_args(&["--format", "json", "status"]).unwrap();
    assert!(cli.format.is_some());
}

#[test]
fn global_verbose_flag() {
    let cli = parse_args(&["-v", "status"]).unwrap();
    assert_eq!(cli.verbose, 1);
}

#[test]
fn global_double_verbose_flag() {
    let cli = parse_args(&["-vv", "status"]).unwrap();
    assert_eq!(cli.verbose, 2);
}

#[test]
fn no_subcommand_enters_repl() {
    let cli = parse_args(&[]).unwrap();
    assert!(cli.command.is_none());
}

// ═══════════════════════════════════════════════════════════════════════
//  Error Cases
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn unknown_subcommand_fails() {
    assert!(parse_args(&["nonexistent"]).is_err());
}

#[test]
fn get_without_id_fails() {
    assert!(parse_args(&["get"]).is_err());
}

#[test]
fn feed_without_args_fails() {
    assert!(parse_args(&["feed"]).is_err());
}
