use std::io::Write;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Editor};

use crate::cli;
use crate::commands;
use crate::connection::ConnectionManager;
use crate::format::Renderer;
use crate::tokenizer;

// ═══════════════════════════════════════════════════════════════════════
//  Tab Completion
// ═══════════════════════════════════════════════════════════════════════

static SUBCOMMANDS: &[&str] = &[
    "remember",
    "get",
    "recall",
    "revise",
    "forget",
    "prime",
    "subscribe",
    "feed",
    "reflect",
    "insights",
    "status",
    "inspect",
    "export",
    "metrics",
    "version",
];

static DOT_COMMANDS: &[&str] = &[
    ".help",
    ".quit",
    ".exit",
    ".connect",
    ".disconnect",
    ".status",
    ".clear",
    ".history",
];

static FLAGS: &[&str] = &[
    "--importance",
    "--context",
    "--entity-id",
    "--edge",
    "--strategy",
    "--top-k",
    "--max-depth",
    "--seed",
    "--content",
    "--context-mode",
    "--ids",
    "--staleness-us",
    "--access-floor",
    "--kind",
    "--decay-floor",
    "--max-memories",
    "--recency-us",
    "--similarity-cue",
    "--confidence",
    "--min-confidence",
    "--max-results",
    "--since-us",
    "--limit",
    "--format",
];

static STRATEGY_VALUES: &[&str] = &["similarity", "temporal", "causal", "analogical"];
static KIND_VALUES: &[&str] = &["episode", "insight", "revision"];

struct HebbsHelper;

impl Completer for HebbsHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let prefix = &line[..pos];

        let word_start = prefix
            .rfind(|c: char| c.is_whitespace())
            .map_or(0, |i| i + 1);
        let word = &prefix[word_start..];

        let candidates: Vec<Pair> = if word.starts_with('.') {
            DOT_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect()
        } else if word.starts_with("--") {
            FLAGS
                .iter()
                .filter(|f| f.starts_with(word))
                .map(|f| Pair {
                    display: f.to_string(),
                    replacement: f.to_string(),
                })
                .collect()
        } else if word_start == 0 {
            SUBCOMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect()
        } else {
            let prev_tokens = tokenizer::tokenize(prefix).unwrap_or_default();
            let prev_flag = prev_tokens.iter().rev().nth(1).map(|s| s.as_str());

            match prev_flag {
                Some("--strategy") | Some("-s") => STRATEGY_VALUES
                    .iter()
                    .filter(|v| v.starts_with(word))
                    .map(|v| Pair {
                        display: v.to_string(),
                        replacement: v.to_string(),
                    })
                    .collect(),
                Some("--kind") => KIND_VALUES
                    .iter()
                    .filter(|v| v.starts_with(word))
                    .map(|v| Pair {
                        display: v.to_string(),
                        replacement: v.to_string(),
                    })
                    .collect(),
                _ => Vec::new(),
            }
        };

        Ok((word_start, candidates))
    }
}

impl Hinter for HebbsHelper {
    type Hint = String;
}

impl Highlighter for HebbsHelper {}

impl Validator for HebbsHelper {}

impl rustyline::Helper for HebbsHelper {}

// ═══════════════════════════════════════════════════════════════════════
//  REPL Loop
// ═══════════════════════════════════════════════════════════════════════

pub async fn run_repl(
    conn: &mut ConnectionManager,
    renderer: &Renderer,
    history_file: &std::path::Path,
    max_history: usize,
    http_port: u16,
) {
    let config = Config::builder()
        .max_history_size(max_history)
        .unwrap_or_else(|_| Config::builder())
        .auto_add_history(true)
        .build();

    let mut rl: Editor<HebbsHelper, rustyline::history::DefaultHistory> =
        match Editor::with_config(config) {
            Ok(e) => e,
            Err(_) => match Editor::new() {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Failed to initialize REPL: {}", e);
                    return;
                }
            },
        };

    rl.set_helper(Some(HebbsHelper));

    if let Some(parent) = history_file.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    rl.load_history(history_file).ok();

    println!(
        "HEBBS CLI v{} -- Type .help for commands, .quit to exit",
        env!("CARGO_PKG_VERSION")
    );

    loop {
        let prompt = if conn.is_connected() {
            let host = conn
                .endpoint()
                .trim_start_matches("http://")
                .trim_start_matches("https://");
            format!("hebbs {}> ", host)
        } else {
            "hebbs (disconnected)> ".to_string()
        };

        let line = rl.readline(&prompt);

        match line {
            Ok(input) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.starts_with('.') {
                    if handle_dot_command(trimmed, conn, renderer, http_port).await {
                        break;
                    }
                    continue;
                }

                let tokens = match tokenizer::tokenize(trimmed) {
                    Ok(t) if t.is_empty() => continue,
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("Parse error: {}", e);
                        continue;
                    }
                };

                let mut args = vec!["hebbs-cli".to_string()];
                args.extend(tokens);

                let cmd_def = cli::build_command();
                let matches = cmd_def.try_get_matches_from(args);

                match matches {
                    Ok(m) => {
                        let reparsed = match parse_from_repl_matches(m) {
                            Some(cmd) => cmd,
                            None => {
                                eprintln!("Unknown command. Type .help for available commands.");
                                continue;
                            }
                        };

                        let _exit_code =
                            commands::execute(reparsed, conn, renderer, http_port).await;
                    }
                    Err(e) => {
                        let rendered = e.render();
                        eprint!("{}", rendered);
                    }
                }
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                println!("Goodbye.");
                break;
            }
            Err(e) => {
                eprintln!("Readline error: {}", e);
                break;
            }
        }
    }

    rl.save_history(history_file).ok();
}

fn parse_from_repl_matches(matches: clap::ArgMatches) -> Option<cli::Commands> {
    use clap::FromArgMatches;

    let cli_result = cli::Cli::from_arg_matches(&matches);
    match cli_result {
        Ok(parsed) => parsed.command,
        Err(_) => None,
    }
}

async fn handle_dot_command(
    input: &str,
    conn: &mut ConnectionManager,
    renderer: &Renderer,
    http_port: u16,
) -> bool {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let cmd = parts.first().copied().unwrap_or("");

    match cmd {
        ".quit" | ".exit" => {
            println!("Goodbye.");
            return true;
        }
        ".help" => {
            print_help();
        }
        ".connect" => {
            if let Some(ep) = parts.get(1) {
                let endpoint = if ep.starts_with("http://") || ep.starts_with("https://") {
                    ep.to_string()
                } else {
                    format!("http://{}", ep)
                };
                conn.set_endpoint(endpoint.clone());
                println!(
                    "Endpoint set to {}. Will connect on next command.",
                    endpoint
                );
            } else {
                println!("Usage: .connect <endpoint>");
                println!("Current endpoint: {}", conn.endpoint());
            }
        }
        ".disconnect" => {
            conn.disconnect();
            println!("Disconnected.");
        }
        ".status" => {
            let _exit_code =
                commands::execute(cli::Commands::Status, conn, renderer, http_port).await;
        }
        ".clear" => {
            print!("\x1B[2J\x1B[H");
            std::io::stdout().flush().ok();
        }
        _ => {
            eprintln!(
                "Unknown dot-command: {}. Type .help for available commands.",
                cmd
            );
        }
    }

    false
}

fn print_help() {
    println!("HEBBS CLI Commands:");
    println!();
    println!("  Memory Operations:");
    println!(
        "    remember <content> [--importance N] [--context JSON] [--entity-id ID] [--edge SPEC]"
    );
    println!("    get <memory-id>");
    println!("    recall <cue> [--strategy S] [--top-k N] [--entity-id ID]");
    println!("    revise <memory-id> [--content TEXT] [--importance N] [--edge SPEC]");
    println!("    forget --ids ID [--entity-id ID] [--staleness-us N]");
    println!("    prime <entity-id> [--context JSON] [--max-memories N]");
    println!();
    println!("  Edge Format: TARGET_ID:EDGE_TYPE[:CONFIDENCE]");
    println!("    Types: caused_by, related_to, followed_by, revised_from, insight_from");
    println!();
    println!("  Streaming:");
    println!("    subscribe [--entity-id ID] [--confidence N]");
    println!("    feed <subscription-id> <text>");
    println!();
    println!("  Reflection:");
    println!("    reflect [--entity-id ID] [--since-us N]");
    println!("    insights [--entity-id ID] [--min-confidence N]");
    println!();
    println!("  Diagnostics:");
    println!("    status           Server health and info");
    println!("    inspect <id>     Memory detail + graph + neighbors");
    println!("    export [--entity-id ID] [--limit N]");
    println!("    metrics          Server Prometheus metrics");
    println!("    version          CLI version");
    println!();
    println!("  Session (dot-commands):");
    println!("    .help            Show this help");
    println!("    .quit / .exit    Exit the REPL");
    println!("    .connect <ep>    Connect to a different server");
    println!("    .disconnect      Disconnect from server");
    println!("    .status          Quick server status");
    println!("    .clear           Clear the terminal");
    println!();
    println!("  Global Flags:");
    println!("    --format human|json|raw    Output format");
    println!("    -v / -vv                   Verbose/trace mode");
}
