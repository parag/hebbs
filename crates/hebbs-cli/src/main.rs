use std::io::IsTerminal;

use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use hebbs_cli::cli::{Cli, ColorArg, FormatArg};
use hebbs_cli::commands;
use hebbs_cli::config::{CliConfig, ColorMode, OutputFormat};
use hebbs_cli::connection::ConnectionManager;
use hebbs_cli::format::Renderer;
use hebbs_cli::repl;

fn main() {
    let cli = Cli::parse();

    let mut config = CliConfig::load();

    if let Some(ref ep) = cli.endpoint {
        let endpoint = if ep.starts_with("http://") || ep.starts_with("https://") {
            ep.clone()
        } else {
            format!("http://{}", ep)
        };
        config.endpoint = endpoint;
    }
    if let Some(hp) = cli.http_port {
        config.http_port = hp;
    }
    if let Some(tm) = cli.timeout {
        config.timeout_ms = tm;
    }
    if let Some(ref fmt) = cli.format {
        config.output_format = match fmt {
            FormatArg::Human => OutputFormat::Human,
            FormatArg::Json => OutputFormat::Json,
            FormatArg::Raw => OutputFormat::Raw,
        };
    }
    if let Some(ref c) = cli.color {
        config.color = match c {
            ColorArg::Always => ColorMode::Always,
            ColorArg::Never => ColorMode::Never,
            ColorArg::Auto => ColorMode::Auto,
        };
    }

    init_tracing(cli.verbose);

    let is_tty = std::io::stdout().is_terminal();
    let use_color = config.should_color(is_tty);
    let renderer = Renderer::new(config.output_format, use_color);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("failed to create tokio runtime");

    rt.block_on(async {
        let mut conn = ConnectionManager::new(config.endpoint.clone(), config.timeout_ms)
            .with_api_key(cli.api_key.clone());

        match cli.command {
            Some(cmd) => {
                let exit_code =
                    commands::execute(cmd, &mut conn, &renderer, config.http_port).await;
                std::process::exit(exit_code);
            }
            None => {
                repl::run_repl(
                    &mut conn,
                    &renderer,
                    &config.history_file,
                    config.max_history,
                    config.http_port,
                )
                .await;
            }
        }
    });
}

fn init_tracing(verbosity: u8) {
    let level = match verbosity {
        0 => return,
        1 => "debug",
        _ => "trace",
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}
