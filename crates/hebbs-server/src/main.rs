use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, EnvFilter};

use hebbs_server::config::HebbsConfig;
use hebbs_server::server;

#[derive(Parser)]
#[command(
    name = "hebbs-server",
    version,
    about = "HEBBS cognitive memory server"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HEBBS server (default)
    Start {
        /// Path to configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// gRPC listen port
        #[arg(long)]
        grpc_port: Option<u16>,

        /// HTTP listen port
        #[arg(long)]
        http_port: Option<u16>,

        /// Bind address
        #[arg(long)]
        bind_address: Option<String>,

        /// Data directory
        #[arg(long)]
        data_dir: Option<String>,
    },
    /// Print version information
    Version,
    /// Validate configuration file
    ConfigCheck {
        /// Path to configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Print resolved configuration
    ConfigDump {
        /// Path to configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Start {
        config: None,
        grpc_port: None,
        http_port: None,
        bind_address: None,
        data_dir: None,
    }) {
        Commands::Start {
            config,
            grpc_port,
            http_port,
            bind_address,
            data_dir,
        } => {
            let mut cfg = match HebbsConfig::load(config.as_deref()) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("configuration error: {}", e);
                    std::process::exit(1);
                }
            };

            if let Some(p) = grpc_port {
                cfg.server.grpc_port = p;
            }
            if let Some(p) = http_port {
                cfg.server.http_port = p;
            }
            if let Some(a) = bind_address {
                cfg.server.bind_address = a;
            }
            if let Some(d) = data_dir {
                cfg.storage.data_dir = d;
            }

            if let Err(e) = cfg.validate() {
                eprintln!("configuration validation failed: {}", e);
                std::process::exit(1);
            }

            init_tracing(&cfg);

            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .max_blocking_threads(cfg.server.max_blocking_threads)
                .build()
                .expect("failed to create tokio runtime");

            rt.block_on(async {
                if let Err(e) = server::run(cfg).await {
                    tracing::error!(error = %e, "server failed");
                    std::process::exit(1);
                }
            });
        }
        Commands::Version => {
            println!(
                "hebbs-server {} ({})",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::ARCH
            );
        }
        Commands::ConfigCheck { config } => match HebbsConfig::load(config.as_deref()) {
            Ok(cfg) => match cfg.validate() {
                Ok(()) => {
                    println!("configuration is valid");
                }
                Err(e) => {
                    eprintln!("configuration validation failed: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("configuration error: {}", e);
                std::process::exit(1);
            }
        },
        Commands::ConfigDump { config } => match HebbsConfig::load(config.as_deref()) {
            Ok(cfg) => {
                let output = toml::to_string_pretty(&cfg).unwrap_or_default();
                println!("{}", output);
            }
            Err(e) => {
                eprintln!("configuration error: {}", e);
                std::process::exit(1);
            }
        },
    }
}

fn init_tracing(config: &HebbsConfig) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.logging.level));

    match config.logging.format.as_str() {
        "json" => {
            fmt().with_env_filter(filter).json().init();
        }
        _ => {
            fmt().with_env_filter(filter).init();
        }
    }
}
