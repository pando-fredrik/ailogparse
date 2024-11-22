mod ollama;

use crate::ollama::Ollama;
use anyhow::Result;
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use tokio::io::{stdout, AsyncWriteExt};
use tokio_stream::StreamExt;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    log_file: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ollama = Ollama::new();

    if let Err(e) = ollama.initialize_model().await {
        eprintln!(
            "Failed to initialize model (is ollama installed and running?): {}",
            e
        );
        std::process::exit(1);
    }

    let args = Args::parse();
    let contents = fs::read_to_string(args.log_file)?;
    println!("Processing log, please wait...");

    let mut stdout = stdout();
    let mut stream = ollama.send(&contents).await?;

    stdout.flush().await?;
    while let Some(Ok(res)) = stream.next().await {
        if let Some(response) = res.message {
            stdout
                .write_all(response.content.as_bytes())
                .await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}
