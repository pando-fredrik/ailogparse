use tokio::sync::Mutex;
use anyhow::{Result};
use ollama_rs::generation::chat::{request::ChatMessageRequest, ChatMessage, ChatMessageResponse};
use std::sync::Arc;
use tokio_stream::{StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use chrono::Local;

const MODEL_NAME: &str = "gemma2:9b";
pub struct Ollama {
    conversation: Arc<Mutex<Vec<ChatMessage>>>,
    ollama_instance: ollama_rs::Ollama,
}

impl Ollama {
    pub fn new() -> Ollama {
        Self {
            conversation: Arc::new(Mutex::new(vec![])),
            ollama_instance: ollama_rs::Ollama::default(),
        }
    }

    async fn has_model(&self, model_name: &str) -> Result<bool> {
        let local_models = self.ollama_instance.list_local_models().await?;

        Ok(local_models
            .iter()
            .find(|&model| model.name == model_name)
            .is_some())
    }

    pub async fn initialize_model(&mut self) -> Result<()> {
        if !self.has_model(MODEL_NAME).await? {
            println!("Installing model.... please wait");
            let mut status_stream = self
                .ollama_instance
                .pull_model_stream(MODEL_NAME.into(), false)
                .await?;

            while let Some(Ok(res)) = status_stream.next().await {
                print!("{} ", res.message);
                if let (Some(total), Some(completed)) = (res.total, res.completed) {
                    print!("{completed}/{total}");
                }
                println!();
            }
        }
        let mut conv = self.conversation.lock().await;

        let current_time = Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let system_prompt = format!(r#"You are an expert log parser and your purpose is to find report any errors or potential problems in the log.
                                              IMPORTANT ALWAYS REFERENCE YOUR FINDINGS.

                                              Current time is {current_time}.

                                              The log is as follows:"#);
        conv.push(ChatMessage::system(system_prompt));

        Ok(())
    }
    pub async fn send(&mut self, message: &str) -> Result<ReceiverStream<Result<ChatMessageResponse>>> {
        let async_conversation = self.conversation.clone();
        let mut locked_conversation = self.conversation.lock().await;

        let (tx, rx) = mpsc::channel(100);
        locked_conversation.push(ChatMessage::user(message.to_string()));

        let history = locked_conversation.clone();

        let mut response_stream = self.ollama_instance.send_chat_messages_stream(ChatMessageRequest::new(
            MODEL_NAME.into(),
            history,
        )).await?;

        tokio::spawn(async move {
            let mut complete_response = String::new();
            while let Some(Ok(res)) = response_stream.next().await {
                tx.send(Ok(res.clone())).await.ok();
                if let Some(assistant_message) = res.message {
                    complete_response += assistant_message.content.as_str();
                }
            }
            let mut conv = async_conversation.lock().await;
            conv.push(ChatMessage::assistant(complete_response));
        });

        Ok(ReceiverStream::new(rx))
    }
}
