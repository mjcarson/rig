use anyhow::Result;
use rig_agent::prelude::*;
use rig_agent::tool::ToolContext;
use rig_vertexai::anthropic::{AnthropicVertexClient, CLAUDE_SONNET_4_5};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Subtracter;

impl Tool for Subtracter {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!(schema_for!(OperationArgs))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Subtracting {} from {}", args.y, args.x);
        let result = args.x - args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Create a Claude-on-Vertex client from GOOGLE_CLOUD_PROJECT,
    // GOOGLE_CLOUD_LOCATION, and the default credentials chain
    let client = AnthropicVertexClient::from_env()?;

    // Create an agent with a calculator tool
    let calculator_agent = client
        .agent(CLAUDE_SONNET_4_5)
        .tool(Subtracter)
        .max_tokens(1024)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 42 - 15");
    let answer = calculator_agent.prompt("Calculate 42 - 15").await?;
    println!("Claude on Vertex AI Calculator Agent: {answer}");

    Ok(())
}
