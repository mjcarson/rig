## Rig-VertexAI

This companion crate integrates Google Cloud Vertex AI as a model provider with Rig:

- **Gemini models** (the `google` publisher) through `rig_vertexai::Client`, unary completions.
- **Claude models** (the `anthropic` publisher) through `rig_vertexai::anthropic::AnthropicVertexClient`, speaking the native Anthropic Messages API over `rawPredict`/`streamRawPredict` with tools, prompt caching, and streaming.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-vertexai = "0.3.5"
rig-core = "0.36.0"
```

You can also run `cargo add rig-vertexai rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Setup

Make sure to have Google Cloud credentials configured. You can use Application Default Credentials (ADC) by running:

```shell
gcloud auth application-default login
```

Both clients read `GOOGLE_CLOUD_PROJECT` (required) and `GOOGLE_CLOUD_LOCATION`
(optional, defaults to `global`) from the environment, and honor
`GOOGLE_CLOUD_SERVICE_ACCOUNT` for service account impersonation when
credentials come from the ADC chain. Explicit credentials can be supplied
through each client's builder; the Claude client additionally accepts a parsed
service account key JSON directly:

```rust,ignore
let client = rig_vertexai::anthropic::AnthropicVertexClient::builder()
    .project("my-project")
    .location("us-east5")
    .service_account_key(serde_json::from_str(&key_json)?)
    .build()?;
```
