---
summary: "Kimi first-class provider notes for the shared Anthropic-style messages runtime"
read_when:
  - You are adding or debugging Kimi support
  - You need the Kimi model helper or environment variable name
  - You want to understand how Kimi fits the unified abstraction
---

# Kimi Provider

Kimi is a **first-class provider identity** in `alchemy_llm` that runs on the crate's shared `AnthropicMessages` runtime.

That means:

- the public abstraction stays the same: `Model<TApi>`, `stream(...)`, and `complete(...)`
- callers can target `KnownProvider::Kimi` directly
- the implementation reuses the shared Anthropic-style request/stream path underneath

## Quick Start

```rust
use alchemy_llm::{kimi_k2_5, stream};
use alchemy_llm::types::{AssistantMessageEvent, Context, Message, UserContent, UserMessage};
use futures::StreamExt;

#[tokio::main]
async fn main() -> alchemy_llm::Result<()> {
    let model = kimi_k2_5();
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello from Kimi".to_string()),
            timestamp: 0,
        })],
        tools: None,
    };

    let mut stream = stream(&model, &context, None)?;

    while let Some(event) = stream.next().await {
        if let AssistantMessageEvent::TextDelta { delta, .. } = event {
            print!("{}", delta);
        }
    }

    Ok(())
}
```

Set `KIMI_API_KEY` before calling `stream(...)` or `complete(...)`.

## Constructor

Use the curated model helper:

```rust
use alchemy_llm::kimi_k2_5;

let model = kimi_k2_5();
```

The helper returns `Model<AnthropicMessages>` with:

- provider: `KnownProvider::Kimi`
- base URL: `https://api.kimi.com/coding`
- model id: `kimi-coding`
- default context window: `128_000`
- default max output tokens: `16_384`
- reasoning: `true`
- default input type: text

## Environment and Request Flow

Authentication is resolved through the normal provider environment lookup:

- `KnownProvider::Kimi`
  -> `KIMI_API_KEY`
  -> `Authorization: Bearer ...` on the shared Anthropic-style request path

At the top-level API, Kimi uses the same entry points as other providers:

- `stream(&model, &context, None)`
- `complete(&model, &context, None)`

## Live Validation Notes

The following behaviors were validated against the live Kimi Coding API during implementation:

- `POST /coding/v1/messages` succeeded
- streaming SSE used Anthropic-style event names such as `message_start`, `content_block_delta`, and `message_delta`
- tool calls returned `content[].type = "tool_use"` in the non-stream response
- streamed tool calls used `content_block_start` with `type = "tool_use"` and `input_json_delta` chunks

## Files to Reference

Implementation and integration points live in:

- `src/models/kimi.rs`
- `src/providers/kimi.rs`
- `src/providers/shared/anthropic_like.rs`
- `src/providers/env.rs`
- `src/stream/mod.rs`

## Related Docs

- [architecture.md](./architecture.md)
- [../README.md](../README.md)
