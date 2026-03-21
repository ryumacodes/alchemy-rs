---
summary: "MiniMax first-class provider notes for the shared OpenAI-compatible runtime"
read_when:
  - You are adding or debugging MiniMax support
  - You need the MiniMax model helper or environment variable name
  - You want to understand how MiniMax reasoning and tool calling works
  - You want to understand how MiniMax fits the unified abstraction
---

# MiniMax Provider

MiniMax is a **first-class provider identity** in `alchemy_llm` with a dedicated runtime that builds on the crate's shared OpenAI-compatible foundations.

That means:

- the public abstraction stays the same: `Model<TApi>`, `stream(...)`, and `complete(...)`
- callers can target `KnownProvider::Minimax` (global) or `KnownProvider::MinimaxCn` (China region)
- the implementation reuses shared helpers for OpenAI-like request serialization and stream handling
- dedicated MiniMax-specific logic handles reasoning normalization and temperature constraints

## Quick Start

```rust
use alchemy_llm::{minimax_m2_5, stream};
use alchemy_llm::types::{AssistantMessageEvent, Context, Message, UserContent, UserMessage};
use futures::StreamExt;

#[tokio::main]
async fn main() -> alchemy_llm::Result<()> {
    let model = minimax_m2_5();
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello from MiniMax".to_string()),
            timestamp: 0,
        })),
        tools: None,
    };

    let mut stream = stream(&model, &context, None)?;

    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { delta, .. } => print!("{}", delta),
            AssistantMessageEvent::ThinkingDelta { delta, .. } => print!("[thinking: {}]", delta),
            _ => {}
        }
    }

    Ok(())
}
```

Set `MINIMAX_API_KEY` before calling `stream(...)` or `complete(...)`.

## Constructors

MiniMax provides curated model helpers for both global and China-region endpoints:

```rust
use alchemy_llm::{
    minimax_m2_7, minimax_m2_7_highspeed,
    minimax_m2_5, minimax_m2_5_highspeed,
    minimax_m2_1, minimax_m2_1_highspeed,
    minimax_m2,
    minimax_cn_m2_7, minimax_cn_m2_7_highspeed,
    minimax_cn_m2_5, minimax_cn_m2_5_highspeed,
    minimax_cn_m2_1, minimax_cn_m2_1_highspeed,
    minimax_cn_m2,
};

let model = minimax_m2_7();
```

All helpers return `Model<MinimaxCompletions>` with:

- provider: `KnownProvider::Minimax` (global) or `KnownProvider::MinimaxCn` (CN)
- base URL: `https://api.minimax.io/v1/chat/completions` (global) or `https://api.minimax.chat/v1/chat/completions` (CN)
- reasoning: `true`
- context window: `204_800`
- max output tokens: `16_384`
- default input type: text

## Environment and Request Flow

Authentication is resolved through the normal provider environment lookup:

- `KnownProvider::Minimax` / `KnownProvider::MinimaxCn`
  -> `MINIMAX_API_KEY`
  -> `Authorization: Bearer ...` on the MiniMax request path

At the top-level API, MiniMax uses the same entry points as other providers:

- `stream(&model, &context, None)`
- `complete(&model, &context, None)`

## Authentication

Set:

```bash
MINIMAX_API_KEY=eyJhbG...
```

Top-level `stream(...)` and `complete(...)` resolve that key through the normal provider environment lookup.

## Reasoning and Streaming Notes

MiniMax supports multiple reasoning serialization styles:

### Explicit reasoning fields

When the model emits reasoning via structured fields, the runtime normalizes them into `Content::Thinking`:

- `reasoning_details` - Array of reasoning objects with `text` fields (preferred)
- `reasoning_content` - Raw reasoning text
- `reasoning` / `reasoning_text` - Alternative reasoning fields

The `thinking_signature` field records which source field provided the reasoning (e.g., `"reasoning_details"`).

### Think-tag fallback

For models that emit reasoning inside `<think>...</think>` tags within the regular content stream, the runtime uses `ThinkTagParser` to extract thinking blocks and emit separate `ThinkingStart/ThinkingDelta/ThinkingEnd` events.

This fallback activates when no explicit reasoning fields are present.

### Request parameters

For reasoning-capable models, the runtime automatically sets:

```json
{
  "reasoning_split": true
}
```

This instructs MiniMax to separate reasoning from content in the response.

## Temperature Constraints

MiniMax clamps temperature to `(0.0, 1.0]`:

- Values above 1.0 are clamped to 1.0
- Values at or below 0.0 are clamped to the smallest positive f64 (`f64::MIN_POSITIVE`)

The provider rejects 0.0 exactly, so the runtime ensures a valid lower bound.

## Tool Calling

MiniMax supports function/tool calling through the OpenAI-compatible schema:

- Tool definitions serialize into the standard OpenAI `tools` format
- Tool calls stream as `tool_calls` delta arrays
- Interleaved tool-call continuation chunks are handled safely
- Multi-turn tool-call argument assembly preserves trailing deltas

The runtime uses shared OpenAI-like stream block handling to merge tool-call chunks and manage interleaved content/reasoning/tool-call sequences.

## Stream Event Mapping

MiniMax SSE chunks map to canonical events:

| MiniMax Chunk | Canonical Event |
|---------------|-------------------|
| `content` delta | `TextStart/TextDelta/TextEnd` |
| `reasoning_details` | `ThinkingStart/ThinkingDelta/ThinkingEnd` |
| `tool_calls` | `ToolCallStart/ToolCallDelta/ToolCallEnd` |
| `usage` | Accumulated into `Usage`, emitted at stream end |
| `finish_reason` | Mapped to `StopReason` |

Block transitions finalize the current block before starting the next, maintaining the canonical event lifecycle.

## Stop Reason Mapping

MiniMax finish reasons map to canonical `StopReason`:

- `stop` -> `StopReason::Stop`
- `length` -> `StopReason::Length`
- `tool_calls` -> `StopReason::ToolUse`
- Provider errors or connection issues -> `StopReason::Error`

## Replay and Same-Provider Fidelity

MiniMax replay preserves:

- Thinking blocks (re-serialized with think tags: `<think>...</think>`)
- Tool-call IDs and arguments
- Block ordering

When replaying an assistant message to MiniMax, thinking content is wrapped in think tags within the assistant message content. This matches MiniMax's expected replay format for reasoning models.

## Headers

If you need provider-specific headers, use the existing extension points:

- `Model.headers`
- `OpenAICompletionsOptions.headers`

No MiniMax-specific public abstraction is required for this.

## Files to Reference

Implementation and integration points live in:

- `src/models/minimax.rs` - Model helpers (M2.5, M2.1, M2, CN variants)
- `src/providers/minimax.rs` - Main streaming runtime
- `src/providers/shared/openai_like_messages.rs` - Shared message serialization
- `src/providers/shared/stream_blocks.rs` - Shared block handling
- `src/providers/env.rs` - Environment variable lookup
- `src/stream/mod.rs` - Top-level dispatch

## Related Docs

- [architecture.md](./architecture.md)
- [../README.md](../README.md)
