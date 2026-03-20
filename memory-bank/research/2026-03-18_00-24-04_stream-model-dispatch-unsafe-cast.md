---
title: "stream model dispatch unsafe cast – Research"
phase: Research
date: "2026-03-18 00:24:04"
owner: "OpenAI Codex"
tags: [research, stream-dispatch, model, unsafe]
---

## Structure

- `README.md:11` states the currently implemented streaming families are Anthropic-style Messages, OpenAI-compatible Completions, MiniMax Completions, and z.ai GLM Completions.
- `src/lib.rs:3-6` declares the top-level `providers`, `stream`, and `types` modules.
- `src/lib.rs:20-24` re-exports provider entry points and re-exports `stream`, `complete`, and `AssistantMessageEventStream`.
- `src/types/mod.rs:1-7` declares the `api` and `model` submodules.
- `src/types/mod.rs:14` re-exports `Api`, `ApiType`, `CompatibilityOptions`, `KnownProvider`, `NoCompat`, and `Provider`.
- `src/types/mod.rs:23-26` re-exports `Model` and the concrete API marker types: `AnthropicMessages`, `BedrockConverseStream`, `GoogleGenerativeAi`, `GoogleVertex`, `MinimaxCompletions`, `OpenAICompletions`, `OpenAIResponses`, and `ZaiCompletions`.
- `src/stream/mod.rs:23` defines the public generic `stream<TApi>(model: &Model<TApi>, ...)` dispatch function.
- `src/stream/mod.rs:115` defines the public generic `complete<TApi>(model: &Model<TApi>, ...)` wrapper over `stream(...)`.

## Key Files

- `src/stream/mod.rs:23-113` → generic dispatch function that reads `model.api.api()`, resolves API-key options, matches on `Api`, and forwards to provider-specific runtime functions.
- `src/stream/mod.rs:52-93` → `match api` dispatch block for implemented runtime families.
- `src/stream/mod.rs:56-57` → raw-pointer cast from `&Model<TApi>` to `&Model<OpenAICompletions>` followed by `unsafe` dereference.
- `src/stream/mod.rs:67-68` → raw-pointer cast from `&Model<TApi>` to `&Model<MinimaxCompletions>` followed by `unsafe` dereference.
- `src/stream/mod.rs:78-79` → raw-pointer cast from `&Model<TApi>` to `&Model<ZaiCompletions>` followed by `unsafe` dereference.
- `src/stream/mod.rs:83-84` → raw-pointer cast from `&Model<TApi>` to `&Model<AnthropicMessages>` followed by `unsafe` dereference.
- `src/types/model.rs:16-28` → generic `Model<TApi: ApiType>` definition; the generic parameter is stored in `api: TApi` and also appears in `compat: Option<TApi::Compat>`.
- `src/types/api.rs:145-147` → public `ApiType` trait definition with associated type `Compat` and method `fn api(&self) -> Api`.
- `src/types/model.rs:34-114` → repository-local `ApiType` implementations for concrete marker structs.
- `src/providers/openai_completions.rs:101-105` → `stream_openai_completions(model: &Model<OpenAICompletions>, ...)`.
- `src/providers/minimax.rs:32-36` → `stream_minimax_completions(model: &Model<MinimaxCompletions>, ...)`.
- `src/providers/zai.rs:25-29` → `stream_zai_completions(model: &Model<ZaiCompletions>, ...)`.
- `src/providers/anthropic.rs:14-18` → `stream_anthropic_messages(model: &Model<AnthropicMessages>, ...)`.
- `src/providers/kimi.rs:14-18` → `stream_kimi_messages(model: &Model<AnthropicMessages>, ...)`.
- `docs/providers/architecture.md:322-325` → provider architecture doc states end-to-end dispatch tests should prove that `stream(...)` routes the model to the correct runtime and `complete(...)` returns the final canonical `AssistantMessage`.

## Patterns Found

### 1. Generic model type carries both API marker and compat type

- `src/types/model.rs:16-28` defines `Model<TApi: ApiType>`.
- `src/types/model.rs:19` stores the generic marker value as `pub api: TApi`.
- `src/types/model.rs:28` stores `pub compat: Option<TApi::Compat>`.
- `src/types/api.rs:145-147` defines the only trait contract required by `TApi`: associated type `Compat` and `fn api(&self) -> Api`.

### 2. Repository-local `ApiType` implementations are marker structs

Repository search for `impl ApiType for` in `src/**/*.rs` returned these locations:

- `src/types/model.rs:34` → `impl ApiType for AnthropicMessages`
- `src/types/model.rs:45` → `impl ApiType for BedrockConverseStream`
- `src/types/model.rs:56` → `impl ApiType for OpenAICompletions`
- `src/types/model.rs:67` → `impl ApiType for OpenAIResponses`
- `src/types/model.rs:78` → `impl ApiType for MinimaxCompletions`
- `src/types/model.rs:89` → `impl ApiType for ZaiCompletions`
- `src/types/model.rs:100` → `impl ApiType for GoogleGenerativeAi`
- `src/types/model.rs:111` → `impl ApiType for GoogleVertex`

### 3. `stream(...)` dispatch is driven by runtime `Api` enum values

- `src/stream/mod.rs:31` reads the runtime discriminator with `let api = model.api.api();`.
- `src/stream/mod.rs:52` begins `match api`.
- `src/stream/mod.rs:54-61` handles `Api::OpenAICompletions`.
- `src/stream/mod.rs:65-72` handles `Api::MinimaxCompletions`.
- `src/stream/mod.rs:76-80` handles `Api::ZaiCompletions`.
- `src/stream/mod.rs:82-93` handles `Api::AnthropicMessages`, with an additional provider check for Kimi at `src/stream/mod.rs:86-92`.

### 4. Provider runtime entry points require concrete `Model<...>` types

- `src/providers/openai_completions.rs:101-105` accepts `&Model<OpenAICompletions>`.
- `src/providers/minimax.rs:32-36` accepts `&Model<MinimaxCompletions>`.
- `src/providers/zai.rs:25-29` accepts `&Model<ZaiCompletions>`.
- `src/providers/anthropic.rs:14-18` accepts `&Model<AnthropicMessages>`.
- `src/providers/kimi.rs:14-18` accepts `&Model<AnthropicMessages>`.

### 5. Repository-local `unsafe` usage is concentrated in `src/stream/mod.rs`

Repository search for `unsafe` in `src/**/*.rs` returned:

- `src/stream/mod.rs:57` → dereference after cast to `&Model<OpenAICompletions>`
- `src/stream/mod.rs:68` → dereference after cast to `&Model<MinimaxCompletions>`
- `src/stream/mod.rs:79` → dereference after cast to `&Model<ZaiCompletions>`
- `src/stream/mod.rs:84` → dereference after cast to `&Model<AnthropicMessages>`

## Dependencies

- `src/lib.rs:20-24` re-exports the stream and provider entry points.
- `src/stream/mod.rs:3-8` imports `Error`/`Result`, provider runtime functions, `OpenAICompletionsOptions`, and the type-layer symbols used in dispatch.
- `src/providers/mod.rs:1-12` declares the provider modules and re-exports `stream_anthropic_messages`, `get_env_api_key`, `stream_kimi_messages`, `stream_minimax_completions`, `stream_openai_completions`, `OpenAICompletionsOptions`, and `stream_zai_completions`.
- `src/stream/mod.rs:23-113` depends on provider runtime functions with concrete model types:
  - `src/providers/openai_completions.rs:101-105`
  - `src/providers/minimax.rs:32-36`
  - `src/providers/zai.rs:25-29`
  - `src/providers/anthropic.rs:14-18`
  - `src/providers/kimi.rs:14-18`
- `docs/providers/architecture.md:145` states the public streaming surface is `AssistantMessageEventStream`.
- `docs/providers/architecture.md:322-325` documents end-to-end dispatch tests as part of the provider contract.

## Tests and Coverage Locations

- `src/stream/mod.rs:136-157` defines `minimax_test_model(base_url: &str) -> Model<MinimaxCompletions>`.
- `src/stream/mod.rs:158-179` defines `zai_test_model(base_url: &str) -> Model<ZaiCompletions>`.
- `src/stream/mod.rs:180-201` defines `featherless_test_model(base_url: &str) -> Model<OpenAICompletions>`.
- `src/stream/mod.rs:202-220` defines `assert_dispatches_to_provider<TApi>(model: Model<TApi>, expected_api: Api)`.
- `src/stream/mod.rs:223-226` tests MiniMax dispatch.
- `src/stream/mod.rs:229-232` tests z.ai dispatch.
- `src/stream/mod.rs:234-255` defines `anthropic_test_model(base_url: &str) -> Model<AnthropicMessages>`.
- `src/stream/mod.rs:257-260` tests Anthropic dispatch.
- `src/stream/mod.rs:263-266` tests Featherless dispatch through the OpenAI-completions runtime.
- `src/stream/mod.rs:269-282` tests API-key requirement logic for Google Vertex and Bedrock.
- `src/stream/mod.rs:284-287` tests stop-reason conversion.

## Symbol Index

- `src/lib.rs:24` → `pub use stream::{complete, stream, AssistantMessageEventStream};`
- `src/types/mod.rs:14` → `pub use api::{Api, ApiType, CompatibilityOptions, KnownProvider, NoCompat, Provider};`
- `src/types/model.rs:16` → `pub struct Model<TApi: ApiType>`
- `src/types/api.rs:7` → `pub enum Api`
- `src/types/api.rs:64` → `pub enum KnownProvider`
- `src/types/api.rs:108` → `pub enum Provider`
- `src/types/api.rs:145` → `pub trait ApiType`
- `src/providers/openai_completions.rs:22` → `pub struct OpenAICompletionsOptions`
- `src/providers/openai_completions.rs:101` → `pub fn stream_openai_completions`
- `src/providers/minimax.rs:32` → `pub fn stream_minimax_completions`
- `src/providers/zai.rs:25` → `pub fn stream_zai_completions`
- `src/providers/anthropic.rs:14` → `pub fn stream_anthropic_messages`
- `src/providers/kimi.rs:14` → `pub fn stream_kimi_messages`
- `src/models/featherless.rs:19` → `pub fn featherless_model(id: impl Into<String>) -> Model<OpenAICompletions>`
- `src/models/kimi.rs:18` → `pub fn kimi_k2_5() -> Model<AnthropicMessages>`
- `src/models/minimax.rs:36` → `pub fn minimax_m2_5() -> Model<MinimaxCompletions>`
- `src/models/zai.rs:39` → `pub fn glm_5() -> Model<ZaiCompletions>`
- `src/models/anthropic.rs:29` → `pub fn claude_opus_4_6() -> Model<AnthropicMessages>`
