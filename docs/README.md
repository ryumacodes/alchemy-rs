---
summary: "Documentation index for Alchemy crate guides, provider docs, and API references"
read_when:
  - You need a starting point for project documentation
  - You want to find provider-specific usage guides
  - You want to understand the latest documented feature set
---

# Alchemy Documentation Index

## Start Here

- [../README.md](../README.md) - Project overview, installation, and quick start
- [api/lib.md](./api/lib.md) - Public API exports from `src/lib.rs`
- [api/error.md](./api/error.md) - Error and `Result` contract

## Provider Guides

- [providers/minimax.md](./providers/minimax.md) - First-class MiniMax provider (global + CN)
- [providers/zai.md](./providers/zai.md) - First-class z.ai GLM provider

## Utilities

- [utils/transform.md](./utils/transform.md) - Cross-provider conversation transformation

## Latest Release (0.1.6)

The latest published crate release focuses on MiniMax streamed tool-call correctness:

- Interleaved tool-call argument deltas are now merged reliably
- Id-less continuation chunks are handled consistently in the shared stream parser
- Regression tests cover continuation + reasoning/text interleave scenarios

For release details, see [../CHANGELOG.md](../CHANGELOG.md#016---2026-02-21).

## Main Branch Updates (post-0.1.6)

- New first-class z.ai provider path (`Api::ZaiCompletions`)
- New built-in GLM model constructors in `src/models/zai.rs`
- New z.ai examples:
  - `examples/zai_glm_simple_chat.rs`
  - `examples/zai_glm_tool_call_smoke.rs`
- Architecture boundary checks via ast-grep rules in `rules/`
