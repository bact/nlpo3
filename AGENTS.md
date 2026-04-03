# Instructions for AI coding assistants

Use this file as the default working agreement for nlpO3.

## Scope and naming

- Use nlpO3 as the human-readable project name.
- Use nlpo3 as the package/module/crate/npm name unless a platform requires another scoped name.
- Keep names ASCII only: letters, digits, hyphen, underscore.

## Core priorities

- Keep APIs clean, intuitive, and consistent across Rust, Python, and Node.js.
- Prefer predictable behavior over convenience shortcuts.
- Prefer explicit errors over panic, silent fallback, or hidden defaults.
- Keep docs and code examples synchronized with real behavior.
- Optimize hot paths and avoid regressions in time or memory complexity.
- Improve developer ergonomics: type hints, stubs, examples, and actionable error messages.

## Reviewer hotspots (do these every time)

- API drift: verify signatures, defaults, and names are aligned across all bindings.
- Docs drift: verify docstrings, README, changelog, and comments match current code.
- Error semantics: verify callers get consistent error types and messages.
- Performance claims: avoid overstated complexity claims; document real guarantees only.
- Packaging metadata: keep manifest metadata complete, accurate, and ecosystem-friendly.
- Language examples: label code fences correctly and separate JavaScript and TypeScript when syntax differs.

## API design rules

- Use one obvious default API path. Advanced behavior should be opt-in.
- Keep equivalent concepts named the same across languages when possible.
- If behavior differs by implementation, document it explicitly.
- Breaking changes MUST include migration notes in changelog and README.
- Public examples MUST be runnable and tested against the current API.

## Error handling and behavior

- Do not introduce new panic paths in public or binding-exposed flows.
- Return rich, contextual errors (include path, parameter, and operation where useful).
- Never swallow real errors in helpers used by public APIs.
- Avoid ambiguous contracts such as panic-or-empty; standardize or document clearly.
- Define and preserve behavior for None/empty input and invalid dictionary/model paths.

## Performance and memory

- In hot paths, avoid accidental O(n^2) patterns and repeated rescans.
- When membership is checked frequently, prefer appropriate structures (for sorted vectors, use binary search).
- Reuse computed indices/maps when converting char and byte offsets repeatedly.
- Keep clone operations cheap for shared immutable state (for example Arc-backed internals).
- Add regression tests for known worst cases and benchmark-sensitive paths.

## Documentation and style

- Write short, direct, unambiguous sentences in American English.
- Use sentence case headings.
- Keep README and CHANGELOG user-focused.
- Put architecture and trade-offs in docs/implementation.md.
- Put future design proposals in docs/design.md.
- Keep terms consistent: tokenizer names, option names, and return types.

## Markdown and examples

- Use correct code fence labels: rust, python, javascript, typescript, json, bash, text.
- Use javascript fences only for plain JS examples.
- Use typescript fences only when types/interfaces/TS-only syntax appear.
- If both audiences matter, provide separate JavaScript and TypeScript snippets.

## Rust (core crate and CLI)

- Follow standard Rust style and keep clippy clean.
- Prefer Result-based APIs for fallible operations; avoid unwrap in non-test code.
- Keep trait and implementation docs accurate for return types and error behavior.
- Ensure thread-safety guarantees are true and tested when stated.
- Validate before finishing:
  - cargo fmt
  - cargo clippy --workspace --all-features --all-targets
  - cargo check --workspace --all-features
  - cargo test --workspace --all-features
  - cargo audit

### Rust metadata and packaging

- Keep Cargo.toml fields accurate: name, version, description, license, repository, homepage, documentation, readme, categories, keywords.
- Keep crates.io keyword limits valid (count and length).
- Remove unused dependencies.

## Node.js, JavaScript, and TypeScript bindings

- Preserve a class-based, ergonomic API aligned with Rust concepts.
- Propagate Rust errors as JS exceptions with clear messages.
- Keep TypeScript declarations and runtime exports synchronized.
- Validate docs and JSDoc imports against actual package name.

### Node.js package metadata

- Keep package.json complete and accurate: name, version, description, license, repository, homepage, bugs, keywords, files, engines.
- Ensure main/module/types/exports are internally consistent.
- Ensure npm pack --dry-run includes exactly required runtime files.
- Keep test script and build script aligned with emitted artifacts.

## Python binding

- Keep Python as a thin, well-typed wrapper over Rust core.
- Public Python APIs should raise Python exceptions, not panic.
- Keep type hints, .pyi stubs, and py.typed consistent with runtime behavior.
- Use Optional and container types precisely; do not overpromise types.
- Keep lint/type tooling clean for changed code.

### Python packaging metadata

- Keep pyproject.toml metadata accurate: name, version, description, license, URLs, classifiers, requires-python.
- Keep PEP 561 support complete: py.typed and matching .pyi stubs in distributions.
- Ensure MANIFEST.in includes files needed for source builds in monorepo layout.
- Verify wheel/sdist publish workflows and artifact naming are consistent.

## Security and dependency hygiene

- Avoid deprecated or unmaintained dependencies and actions.
- Recheck versions before pinning or widening constraints.
- Do not hardcode secrets or API keys.

## Changelog and release hygiene

- Follow Keep a Changelog and semantic versioning.
- Mark breaking changes clearly and include migration steps.
- Do not claim behavior or complexity that code does not guarantee.

## SPDX and file hygiene

- When practical, include SPDX file header tags.
- Defaults:
  - code: Apache-2.0
  - documentation: CC0-1.0
- Remove trailing whitespace.

## Done checklist (required)

- API and docs are aligned across Rust, Python, Node.js.
- Errors are explicit and consistent.
- Performance-sensitive changes include tests/bench updates when relevant.
- Package metadata is updated in all touched packages.
- README and CHANGELOG entries are updated for user-visible changes.
- Formatting, linting, tests, and audit pass for affected parts.
