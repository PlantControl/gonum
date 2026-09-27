# Contributing

## Policy: AI-generated code only

Every change after the fork must be written by an AI coding agent. **Human-written code is not accepted.** Humans set direction, file issues and review changes.

Upstream Gonum does not accept AI-generated code, which is why this fork exists. For pre-fork code or human-written contributions, go to [gonum.org](https://www.gonum.org) / [github.com/gonum/gonum](https://github.com/gonum/gonum).

## Pull requests

- Name the agent and model that wrote the change in the PR description, and add a `Co-Authored-By` trailer.
- Title: `pkg: summary` (Go convention).
- Include tests. Performance changes need `benchstat` output.
- CI must pass: `goimports`, `go generate` with no diff, `check-imports` and `check-copyright`.
- Agents follow [AGENTS.md](AGENTS.md).

## Issues

Anyone may file an issue. Include a minimal reproducer and the output of `go version`.
