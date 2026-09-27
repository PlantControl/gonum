# AGENTS.md

PlantControl fork of Gonum. Module `plantcontrol.org/v1/gonum`. Everything after the fork is AI-generated (see CONTRIBUTING.md).

## Scope

- Focus on `blas/` and `lapack/`. Add LAPACK routines when a consumer needs them.
- SIMD work lives on `codex/arm64-simd-blas`; `main` stays portable until Go SIMD is GA.
- Other packages get maintenance only. Don't port upstream churn unless asked.

## Commands

```sh
go build ./... && go test -short ./...
go test -tags noasm ./blas/... ./lapack/...   # also: safe, bounds
go tool golang.org/x/tools/cmd/goimports -w . && gofmt -s -w .
go generate ./...                              # rdf requires ragel
.github/workflows/script.d/check-imports.sh
.github/workflows/script.d/check-copyright.sh
```

## Layering

`mat` → `blas64`/`lapack64` → `blas/gonum`, `lapack/gonum`.

## LAPACK

- Implementation: `lapack/gonum/dxxxxx.go`. Reusable tests: `lapack/testlapack/dxxxxx.go`, wired up in `lapack/gonum/lapack_test.go`. Benchmarks: `lapack/testlapack/dxxxxx_bench.go`, wired up in `lapack/gonum/bench_test.go`.
- Fork-only tests: `lapack/gonum/*_test.go`, or `*_internal_test.go` for unexported code.
- Netlib oracle: cgo bridge in `lapack/gonum/internal/netlib/`, used only from `dxxxxx_netlib_test.go`.
- Panic on invalid input and return bool on numerical failure. Support workspace queries (`lwork == -1`). Use `Dgeev` and `Dhseqr` as style references.
- Netlib translations: keep its BSD notice, cite the reference version, note any deviations, and don't imply endorsement.

## Rules

- Keep the header `// Copyright ©20XX The Gonum Authors. All rights reserved.`; CI requires it.
- Don't use `math/rand` (use `math/rand/v2`) or `github.com/gonum/*` imports.
- Cite algorithm sources in comments. Put fixtures in the package's `testdata/`.
- Commits: `pkg: summary` + `Co-Authored-By` naming the agent.
