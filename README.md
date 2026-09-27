# PlantControl Gonum

[![CI](https://github.com/PlantControl/gonum/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PlantControl/gonum/actions/workflows/ci.yml)
[![go.dev reference](https://pkg.go.dev/badge/plantcontrol.org/v1/gonum)](https://pkg.go.dev/plantcontrol.org/v1/gonum)

A fork of [Gonum](https://www.gonum.org), maintained by PlantControl for control-systems work.

**All code written after the fork is AI-generated.** We do not accept human-written code. See [CONTRIBUTING.md](CONTRIBUTING.md).

```sh
go get plantcontrol.org/v1/gonum@latest
```

## Focus

- BLAS (`blas/...`) and LAPACK (`lapack/...`) are the core. New LAPACK routines are added when we need them.
- Branch [`simd`](https://github.com/PlantControl/gonum/tree/simd) holds Go SIMD kernels for ARM64 and AMD64, which outperform upstream Gonum. Most development happens there. It will be merged into `main` once Go SIMD reaches general availability; until then `main` stays portable.
- The other packages are kept in sync but are not actively developed.

## Provenance

Forked from `gonum/gonum` at [`fc402bc4`](https://github.com/gonum/gonum/commit/fc402bc4) (after `v0.17.0`, 2025-12-29). Code up to that point was written by humans and belongs to [The Gonum Authors](AUTHORS) and [contributors](CONTRIBUTORS). For that code, use [gonum.org](https://www.gonum.org) and [github.com/gonum/gonum](https://github.com/gonum/gonum).

## Build tags

`safe` (no assembly or unsafe), `noasm` (no assembly), `bounds` (extra bounds checks).

## License

BSD 3-clause; see [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). `graph/formats/dot` is also released under CC0. The W3C test suites in `graph/formats/rdf` are also covered by the W3C licenses.
