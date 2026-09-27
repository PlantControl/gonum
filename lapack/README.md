Gonum LAPACK
======
[![go.dev reference](https://pkg.go.dev/badge/plantcontrol.org/v1/gonum/lapack)](https://pkg.go.dev/plantcontrol.org/v1/gonum/lapack)
[![GoDoc](https://godocs.io/plantcontrol.org/v1/gonum/lapack?status.svg)](https://godocs.io/plantcontrol.org/v1/gonum/lapack)

A collection of packages to provide LAPACK functionality for the Go programming
language (http://golang.org). This provides a partial implementation in native go
and a wrapper using cgo to a c-based implementation.

## Installation

```
  go get plantcontrol.org/v1/gonum/lapack/...
```

## Packages

### lapack

Defines the LAPACK API based on http://www.netlib.org/lapack/lapacke.html

### lapack/gonum

Go implementation of the LAPACK API (incomplete, implements the `float64` API).

### lapack/lapack64

Wrappers for an implementation of the double (i.e., `float64`) precision real parts of
the LAPACK API.
