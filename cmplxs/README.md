# Gonum cmplxs

[![go.dev reference](https://pkg.go.dev/badge/plantcontrol.org/v1/gonum/cmplxs)](https://pkg.go.dev/plantcontrol.org/v1/gonum/cmplxs)
[![GoDoc](https://godocs.io/plantcontrol.org/v1/gonum/cmplxs?status.svg)](https://godocs.io/plantcontrol.org/v1/gonum/cmplxs)

Package cmplxs provides a set of helper routines for dealing with slices of complex128.
The functions avoid allocations to allow for use within tight loops without garbage collection overhead.
