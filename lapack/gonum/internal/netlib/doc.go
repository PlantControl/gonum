// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package netlib calls reference LAPACK through cgo for the differential
// tests built with -tags netlib (Homebrew LAPACK on darwin).
//
// Reference LAPACK's XERBLA exits the process with status 0 on an illegal
// argument, so "go test -tags netlib" can print ok after running only part
// of the tests. Run the netlib tests through check.sh in this directory,
// which fails unless every listed test finished:
//
//	sh lapack/gonum/internal/netlib/check.sh
package netlib
