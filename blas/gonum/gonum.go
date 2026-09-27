// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./single_precision.bash

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/internal/math32"
)

type Implementation struct{}

// [SD]gemm behavior constants. These are kept here to keep them out of the
// way during single precision code generation.
const (
	blockSize   = 64 // b x b matrix
	minParBlock = 4  // minimum number of (m,n) blocks needed to go parallel
	// The Dgemm serial/parallel crossover differs substantially for the pure-Go
	// kernels used on Darwin arm64. Preserve the existing calibration as the
	// default and use the measured lower floor only on that platform.
	minParFLOPsPerWorkerDefault     = 1 << 17
	minParFLOPsPerWorkerDarwinARM64 = 1 << 15
	// The Sgemm crossover was measured separately (BenchmarkSgemmCrossover,
	// amd64): despite float32 doing twice the multiply-adds per cache line,
	// the serial/parallel boundary lands in the same bracket as Dgemm —
	// serial wins at 80x80x80 (512e3 ops, 4 blocks), parallel wins at
	// 100x100x100 (1e6 ops) — so the per-worker floor matches. The Darwin
	// arm64 floor reuses the Dgemm default/Darwin ratio; it has not been
	// measured on that platform.
	sgemmMinParFLOPsPerWorkerDefault     = 1 << 17
	sgemmMinParFLOPsPerWorkerDarwinARM64 = 1 << 15
)

// blocks returns the number of divisions of the dimension length with the given
// block size.
func blocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

// dcabs1 returns |real(z)|+|imag(z)|.
func dcabs1(z complex128) float64 {
	return math.Abs(real(z)) + math.Abs(imag(z))
}

// scabs1 returns |real(z)|+|imag(z)|.
func scabs1(z complex64) float32 {
	return math32.Abs(real(z)) + math32.Abs(imag(z))
}
