// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// BenchmarkSgemmCrossover measures the serial/parallel crossover boundary of
// the Sgemm dispatch gate (see sgemmParallelWorkerCount and
// sgemmMinParFLOPsPerWorker) on shapes near the cutoff, at four workers and at
// the machine default. The grid matches BenchmarkDgemmCrossover so the f32 and
// f64 crossovers can be compared directly. mode=serial forces sgemmSerial,
// mode=parallel forces sgemmParallelBlocked, and mode=dispatch goes through
// Sgemm, taking whatever path the platform-calibrated gate picks. Compare the
// forced paths with:
//
//	go test ./blas/gonum/ -run=NONE -bench=BenchmarkSgemmCrossover -count=10 | benchstat -col /mode -
func BenchmarkSgemmCrossover(b *testing.B) {
	shapes := []struct{ m, n, k int }{
		{60, 60, 60},    // 216e3 ops, 1 (m,n) block: block gate keeps it serial
		{80, 80, 80},    // 512e3 ops, 4 blocks
		{100, 100, 100}, // 1.00e6 ops, 4 blocks: the darwin/arm64 four-worker Dgemm regression shape
		{126, 126, 126}, // 2.00e6 ops, 4 blocks
		{160, 160, 160}, // 4.10e6 ops, 9 blocks
		{200, 200, 200}, // 8.00e6 ops, 16 blocks
		{1000, 100, 10}, // 1.00e6 ops, 32 blocks, skinny
		{100, 1000, 10}, // 1.00e6 ops, 32 blocks, skinny
		{128, 128, 8},   // 131e3 ops, 4 blocks: architecture-sensitive crossover
	}
	workerCounts := []int{4}
	if def := runtime.GOMAXPROCS(0); def != 4 {
		workerCounts = append(workerCounts, def)
	}
	for _, s := range shapes {
		for _, w := range workerCounts {
			for _, mode := range []string{"serial", "parallel", "dispatch"} {
				name := fmt.Sprintf("shape=%dx%dx%d/workers=%d/mode=%s", s.m, s.n, s.k, w, mode)
				b.Run(name, func(b *testing.B) {
					benchSgemmCrossover(b, s.m, s.n, s.k, w, mode)
				})
			}
		}
	}
}

func benchSgemmCrossover(b *testing.B, m, n, k, workers int, mode string) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(workers))
	rnd := rand.New(rand.NewPCG(uint64(m*1_000_000+n*1000+k), 29))
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	c := make([]float32, m*n)
	for i := range a {
		a[i] = float32(rnd.NormFloat64())
	}
	for i := range bb {
		bb[i] = float32(rnd.NormFloat64())
	}
	var impl Implementation
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		switch mode {
		case "serial":
			sgemmSerial(false, false, m, n, k, a, k, bb, n, c, n, 1)
		case "parallel":
			sgemmParallelBlocked(false, false, m, n, k, a, k, bb, n, c, n, 1)
		case "dispatch":
			impl.Sgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1, a, k, bb, n, 0, c, n)
		}
	}
}
