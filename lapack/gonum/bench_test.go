// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/lapack/testlapack"
)

func BenchmarkDgeev(b *testing.B)         { testlapack.DgeevBenchmark(b, impl) }
func BenchmarkDggev(b *testing.B)         { testlapack.DggevBenchmark(b, impl) }
func BenchmarkDggevRightEV(b *testing.B)  { testlapack.DggevRightEVBenchmark(b, impl) }
func BenchmarkDggevSingular(b *testing.B) { testlapack.DggevSingularBenchmark(b, impl) }
func BenchmarkDggbal(b *testing.B)        { testlapack.DggbalBenchmark(b, impl) }
func BenchmarkDgges(b *testing.B)         { testlapack.DggesBenchmark(b, impl) }
func BenchmarkDgghrd(b *testing.B)        { testlapack.DgghrdBenchmark(b, impl) }
func BenchmarkDhgeqz(b *testing.B)        { testlapack.DhgeqzBenchmark(b, impl) }
func BenchmarkDlangb(b *testing.B)        { testlapack.DlangbBenchmark(b, impl) }
func BenchmarkDlantb(b *testing.B)        { testlapack.DlantbBenchmark(b, impl) }
func BenchmarkDlaqr5(b *testing.B)        { testlapack.Dlaqr5Benchmark(b, impl) }
func BenchmarkDlaic1(b *testing.B)        { testlapack.Dlaic1Benchmark(b, impl) }
func BenchmarkDtzrzf(b *testing.B)        { testlapack.DtzrzfBenchmark(b, impl) }
func BenchmarkDormrz(b *testing.B)        { testlapack.DormrzBenchmark(b, impl) }

func BenchmarkDggesScaledSort(b *testing.B) {
	testlapack.DggesScaledSortBenchmark(b, impl)
}

func BenchmarkDggesIsolated(b *testing.B) {
	testlapack.DggesIsolatedBenchmark(b, impl)
}

// The external-oracle test file installs this factory only in Netlib builds.
var dggesNetlibBenchmarkRun func(n int, kind, vectors string, sorting, native bool) func() (int, bool)
var dggesNetlibBenchmarkKinds []string

func BenchmarkDggesNetlib(b *testing.B) {
	if dggesNetlibBenchmarkRun == nil {
		b.Skip("requires netlib, darwin, and cgo")
	}
	testlapack.DggesComparisonBenchmark(b, dggesNetlibBenchmarkKinds, dggesNetlibBenchmarkRun)
}
