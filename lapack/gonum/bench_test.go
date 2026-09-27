// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/testlapack"
)

func BenchmarkDgeev(b *testing.B)         { testlapack.DgeevBenchmark(b, impl) }
func BenchmarkDgebrd(b *testing.B)        { testlapack.DgebrdBenchmark(b, impl) }
func BenchmarkDgesvd(b *testing.B)        { testlapack.DgesvdBenchmark(b, impl) }
func BenchmarkDggev(b *testing.B)         { testlapack.DggevBenchmark(b, impl) }
func BenchmarkDggevRightEV(b *testing.B)  { testlapack.DggevRightEVBenchmark(b, impl) }
func BenchmarkDggevSingular(b *testing.B) { testlapack.DggevSingularBenchmark(b, impl) }
func BenchmarkDggbal(b *testing.B)        { testlapack.DggbalBenchmark(b, impl) }
func BenchmarkDgges(b *testing.B)         { testlapack.DggesBenchmark(b, impl) }
func BenchmarkDggesControl(b *testing.B)  { testlapack.DggesControlBenchmark(b, impl) }
func BenchmarkDgghrd(b *testing.B)        { testlapack.DgghrdBenchmark(b, impl) }
func BenchmarkDgghrdControl(b *testing.B) { testlapack.DgghrdControlBenchmark(b, impl) }
func BenchmarkDhgeqz(b *testing.B)        { testlapack.DhgeqzBenchmark(b, impl) }

func BenchmarkDhgeqzSweepDouble(b *testing.B) {
	for _, mode := range []struct {
		name string
		q, z bool
	}{
		{"none", false, false},
		{"q", true, false},
		{"z", false, true},
		{"both", true, true},
	} {
		for _, size := range []struct{ n, extra int }{
			{3, 0}, {4, 0}, {5, 0}, {32, 0}, {64, 0},
			{128, 0}, {256, 0}, {256, 3},
		} {
			n, ld := size.n, size.n+size.extra
			b.Run(fmt.Sprintf("mode=%s/n=%d/stride=%d", mode.name, n, ld), func(b *testing.B) {
				hOrig := make([]float64, (n-1)*ld+n)
				tOrig := make([]float64, len(hOrig))
				vOrig := make([]float64, len(hOrig))
				for i := range n {
					for j := max(0, i-1); j < n; j++ {
						hOrig[i*ld+j] = float64((i*17+j*13+7)%23-11) / 16
					}
					for j := i; j < n; j++ {
						tOrig[i*ld+j] = float64((i*11+j*7+3)%19-9) / 16
					}
					tOrig[i*ld+i] += float64(n)
					vOrig[i*ld+i] = 1
				}
				h, t := make([]float64, len(hOrig)), make([]float64, len(tOrig))
				var q, z []float64
				if mode.q {
					q = make([]float64, len(vOrig))
				}
				if mode.z {
					z = make([]float64, len(vOrig))
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(h, hOrig)
					copy(t, tOrig)
					copy(q, vOrig)
					copy(z, vOrig)
					Implementation{}.doQZSweepDouble(true, mode.q, mode.z, n, 0, n-1, 0, n-1,
						h, ld, t, ld, q, ld, z, ld, 1, 1/float64(n), dlamchS)
				}
				b.StopTimer()
				for _, a := range [][]float64{h, t, q, z} {
					for _, v := range a {
						if math.IsNaN(v) || math.IsInf(v, 0) {
							b.Fatal("non-finite sweep output")
						}
					}
				}
			})
		}
	}
}
func BenchmarkDlangb(b *testing.B)        { testlapack.DlangbBenchmark(b, impl) }
func BenchmarkDlantb(b *testing.B)        { testlapack.DlantbBenchmark(b, impl) }
func BenchmarkDlaqr5(b *testing.B)        { testlapack.Dlaqr5Benchmark(b, impl) }
func BenchmarkDlaic1(b *testing.B)        { testlapack.Dlaic1Benchmark(b, impl) }
func BenchmarkDlasr(b *testing.B)         { testlapack.DlasrBenchmark(b, impl) }
func BenchmarkDlasrVariable(b *testing.B) { testlapack.DlasrVariableBenchmark(b, impl) }
func BenchmarkDlarft(b *testing.B)        { testlapack.DlarftBenchmark(b, impl) }
func BenchmarkZlarfg(b *testing.B)        { testlapack.ZlarfgBenchmark(b, impl) }
func BenchmarkDtzrzf(b *testing.B)        { testlapack.DtzrzfBenchmark(b, impl) }
func BenchmarkDormrz(b *testing.B)        { testlapack.DormrzBenchmark(b, impl) }
func BenchmarkDormlqLeft(b *testing.B)    { testlapack.DormlqLeftBenchmark(b, impl) }
func BenchmarkDtgex2Scratch(b *testing.B) { testlapack.Dtgex2ScratchBenchmark(b, impl) }
func BenchmarkDtgsy2Scratch(b *testing.B) { testlapack.Dtgsy2ScratchBenchmark(b, impl) }

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
