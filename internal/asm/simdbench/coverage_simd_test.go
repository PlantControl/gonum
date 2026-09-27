// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"math"
	"testing"

	asmc128 "plantcontrol.org/v1/gonum/internal/asm/c128"
	asmc64 "plantcontrol.org/v1/gonum/internal/asm/c64"
	asmf32 "plantcontrol.org/v1/gonum/internal/asm/f32"
	asmf64 "plantcontrol.org/v1/gonum/internal/asm/f64"
)

type kernelRun struct {
	run    func()
	result func() any
}

func TestSIMDCandidateEquivalence(t *testing.T) {
	for _, entry := range AMD64Assembly {
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65} {
			t.Run(fmt.Sprintf("%s/%s/n=%d", entry.Package, entry.Symbol, n), func(t *testing.T) {
				current := newKernelRun(entry, n, false, false)
				candidate := newKernelRun(entry, n, true, false)
				current.run()
				candidate.run()
				if !sameKernelResult(current.result(), candidate.result()) {
					t.Fatalf("current=%v SIMD=%v", current.result(), candidate.result())
				}
			})
		}
	}
}

func TestSIMDCandidateAllocations(t *testing.T) {
	for _, entry := range AMD64Assembly {
		for _, useSIMD := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/simd=%t", entry.Package, entry.Symbol, useSIMD), func(t *testing.T) {
				runner := newKernelRun(entry, 33, useSIMD, true)
				allocs := testing.AllocsPerRun(100, runner.run)
				benchmarkSink = runner.result()
				if allocs != 0 {
					t.Fatalf("got %g allocations per call, want zero", allocs)
				}
			})
		}
	}
}

var benchmarkSink any

func BenchmarkCurrentVsSIMD(b *testing.B) {
	for _, entry := range AMD64Assembly {
		for _, size := range benchmarkSizes(entry) {
			for _, implementation := range []struct {
				name string
				simd bool
			}{{"current", false}, {"simd", true}} {
				name := fmt.Sprintf("%s/%s/n=%d/implementation=%s", entry.Package, entry.Symbol, size, implementation.name)
				b.Run(name, func(b *testing.B) {
					runner := newKernelRun(entry, size, implementation.simd, true)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						clearBenchmarkAVXState()
						runner.run()
					}
					b.StopTimer()
					benchmarkSink = runner.result()
				})
			}
		}
	}
}

func benchmarkSizes(entry Entry) []int {
	if entry.Symbol == "Ger" || entry.Symbol == "GemvN" || entry.Symbol == "GemvT" {
		return []int{8, 64}
	}
	return []int{31, 4096}
}

func choose[T any](useSIMD bool, candidate, assembly T) T {
	if useSIMD {
		return candidate
	}
	return assembly
}

func newKernelRun(entry Entry, n int, useSIMD, stableInputs bool) kernelRun {
	return newKernelRunStride(entry, n, 2, useSIMD, stableInputs)
}

// newKernelRunStride keeps the timed call identical across stride experiments.
// Strides are positive; dependency and negative-stride tests live with the kernels.
func newKernelRunStride(entry Entry, n, stride int, useSIMD, stableInputs bool) kernelRun {
	realScale := 0.75
	complexScale := complex(0.75, -0.25)
	if stableInputs {
		// Repeated in-place scaling must not decay into subnormal operands.
		realScale = -1
		complexScale = 1i
	}
	key := entry.Package + "." + entry.Symbol
	switch key {
	case "f64.Add":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, asmf64.AddSIMD, asmf64.Add)
		return sliceRun(func() { fn(y, x) }, y)
	case "f64.AddConst":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, asmf64.AddConstSIMD, asmf64.AddConst)
		return sliceRun(func() { fn(0.75, x) }, x)
	case "f64.AxpyUnitary":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, asmf64.AxpyUnitarySIMD, asmf64.AxpyUnitary)
		return sliceRun(func() { fn(0.75, x, y) }, y)
	case "f64.AxpyUnitaryTo":
		x, y, dst := f64Values(n, 0.2), f64Values(n, 0.7), make([]float64, n)
		fn := choose(useSIMD, asmf64.AxpyUnitaryToSIMD, asmf64.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x, y) }, dst)
	case "f64.AxpyInc":
		x, y := f64Values(stride*n, 0.2), f64Values(stride*n, 0.7)
		fn := choose(useSIMD, asmf64.AxpyIncSIMD, asmf64.AxpyInc)
		return sliceRun(func() { fn(0.75, x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, y)
	case "f64.AxpyIncTo":
		x, y, dst := f64Values(stride*n, 0.2), f64Values(stride*n, 0.7), make([]float64, stride*n)
		fn := choose(useSIMD, asmf64.AxpyIncToSIMD, asmf64.AxpyIncTo)
		return sliceRun(func() { fn(dst, uintptr(stride), 0, 0.75, x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, dst)
	case "f64.CumSum":
		x, dst := f64Values(n, 0.2), make([]float64, n)
		fn := choose(useSIMD, asmf64.CumSumSIMD, asmf64.CumSum)
		return sliceRun(func() { fn(dst, x) }, dst)
	case "f64.CumProd":
		x, dst := f64Values(n, 0.98), make([]float64, n)
		if stableInputs {
			// Reciprocal pairs keep every prefix normal. The previous values
			// exceeded one in magnitude and overflowed on long inputs.
			for i := range x {
				x[i] = 1.25
				if i%2 != 0 {
					x[i] = 0.8
				}
				if i%3 == 0 {
					x[i] = -x[i]
				}
			}
		}
		fn := choose(useSIMD, asmf64.CumProdSIMD, asmf64.CumProd)
		return sliceRun(func() { fn(dst, x) }, dst)
	case "f64.Div":
		x, y := f64Values(n, 1.2), f64Values(n, 2.1)
		if stableInputs {
			for i := range x {
				x[i] = -1
			}
		}
		fn := choose(useSIMD, asmf64.DivSIMD, asmf64.Div)
		return sliceRun(func() { fn(y, x) }, y)
	case "f64.DivTo":
		x, y, dst := f64Values(n, 1.2), f64Values(n, 2.1), make([]float64, n)
		fn := choose(useSIMD, asmf64.DivToSIMD, asmf64.DivTo)
		return sliceRun(func() { fn(dst, x, y) }, dst)
	case "f64.DotUnitary":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, asmf64.DotUnitarySIMD, asmf64.DotUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.DotInc":
		x, y := f64Values(stride*n, 0.2), f64Values(stride*n, 0.7)
		fn := choose(useSIMD, asmf64.DotIncSIMD, asmf64.DotInc)
		var result float64
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "f64.L1Norm":
		x := f64Values(n, -0.5)
		fn := choose(useSIMD, asmf64.L1NormSIMD, asmf64.L1Norm)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.L1NormInc":
		x := f64Values(stride*n, -0.5)
		fn := choose(useSIMD, asmf64.L1NormIncSIMD, asmf64.L1NormInc)
		var result float64
		return scalarRun(func() { result = fn(x, n, stride) }, &result)
	case "f64.L1Dist":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, asmf64.L1DistSIMD, asmf64.L1Dist)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.LinfDist":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, asmf64.LinfDistSIMD, asmf64.LinfDist)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.L2NormUnitary":
		x := f64Values(n, -0.5)
		fn := choose(useSIMD, asmf64.L2NormUnitarySIMD, asmf64.L2NormUnitary)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.L2NormInc":
		x := f64Values(stride*n, -0.5)
		fn := choose(useSIMD, asmf64.L2NormIncSIMD, asmf64.L2NormInc)
		var result float64
		return scalarRun(func() { result = fn(x, uintptr(n), uintptr(stride)) }, &result)
	case "f64.L2DistanceUnitary":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, asmf64.L2DistanceUnitarySIMD, asmf64.L2DistanceUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.ScalUnitary":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, asmf64.ScalUnitarySIMD, asmf64.ScalUnitary)
		return sliceRun(func() { fn(realScale, x) }, x)
	case "f64.ScalUnitaryTo":
		x, dst := f64Values(n, 0.2), make([]float64, n)
		fn := choose(useSIMD, asmf64.ScalUnitaryToSIMD, asmf64.ScalUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x) }, dst)
	case "f64.ScalInc":
		x := f64Values(stride*n, 0.2)
		fn := choose(useSIMD, asmf64.ScalIncSIMD, asmf64.ScalInc)
		return sliceRun(func() { fn(realScale, x, uintptr(n), uintptr(stride)) }, x)
	case "f64.ScalIncTo":
		x, dst := f64Values(stride*n, 0.2), make([]float64, stride*n)
		fn := choose(useSIMD, asmf64.ScalIncToSIMD, asmf64.ScalIncTo)
		return sliceRun(func() { fn(dst, uintptr(stride), 0.75, x, uintptr(n), uintptr(stride)) }, dst)
	case "f64.Sum":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, asmf64.SumSIMD, asmf64.Sum)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.Ger":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		a := f64Values(n*n, 0.1)
		fn := choose(useSIMD, asmf64.GerSIMD, asmf64.Ger)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, x, 1, y, 1, a, uintptr(n)) }, a)
	case "f64.GemvN":
		a, x, y := f64Values(n*n, 0.1), f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, asmf64.GemvNSIMD, asmf64.GemvN)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, a, uintptr(n), x, 1, 0.25, y, 1) }, y)
	case "f64.GemvT":
		a, x, y := f64Values(n*n, 0.1), f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, asmf64.GemvTSIMD, asmf64.GemvT)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, a, uintptr(n), x, 1, 0.25, y, 1) }, y)
	case "f32.AxpyUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, asmf32.AxpyUnitarySIMD, asmf32.AxpyUnitary)
		return sliceRun(func() { fn(0.75, x, y) }, y)
	case "f32.AxpyUnitaryTo":
		x, y, dst := f32Values(n, 0.2), f32Values(n, 0.7), make([]float32, n)
		fn := choose(useSIMD, asmf32.AxpyUnitaryToSIMD, asmf32.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x, y) }, dst)
	case "f32.AxpyInc":
		x, y := f32Values(stride*n, 0.2), f32Values(stride*n, 0.7)
		fn := choose(useSIMD, asmf32.AxpyIncSIMD, asmf32.AxpyInc)
		return sliceRun(func() { fn(0.75, x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, y)
	case "f32.AxpyIncTo":
		x, y, dst := f32Values(stride*n, 0.2), f32Values(stride*n, 0.7), make([]float32, stride*n)
		fn := choose(useSIMD, asmf32.AxpyIncToSIMD, asmf32.AxpyIncTo)
		return sliceRun(func() { fn(dst, uintptr(stride), 0, 0.75, x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, dst)
	case "f32.DotUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, asmf32.DotUnitarySIMD, asmf32.DotUnitary)
		var result float32
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f32.DotInc":
		x, y := f32Values(stride*n, 0.2), f32Values(stride*n, 0.7)
		fn := choose(useSIMD, asmf32.DotIncSIMD, asmf32.DotInc)
		var result float32
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "f32.DdotUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, asmf32.DdotUnitarySIMD, asmf32.DdotUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f32.DdotInc":
		x, y := f32Values(stride*n, 0.2), f32Values(stride*n, 0.7)
		fn := choose(useSIMD, asmf32.DdotIncSIMD, asmf32.DdotInc)
		var result float64
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "f32.Sum":
		x := f32Values(n, 0.2)
		fn := choose(useSIMD, asmf32.SumSIMD, asmf32.Sum)
		var result float32
		return scalarRun(func() { result = fn(x) }, &result)
	case "f32.Ger":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		a := f32Values(n*n, 0.1)
		fn := choose(useSIMD, asmf32.GerSIMD, asmf32.Ger)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, x, 1, y, 1, a, uintptr(n)) }, a)
	case "c64.AxpyUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, asmc64.AxpyUnitarySIMD, asmc64.AxpyUnitary)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y) }, y)
	case "c64.AxpyUnitaryTo":
		x, y, dst := c64Values(n, 0.2), c64Values(n, 0.7), make([]complex64, n)
		fn := choose(useSIMD, asmc64.AxpyUnitaryToSIMD, asmc64.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, complex(0.75, -0.25), x, y) }, dst)
	case "c64.AxpyInc":
		x, y := c64Values(stride*n, 0.2), c64Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc64.AxpyIncSIMD, asmc64.AxpyInc)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, y)
	case "c64.AxpyIncTo":
		x, y, dst := c64Values(stride*n, 0.2), c64Values(stride*n, 0.7), make([]complex64, stride*n)
		fn := choose(useSIMD, asmc64.AxpyIncToSIMD, asmc64.AxpyIncTo)
		return sliceRun(func() {
			fn(dst, uintptr(stride), 0, complex(0.75, -0.25), x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0)
		}, dst)
	case "c64.DotcUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, asmc64.DotcUnitarySIMD, asmc64.DotcUnitary)
		var result complex64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c64.DotuUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, asmc64.DotuUnitarySIMD, asmc64.DotuUnitary)
		var result complex64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c64.DotcInc":
		x, y := c64Values(stride*n, 0.2), c64Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc64.DotcIncSIMD, asmc64.DotcInc)
		var result complex64
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "c64.DotuInc":
		x, y := c64Values(stride*n, 0.2), c64Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc64.DotuIncSIMD, asmc64.DotuInc)
		var result complex64
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "c128.AxpyUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, asmc128.AxpyUnitarySIMD, asmc128.AxpyUnitary)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y) }, y)
	case "c128.AxpyUnitaryTo":
		x, y, dst := c128Values(n, 0.2), c128Values(n, 0.7), make([]complex128, n)
		fn := choose(useSIMD, asmc128.AxpyUnitaryToSIMD, asmc128.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, complex(0.75, -0.25), x, y) }, dst)
	case "c128.AxpyInc":
		x, y := c128Values(stride*n, 0.2), c128Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc128.AxpyIncSIMD, asmc128.AxpyInc)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, y)
	case "c128.AxpyIncTo":
		x, y, dst := c128Values(stride*n, 0.2), c128Values(stride*n, 0.7), make([]complex128, stride*n)
		fn := choose(useSIMD, asmc128.AxpyIncToSIMD, asmc128.AxpyIncTo)
		return sliceRun(func() {
			fn(dst, uintptr(stride), 0, complex(0.75, -0.25), x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0)
		}, dst)
	case "c128.DscalUnitary":
		x := c128Values(n, 0.2)
		fn := choose(useSIMD, asmc128.DscalUnitarySIMD, asmc128.DscalUnitary)
		return sliceRun(func() { fn(realScale, x) }, x)
	case "c128.DscalInc":
		x := c128Values(stride*n, 0.2)
		fn := choose(useSIMD, asmc128.DscalIncSIMD, asmc128.DscalInc)
		return sliceRun(func() { fn(realScale, x, uintptr(n), uintptr(stride)) }, x)
	case "c128.ScalUnitary":
		x := c128Values(n, 0.2)
		fn := choose(useSIMD, asmc128.ScalUnitarySIMD, asmc128.ScalUnitary)
		return sliceRun(func() { fn(complexScale, x) }, x)
	case "c128.ScalInc":
		x := c128Values(stride*n, 0.2)
		fn := choose(useSIMD, asmc128.ScalIncSIMD, asmc128.ScalInc)
		return sliceRun(func() { fn(complexScale, x, uintptr(n), uintptr(stride)) }, x)
	case "c128.DotcUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, asmc128.DotcUnitarySIMD, asmc128.DotcUnitary)
		var result complex128
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c128.DotuUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, asmc128.DotuUnitarySIMD, asmc128.DotuUnitary)
		var result complex128
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c128.DotcInc":
		x, y := c128Values(stride*n, 0.2), c128Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc128.DotcIncSIMD, asmc128.DotcInc)
		var result complex128
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	case "c128.DotuInc":
		x, y := c128Values(stride*n, 0.2), c128Values(stride*n, 0.7)
		fn := choose(useSIMD, asmc128.DotuIncSIMD, asmc128.DotuInc)
		var result complex128
		return scalarRun(func() { result = fn(x, y, uintptr(n), uintptr(stride), uintptr(stride), 0, 0) }, &result)
	default:
		panic("missing benchmark and equivalence runner for " + key)
	}
}

func sliceRun[T any](run func(), result []T) kernelRun {
	return kernelRun{run: run, result: func() any { return result }}
}

func scalarRun[T any](run func(), result *T) kernelRun {
	return kernelRun{run: run, result: func() any { return *result }}
}

func f64Values(n int, offset float64) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = offset + float64(i%17+1)/19
		if i%3 == 0 {
			values[i] = -values[i]
		}
	}
	return values
}

func f32Values(n int, offset float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = offset + float32(i%17+1)/19
		if i%3 == 0 {
			values[i] = -values[i]
		}
	}
	return values
}

func c64Values(n int, offset float32) []complex64 {
	realPart := f32Values(n, offset)
	imagPart := f32Values(n, offset+0.3)
	values := make([]complex64, n)
	for i := range values {
		values[i] = complex(realPart[i], imagPart[n-1-i])
	}
	return values
}

func c128Values(n int, offset float64) []complex128 {
	realPart := f64Values(n, offset)
	imagPart := f64Values(n, offset+0.3)
	values := make([]complex128, n)
	for i := range values {
		values[i] = complex(realPart[i], imagPart[n-1-i])
	}
	return values
}

func sameKernelResult(a, b any) bool {
	switch a := a.(type) {
	case float32:
		return closeFloat(float64(a), float64(b.(float32)), 2e-5)
	case float64:
		return closeFloat(a, b.(float64), 2e-12)
	case complex64:
		value := b.(complex64)
		return closeFloat(float64(real(a)), float64(real(value)), 2e-5) && closeFloat(float64(imag(a)), float64(imag(value)), 2e-5)
	case complex128:
		value := b.(complex128)
		return closeFloat(real(a), real(value), 2e-12) && closeFloat(imag(a), imag(value), 2e-12)
	case []float32:
		value := b.([]float32)
		for i := range a {
			if !closeFloat(float64(a[i]), float64(value[i]), 2e-5) {
				return false
			}
		}
		return true
	case []float64:
		value := b.([]float64)
		for i := range a {
			if !closeFloat(a[i], value[i], 2e-12) {
				return false
			}
		}
		return true
	case []complex64:
		value := b.([]complex64)
		for i := range a {
			if !sameKernelResult(a[i], value[i]) {
				return false
			}
		}
		return true
	case []complex128:
		value := b.([]complex128)
		for i := range a {
			if !sameKernelResult(a[i], value[i]) {
				return false
			}
		}
		return true
	default:
		panic(fmt.Sprintf("unsupported result type %T", a))
	}
}

func closeFloat(a, b, tolerance float64) bool {
	if a == b {
		return true
	}
	return math.Abs(a-b) <= tolerance*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
