// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && linux && simdasmstate && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"runtime"
	"simd/archsimd"
	"syscall"
	"testing"
	"unsafe"

	asmc128 "plantcontrol.org/v1/gonum/internal/asm/c128"
	asmf64 "plantcontrol.org/v1/gonum/internal/asm/f64"
)

var asmStateSource = [4]float64{1, 2, 3, 4}
var asmStateSink [4]float64

// Keep the nonzero AVX load/store visible for objdump. This diagnostic helper
// intentionally returns with dirty upper vector state; it is not a kernel.
//
//go:noinline
func dirtyASMState() {
	archsimd.LoadFloat64x4Array(&asmStateSource).StoreArray(&asmStateSink)
}

// BenchmarkASMState diagnoses instruction-state effects on unchanged leaves.
// "native" is the ordinary closure call; the other states are explicit
// interventions, not replacement baseline measurements. Select only the cases
// needed for an investigation, and inspect generated code after toolchain changes.
func BenchmarkASMState(b *testing.B) {
	if !archsimd.X86.AVX() {
		b.Skip("AVX is required for the instruction-state diagnostic")
	}
	for _, entry := range []Entry{{Package: "f64", Symbol: "AddConst"}, {Package: "f64", Symbol: "L2NormInc"}, {Package: "c128", Symbol: "DotuInc"}} {
		for _, n := range []int{31, 4096} {
			for _, offset := range []int{0, 1} {
				for _, state := range []string{"native", "clean-once", "dirty-once", "clean-call", "dirty-call"} {
					for _, useSIMD := range []bool{false, true} {
						implementation := "current"
						if useSIMD {
							implementation = "simd"
						}
						b.Run(fmt.Sprintf("%s/%s/n=%d/offset=%d/state=%s/implementation=%s", entry.Package, entry.Symbol, n, offset, state, implementation), func(b *testing.B) {
							runner, alignment := newASMStateRun(entry, n, offset, useSIMD)
							var before, after runtime.MemStats
							var usageBefore, usageAfter syscall.Rusage
							runtime.ReadMemStats(&before)
							if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usageBefore); err != nil {
								b.Fatal(err)
							}
							b.ReportAllocs()
							b.ResetTimer()
							switch state {
							case "native", "clean-once", "dirty-once":
								if state == "clean-once" {
									archsimd.ClearAVXUpperBits()
								} else if state == "dirty-once" {
									dirtyASMState()
								}
								for i := 0; i < b.N; i++ {
									runner.run()
								}
							case "clean-call":
								for i := 0; i < b.N; i++ {
									archsimd.ClearAVXUpperBits()
									runner.run()
								}
							case "dirty-call":
								for i := 0; i < b.N; i++ {
									dirtyASMState()
									runner.run()
								}
							}
							b.StopTimer()
							archsimd.ClearAVXUpperBits()
							if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usageAfter); err != nil {
								b.Fatal(err)
							}
							runtime.ReadMemStats(&after)
							benchmarkSink = runner.result()
							b.ReportMetric(float64(alignment), "x-address-mod64")
							b.ReportMetric(float64(after.NumGC-before.NumGC), "GCs/trial")
							b.ReportMetric(float64(after.PauseTotalNs-before.PauseTotalNs), "GC-pause-ns/trial")
							b.ReportMetric(float64(usageAfter.Nivcsw-usageBefore.Nivcsw), "involuntary-switches/trial")
							b.ReportMetric(float64(usageAfter.Nvcsw-usageBefore.Nvcsw), "voluntary-switches/trial")
							b.ReportMetric(float64(usageAfter.Minflt-usageBefore.Minflt), "minor-faults/trial")
							userNS := (usageAfter.Utime.Sec-usageBefore.Utime.Sec)*1e9 + (usageAfter.Utime.Usec-usageBefore.Utime.Usec)*1e3
							b.ReportMetric(float64(userNS)/float64(b.N), "user-ns/op")
						})
					}
				}
			}
		}
	}
}

// Use the original fixtures and closure call shape. Offset changes alignment,
// not the logical data, stride, arithmetic, or original assembly implementation.
func newASMStateRun(entry Entry, n, offset int, useSIMD bool) (kernelRun, uintptr) {
	switch entry.Package + "." + entry.Symbol {
	case "f64.AddConst":
		x := make([]float64, offset+n)[offset:]
		copy(x, f64Values(n, 0.2))
		fn := choose(useSIMD, asmf64.AddConstSIMD, asmf64.AddConst)
		return sliceRun(func() { fn(0.75, x) }, x), uintptr(unsafe.Pointer(&x[0])) % 64
	case "f64.L2NormInc":
		x := make([]float64, offset+2*n)[offset:]
		copy(x, f64Values(2*n, -0.5))
		fn := choose(useSIMD, asmf64.L2NormIncSIMD, asmf64.L2NormInc)
		var result float64
		return scalarRun(func() { result = fn(x, uintptr(n), 2) }, &result), uintptr(unsafe.Pointer(&x[0])) % 64
	case "c128.DotuInc":
		x, y := make([]complex128, offset+2*n)[offset:], make([]complex128, offset+2*n)[offset:]
		copy(x, c128Values(2*n, 0.2))
		copy(y, c128Values(2*n, 0.7))
		fn := choose(useSIMD, asmc128.DotuIncSIMD, asmc128.DotuInc)
		var result complex128
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result), uintptr(unsafe.Pointer(&x[0])) % 64
	default:
		panic("missing ASM instruction-state diagnostic")
	}
}

func TestASMStateFixtures(t *testing.T) {
	if !archsimd.X86.AVX() {
		t.Skip("AVX is required for the instruction-state diagnostic")
	}
	for _, entry := range []Entry{{Package: "f64", Symbol: "AddConst"}, {Package: "f64", Symbol: "L2NormInc"}, {Package: "c128", Symbol: "DotuInc"}} {
		for _, n := range []int{31, 4096} {
			for _, offset := range []int{0, 1} {
				for _, useSIMD := range []bool{false, true} {
					want := newKernelRun(entry, n, useSIMD, true)
					got, _ := newASMStateRun(entry, n, offset, useSIMD)
					archsimd.ClearAVXUpperBits()
					want.run()
					dirtyASMState()
					got.run()
					archsimd.ClearAVXUpperBits()
					if !sameKernelResult(got.result(), want.result()) {
						t.Fatalf("%s/%s/n=%d/offset=%d/simd=%t: diagnostic changed fixture result", entry.Package, entry.Symbol, n, offset, useSIMD)
					}
				}
			}
		}
	}
}
