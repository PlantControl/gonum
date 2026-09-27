// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

type level1Oracle interface {
	blas.Float32Level1
	blas.Float64Level1
	blas.Complex64Level1
	blas.Complex128Level1
}

var level1BenchmarkSink complex128

func level1BenchmarkNames() []string {
	return []string{"Sdsdot", "Dsdot", "Sdot", "Snrm2", "Sasum", "Isamax", "Sswap", "Scopy", "Saxpy", "Srotg", "Srotmg", "Srot", "Srotm", "Sscal", "Ddot", "Dnrm2", "Dasum", "Idamax", "Dswap", "Dcopy", "Daxpy", "Drotg", "Drotmg", "Drot", "Drotm", "Dscal", "Cdotu", "Cdotc", "Scnrm2", "Scasum", "Icamax", "Cswap", "Ccopy", "Caxpy", "Cscal", "Csscal", "Zdotu", "Zdotc", "Dznrm2", "Dzasum", "Izamax", "Zswap", "Zcopy", "Zaxpy", "Zscal", "Zdscal"}
}

func BenchmarkLevel1Netlib(b *testing.B) {
	for _, routine := range level1BenchmarkNames() {
		sizes := []int{16, 256, 4096, 65536}
		strides := []int{1, 2}
		if level1BoundaryBenchmark(routine) {
			sizes = append([]int{4, 31, 32, 33}, sizes...)
		}
		if strings.HasSuffix(routine, "rotg") || strings.HasSuffix(routine, "rotmg") {
			sizes, strides = []int{1}, []int{1}
		}
		for _, n := range sizes {
			for _, inc := range strides {
				for _, backend := range []string{"Gonum", "Netlib"} {
					name := fmt.Sprintf("routine=%s/n=%d/inc=%d/implementation=%s", routine, n, inc, backend)
					b.Run(name, func(b *testing.B) {
						repeat := 32
						if n >= 65536 {
							repeat = 4
						}
						var impl level1Oracle = Implementation{}
						if backend == "Netlib" {
							impl = netlib.Implementation{Repeat: repeat}
						}
						call, valid := level1BenchmarkCall(routine, n, inc, impl)
						var sum complex128
						b.ReportAllocs()
						b.ResetTimer()
						if backend == "Netlib" {
							for i := 0; i < b.N; i++ {
								sum += call(1)
							}
						} else {
							for i := 0; i < b.N; i++ {
								for j := 0; j < repeat; j++ {
									sign := 1.0
									if j&1 != 0 {
										sign = -1
									}
									sum += call(sign)
								}
							}
						}
						b.StopTimer()
						// ns/op is per BLAS call; allocation counters remain per outer batch.
						b.ReportMetric(float64(b.Elapsed().Nanoseconds())/(float64(b.N)*float64(repeat)), "ns/op")
						b.ReportMetric(float64(repeat), "calls/batch")
						level1BenchmarkSink = sum
						if !valid() || !finiteLevel1(real(sum)) || !finiteLevel1(imag(sum)) {
							b.Fatal("non-finite benchmark state")
						}
					})
				}
			}
		}
	}
}

func level1BoundaryBenchmark(routine string) bool {
	switch routine {
	case "Isamax", "Idamax", "Scasum", "Dzasum", "Dznrm2", "Snrm2", "Scnrm2":
		return true
	default:
		return false
	}
}

func finiteLevel1(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func level1BenchmarkCall(routine string, n, inc int, impl level1Oracle) (func(float64) complex128, func() bool) {
	size := max(0, (n-1)*inc+1)
	xs, ys := make([]float32, size), make([]float32, size)
	xd, yd := make([]float64, size), make([]float64, size)
	xc, yc := make([]complex64, size), make([]complex64, size)
	xz, yz := make([]complex128, size), make([]complex128, size)
	rnd := rand.New(rand.NewPCG(1, 2))
	for i := 0; i < size; i++ {
		x, y := rnd.Float64()*2-1, rnd.Float64()*2-1
		xd[i], yd[i], xs[i], ys[i] = x, y, float32(x), float32(y)
		xz[i], yz[i] = complex(x, rnd.Float64()*2-1), complex(y, rnd.Float64()*2-1)
		xc[i], yc[i] = complex64(xz[i]), complex64(yz[i])
	}
	valid := func() bool {
		for i := 0; i < size; i++ {
			for _, v := range []float64{float64(xs[i]), float64(ys[i]), xd[i], yd[i], float64(real(xc[i])), float64(imag(xc[i])), float64(real(yc[i])), float64(imag(yc[i])), real(xz[i]), imag(xz[i]), real(yz[i]), imag(yz[i])} {
				if !finiteLevel1(v) {
					return false
				}
			}
		}
		return true
	}
	var call func(float64) complex128
	switch routine {
	case "Sdsdot":
		call = func(sign float64) complex128 { return complex(float64(impl.Sdsdot(n, 0.25, xs, inc, ys, inc)), 0) }
	case "Dsdot":
		call = func(sign float64) complex128 { return complex(impl.Dsdot(n, xs, inc, ys, inc), 0) }
	case "Sdot":
		call = func(sign float64) complex128 { return complex(float64(impl.Sdot(n, xs, inc, ys, inc)), 0) }
	case "Snrm2":
		call = func(sign float64) complex128 { return complex(float64(impl.Snrm2(n, xs, inc)), 0) }
	case "Sasum":
		call = func(sign float64) complex128 { return complex(float64(impl.Sasum(n, xs, inc)), 0) }
	case "Isamax":
		call = func(sign float64) complex128 { return complex(float64(impl.Isamax(n, xs, inc)), 0) }
	case "Sswap":
		call = func(sign float64) complex128 { impl.Sswap(n, xs, inc, ys, inc); return 0 }
	case "Scopy":
		call = func(sign float64) complex128 { impl.Scopy(n, xs, inc, ys, inc); return 0 }
	case "Saxpy":
		call = func(sign float64) complex128 { impl.Saxpy(n, float32(sign*0.25), xs, inc, ys, inc); return 0 }
	case "Srotg":
		call = func(sign float64) complex128 { c, s, r, z := impl.Srotg(3, 4); return complex(float64(c+s+r+z), 0) }
	case "Srotmg":
		call = func(sign float64) complex128 {
			p, d1, d2, x1 := impl.Srotmg(2, 3, 4, 5)
			return complex(float64(p.Flag)+float64(d1+d2+x1), 0)
		}
	case "Srot":
		call = func(sign float64) complex128 { impl.Srot(n, xs, inc, ys, inc, 0, 1); return 0 }
	case "Srotm":
		call = func(sign float64) complex128 {
			impl.Srotm(n, xs, inc, ys, inc, blas.SrotmParams{Flag: blas.Rescaling, H: [4]float32{0, -1, 1, 0}})
			return 0
		}
	case "Sscal":
		call = func(sign float64) complex128 { impl.Sscal(n, -1, xs, inc); return 0 }
	case "Ddot":
		call = func(sign float64) complex128 { return complex(float64(impl.Ddot(n, xd, inc, yd, inc)), 0) }
	case "Dnrm2":
		call = func(sign float64) complex128 { return complex(float64(impl.Dnrm2(n, xd, inc)), 0) }
	case "Dasum":
		call = func(sign float64) complex128 { return complex(float64(impl.Dasum(n, xd, inc)), 0) }
	case "Idamax":
		call = func(sign float64) complex128 { return complex(float64(impl.Idamax(n, xd, inc)), 0) }
	case "Dswap":
		call = func(sign float64) complex128 { impl.Dswap(n, xd, inc, yd, inc); return 0 }
	case "Dcopy":
		call = func(sign float64) complex128 { impl.Dcopy(n, xd, inc, yd, inc); return 0 }
	case "Daxpy":
		call = func(sign float64) complex128 { impl.Daxpy(n, float64(sign*0.25), xd, inc, yd, inc); return 0 }
	case "Drotg":
		call = func(sign float64) complex128 { c, s, r, z := impl.Drotg(3, 4); return complex(float64(c+s+r+z), 0) }
	case "Drotmg":
		call = func(sign float64) complex128 {
			p, d1, d2, x1 := impl.Drotmg(2, 3, 4, 5)
			return complex(float64(p.Flag)+float64(d1+d2+x1), 0)
		}
	case "Drot":
		call = func(sign float64) complex128 { impl.Drot(n, xd, inc, yd, inc, 0, 1); return 0 }
	case "Drotm":
		call = func(sign float64) complex128 {
			impl.Drotm(n, xd, inc, yd, inc, blas.DrotmParams{Flag: blas.Rescaling, H: [4]float64{0, -1, 1, 0}})
			return 0
		}
	case "Dscal":
		call = func(sign float64) complex128 { impl.Dscal(n, -1, xd, inc); return 0 }
	case "Cdotu":
		call = func(sign float64) complex128 { return complex128(impl.Cdotu(n, xc, inc, yc, inc)) }
	case "Cdotc":
		call = func(sign float64) complex128 { return complex128(impl.Cdotc(n, xc, inc, yc, inc)) }
	case "Scnrm2":
		call = func(sign float64) complex128 { return complex(float64(impl.Scnrm2(n, xc, inc)), 0) }
	case "Scasum":
		call = func(sign float64) complex128 { return complex(float64(impl.Scasum(n, xc, inc)), 0) }
	case "Icamax":
		call = func(sign float64) complex128 { return complex(float64(impl.Icamax(n, xc, inc)), 0) }
	case "Cswap":
		call = func(sign float64) complex128 { impl.Cswap(n, xc, inc, yc, inc); return 0 }
	case "Ccopy":
		call = func(sign float64) complex128 { impl.Ccopy(n, xc, inc, yc, inc); return 0 }
	case "Caxpy":
		call = func(sign float64) complex128 {
			impl.Caxpy(n, complex64(complex(sign*0.25, sign*0.125)), xc, inc, yc, inc)
			return 0
		}
	case "Cscal":
		call = func(sign float64) complex128 { impl.Cscal(n, 1i, xc, inc); return 0 }
	case "Csscal":
		call = func(sign float64) complex128 { impl.Csscal(n, -1, xc, inc); return 0 }
	case "Zdotu":
		call = func(sign float64) complex128 { return complex128(impl.Zdotu(n, xz, inc, yz, inc)) }
	case "Zdotc":
		call = func(sign float64) complex128 { return complex128(impl.Zdotc(n, xz, inc, yz, inc)) }
	case "Dznrm2":
		call = func(sign float64) complex128 { return complex(float64(impl.Dznrm2(n, xz, inc)), 0) }
	case "Dzasum":
		call = func(sign float64) complex128 { return complex(float64(impl.Dzasum(n, xz, inc)), 0) }
	case "Izamax":
		call = func(sign float64) complex128 { return complex(float64(impl.Izamax(n, xz, inc)), 0) }
	case "Zswap":
		call = func(sign float64) complex128 { impl.Zswap(n, xz, inc, yz, inc); return 0 }
	case "Zcopy":
		call = func(sign float64) complex128 { impl.Zcopy(n, xz, inc, yz, inc); return 0 }
	case "Zaxpy":
		call = func(sign float64) complex128 {
			impl.Zaxpy(n, complex128(complex(sign*0.25, sign*0.125)), xz, inc, yz, inc)
			return 0
		}
	case "Zscal":
		call = func(sign float64) complex128 { impl.Zscal(n, 1i, xz, inc); return 0 }
	case "Zdscal":
		call = func(sign float64) complex128 { impl.Zdscal(n, -1, xz, inc); return 0 }
	default:
		panic("unknown Level 1 routine")
	}
	return call, valid
}
