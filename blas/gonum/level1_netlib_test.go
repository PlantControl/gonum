// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

var (
	_ blas.Float32Level1    = netlib.Implementation{}
	_ blas.Float64Level1    = netlib.Implementation{}
	_ blas.Complex64Level1  = netlib.Implementation{}
	_ blas.Complex128Level1 = netlib.Implementation{}
)

func TestLevel1NetlibManifest(t *testing.T) {
	want := []string{
		"Sdsdot", "Dsdot", "Sdot", "Snrm2", "Sasum", "Isamax", "Sswap", "Scopy", "Saxpy", "Srotg", "Srotmg", "Srot", "Srotm", "Sscal",
		"Ddot", "Dnrm2", "Dasum", "Idamax", "Dswap", "Dcopy", "Daxpy", "Drotg", "Drotmg", "Drot", "Drotm", "Dscal",
		"Cdotu", "Cdotc", "Scnrm2", "Scasum", "Icamax", "Cswap", "Ccopy", "Caxpy", "Cscal", "Csscal",
		"Zdotu", "Zdotc", "Dznrm2", "Dzasum", "Izamax", "Zswap", "Zcopy", "Zaxpy", "Zscal", "Zdscal",
	}
	var got []string
	for _, typ := range []reflect.Type{
		reflect.TypeOf((*blas.Float32Level1)(nil)).Elem(),
		reflect.TypeOf((*blas.Float64Level1)(nil)).Elem(),
		reflect.TypeOf((*blas.Complex64Level1)(nil)).Elem(),
		reflect.TypeOf((*blas.Complex128Level1)(nil)).Elem(),
	} {
		for i := 0; i < typ.NumMethod(); i++ {
			got = append(got, typ.Method(i).Name)
		}
	}
	sort.Strings(got)
	sort.Strings(want)
	if !slices.Equal(got, want) {
		t.Fatalf("Level-1 manifest mismatch:\ngot  %v\nwant %v", got, want)
	}
	bench := level1BenchmarkNames()
	sort.Strings(bench)
	if !slices.Equal(bench, want) {
		t.Fatalf("Level-1 benchmark manifest mismatch:\ngot  %v\nwant %v", bench, want)
	}
}

func TestLevel1NetlibDifferential(t *testing.T) {
	g, n := Implementation{}, netlib.Implementation{}
	xs := []float32{1, -5, 5, 2, -3, 91}
	ys := []float32{-2, 3, -4, 5, -6, 92}
	xd := []float64{1, -5, 5, 2, -3, 91}
	yd := []float64{-2, 3, -4, 5, -6, 92}
	xc := []complex64{1 + 2i, -5 + 1i, 3 - 4i, 2 + 3i, -3 - 2i, 91 + 92i}
	yc := []complex64{-2 + 1i, 3 - 2i, -4 + 5i, 5 + 1i, -6 - 3i, 93 + 94i}
	xz := []complex128{1 + 2i, -5 + 1i, 3 - 4i, 2 + 3i, -3 - 2i, 91 + 92i}
	yz := []complex128{-2 + 1i, 3 - 2i, -4 + 5i, 5 + 1i, -6 - 3i, 93 + 94i}
	const count = 5

	check32(t, "Sdsdot", g.Sdsdot(count, 1.25, xs, 1, ys, 1), n.Sdsdot(count, 1.25, xs, 1, ys, 1))
	check64(t, "Dsdot", g.Dsdot(count, xs, 1, ys, 1), n.Dsdot(count, xs, 1, ys, 1))
	check32(t, "Sdot", g.Sdot(count, xs, 1, ys, 1), n.Sdot(count, xs, 1, ys, 1))
	check32(t, "Snrm2", g.Snrm2(count, xs, 1), n.Snrm2(count, xs, 1))
	check32(t, "Sasum", g.Sasum(count, xs, 1), n.Sasum(count, xs, 1))
	checkInt(t, "Isamax", g.Isamax(count, xs, 1), n.Isamax(count, xs, 1))
	compareMut32(t, "Sswap", xs, ys, func(x, y []float32) { g.Sswap(count, x, 1, y, 1) }, func(x, y []float32) { n.Sswap(count, x, 1, y, 1) })
	compareMut32(t, "Scopy", xs, ys, func(x, y []float32) { g.Scopy(count, x, 1, y, 1) }, func(x, y []float32) { n.Scopy(count, x, 1, y, 1) })
	compareMut32(t, "Saxpy", xs, ys, func(x, y []float32) { g.Saxpy(count, 0.5, x, 1, y, 1) }, func(x, y []float32) { n.Saxpy(count, 0.5, x, 1, y, 1) })
	gc, gs, gr, gz := g.Srotg(3, 4)
	nc, ns, nr, nz := n.Srotg(3, 4)
	check32s(t, "Srotg", []float32{gc, gs, gr, gz}, []float32{nc, ns, nr, nz})
	gsp, gd1, gd2, gb1 := g.Srotmg(2, 3, 4, 5)
	nsp, nd1, nd2, nb1 := n.Srotmg(2, 3, 4, 5)
	checkSrotmg(t, gsp, nsp)
	check32s(t, "Srotmg scalars", []float32{gd1, gd2, gb1}, []float32{nd1, nd2, nb1})
	compareMut32(t, "Srot", xs, ys, func(x, y []float32) { g.Srot(count, x, 1, y, 1, 0, 1) }, func(x, y []float32) { n.Srot(count, x, 1, y, 1, 0, 1) })
	sp := blas.SrotmParams{Flag: blas.Rescaling, H: [4]float32{0.5, -0.25, 0.25, 2}}
	compareMut32(t, "Srotm", xs, ys, func(x, y []float32) { g.Srotm(count, x, 1, y, 1, sp) }, func(x, y []float32) { n.Srotm(count, x, 1, y, 1, sp) })
	compareMut32(t, "Sscal", xs, nil, func(x, _ []float32) { g.Sscal(count, -1, x, 1) }, func(x, _ []float32) { n.Sscal(count, -1, x, 1) })

	check64(t, "Ddot", g.Ddot(count, xd, 1, yd, 1), n.Ddot(count, xd, 1, yd, 1))
	check64(t, "Dnrm2", g.Dnrm2(count, xd, 1), n.Dnrm2(count, xd, 1))
	check64(t, "Dasum", g.Dasum(count, xd, 1), n.Dasum(count, xd, 1))
	checkInt(t, "Idamax", g.Idamax(count, xd, 1), n.Idamax(count, xd, 1))
	compareMut64(t, "Dswap", xd, yd, func(x, y []float64) { g.Dswap(count, x, 1, y, 1) }, func(x, y []float64) { n.Dswap(count, x, 1, y, 1) })
	compareMut64(t, "Dcopy", xd, yd, func(x, y []float64) { g.Dcopy(count, x, 1, y, 1) }, func(x, y []float64) { n.Dcopy(count, x, 1, y, 1) })
	compareMut64(t, "Daxpy", xd, yd, func(x, y []float64) { g.Daxpy(count, 0.5, x, 1, y, 1) }, func(x, y []float64) { n.Daxpy(count, 0.5, x, 1, y, 1) })
	dgc, dgs, dgr, dgz := g.Drotg(3, 4)
	dnc, dns, dnr, dnz := n.Drotg(3, 4)
	check64s(t, "Drotg", []float64{dgc, dgs, dgr, dgz}, []float64{dnc, dns, dnr, dnz})
	gdp, gdd1, gdd2, gdb1 := g.Drotmg(2, 3, 4, 5)
	ndp, ndd1, ndd2, ndb1 := n.Drotmg(2, 3, 4, 5)
	checkDrotmg(t, gdp, ndp)
	check64s(t, "Drotmg scalars", []float64{gdd1, gdd2, gdb1}, []float64{ndd1, ndd2, ndb1})
	compareMut64(t, "Drot", xd, yd, func(x, y []float64) { g.Drot(count, x, 1, y, 1, 0, 1) }, func(x, y []float64) { n.Drot(count, x, 1, y, 1, 0, 1) })
	dp := blas.DrotmParams{Flag: blas.Rescaling, H: [4]float64{0.5, -0.25, 0.25, 2}}
	compareMut64(t, "Drotm", xd, yd, func(x, y []float64) { g.Drotm(count, x, 1, y, 1, dp) }, func(x, y []float64) { n.Drotm(count, x, 1, y, 1, dp) })
	compareMut64(t, "Dscal", xd, nil, func(x, _ []float64) { g.Dscal(count, -1, x, 1) }, func(x, _ []float64) { n.Dscal(count, -1, x, 1) })

	checkC64(t, "Cdotu", g.Cdotu(count, xc, 1, yc, 1), n.Cdotu(count, xc, 1, yc, 1))
	checkC64(t, "Cdotc", g.Cdotc(count, xc, 1, yc, 1), n.Cdotc(count, xc, 1, yc, 1))
	check32(t, "Scnrm2", g.Scnrm2(count, xc, 1), n.Scnrm2(count, xc, 1))
	check32(t, "Scasum", g.Scasum(count, xc, 1), n.Scasum(count, xc, 1))
	checkInt(t, "Icamax", g.Icamax(count, xc, 1), n.Icamax(count, xc, 1))
	compareMutC64(t, "Cswap", xc, yc, func(x, y []complex64) { g.Cswap(count, x, 1, y, 1) }, func(x, y []complex64) { n.Cswap(count, x, 1, y, 1) })
	compareMutC64(t, "Ccopy", xc, yc, func(x, y []complex64) { g.Ccopy(count, x, 1, y, 1) }, func(x, y []complex64) { n.Ccopy(count, x, 1, y, 1) })
	compareMutC64(t, "Caxpy", xc, yc, func(x, y []complex64) { g.Caxpy(count, 0.5+0.25i, x, 1, y, 1) }, func(x, y []complex64) { n.Caxpy(count, 0.5+0.25i, x, 1, y, 1) })
	compareMutC64(t, "Cscal", xc, nil, func(x, _ []complex64) { g.Cscal(count, 0+1i, x, 1) }, func(x, _ []complex64) { n.Cscal(count, 0+1i, x, 1) })
	compareMutC64(t, "Csscal", xc, nil, func(x, _ []complex64) { g.Csscal(count, -1, x, 1) }, func(x, _ []complex64) { n.Csscal(count, -1, x, 1) })

	checkC128(t, "Zdotu", g.Zdotu(count, xz, 1, yz, 1), n.Zdotu(count, xz, 1, yz, 1))
	checkC128(t, "Zdotc", g.Zdotc(count, xz, 1, yz, 1), n.Zdotc(count, xz, 1, yz, 1))
	check64(t, "Dznrm2", g.Dznrm2(count, xz, 1), n.Dznrm2(count, xz, 1))
	check64(t, "Dzasum", g.Dzasum(count, xz, 1), n.Dzasum(count, xz, 1))
	checkInt(t, "Izamax", g.Izamax(count, xz, 1), n.Izamax(count, xz, 1))
	compareMutC128(t, "Zswap", xz, yz, func(x, y []complex128) { g.Zswap(count, x, 1, y, 1) }, func(x, y []complex128) { n.Zswap(count, x, 1, y, 1) })
	compareMutC128(t, "Zcopy", xz, yz, func(x, y []complex128) { g.Zcopy(count, x, 1, y, 1) }, func(x, y []complex128) { n.Zcopy(count, x, 1, y, 1) })
	compareMutC128(t, "Zaxpy", xz, yz, func(x, y []complex128) { g.Zaxpy(count, 0.5+0.25i, x, 1, y, 1) }, func(x, y []complex128) { n.Zaxpy(count, 0.5+0.25i, x, 1, y, 1) })
	compareMutC128(t, "Zscal", xz, nil, func(x, _ []complex128) { g.Zscal(count, 0+1i, x, 1) }, func(x, _ []complex128) { n.Zscal(count, 0+1i, x, 1) })
	compareMutC128(t, "Zdscal", xz, nil, func(x, _ []complex128) { g.Zdscal(count, -1, x, 1) }, func(x, _ []complex128) { n.Zdscal(count, -1, x, 1) })
}

func TestLevel1NetlibVectorCases(t *testing.T) {
	for _, tc := range []struct {
		n, incX, incY int
	}{
		{0, 1, 1},
		{1, 1, 1},
		{5, 2, 3},
		{33, 1, 1},
		{5, -1, -2},
	} {
		name := fmt.Sprintf("n=%d/incX=%d/incY=%d", tc.n, tc.incX, tc.incY)
		t.Run(name, func(t *testing.T) {
			testLevel1Float32Case(t, tc.n, tc.incX, tc.incY)
			testLevel1Float64Case(t, tc.n, tc.incX, tc.incY)
			testLevel1Complex64Case(t, tc.n, tc.incX, tc.incY)
			testLevel1Complex128Case(t, tc.n, tc.incX, tc.incY)
		})
	}
}

func testLevel1Float32Case(t *testing.T, count, incX, incY int) {
	t.Helper()
	g, n := Implementation{}, netlib.Implementation{}
	x, y := level1Data32(count, incX, 1), level1Data32(count, incY, -2)
	check32(t, "Sdsdot", g.Sdsdot(count, 1.25, x, incX, y, incY), n.Sdsdot(count, 1.25, x, incX, y, incY))
	check64(t, "Dsdot", g.Dsdot(count, x, incX, y, incY), n.Dsdot(count, x, incX, y, incY))
	check32(t, "Sdot", g.Sdot(count, x, incX, y, incY), n.Sdot(count, x, incX, y, incY))
	compareMut32(t, "Sswap", x, y, func(x, y []float32) { g.Sswap(count, x, incX, y, incY) }, func(x, y []float32) { n.Sswap(count, x, incX, y, incY) })
	compareMut32(t, "Scopy", x, y, func(x, y []float32) { g.Scopy(count, x, incX, y, incY) }, func(x, y []float32) { n.Scopy(count, x, incX, y, incY) })
	compareMut32(t, "Saxpy", x, y, func(x, y []float32) { g.Saxpy(count, 0.5, x, incX, y, incY) }, func(x, y []float32) { n.Saxpy(count, 0.5, x, incX, y, incY) })
	compareMut32(t, "Srot", x, y, func(x, y []float32) { g.Srot(count, x, incX, y, incY, 0, 1) }, func(x, y []float32) { n.Srot(count, x, incX, y, incY, 0, 1) })
	p := blas.SrotmParams{Flag: blas.Rescaling, H: [4]float32{0.5, -0.25, 0.25, 2}}
	compareMut32(t, "Srotm", x, y, func(x, y []float32) { g.Srotm(count, x, incX, y, incY, p) }, func(x, y []float32) { n.Srotm(count, x, incX, y, incY, p) })
	if incX > 0 {
		check32(t, "Snrm2", g.Snrm2(count, x, incX), n.Snrm2(count, x, incX))
		check32(t, "Sasum", g.Sasum(count, x, incX), n.Sasum(count, x, incX))
		checkInt(t, "Isamax", g.Isamax(count, x, incX), n.Isamax(count, x, incX))
		compareMut32(t, "Sscal", x, nil, func(x, _ []float32) { g.Sscal(count, -1, x, incX) }, func(x, _ []float32) { n.Sscal(count, -1, x, incX) })
	}
}

func testLevel1Float64Case(t *testing.T, count, incX, incY int) {
	t.Helper()
	g, n := Implementation{}, netlib.Implementation{}
	x, y := level1Data64(count, incX, 1), level1Data64(count, incY, -2)
	check64(t, "Ddot", g.Ddot(count, x, incX, y, incY), n.Ddot(count, x, incX, y, incY))
	compareMut64(t, "Dswap", x, y, func(x, y []float64) { g.Dswap(count, x, incX, y, incY) }, func(x, y []float64) { n.Dswap(count, x, incX, y, incY) })
	compareMut64(t, "Dcopy", x, y, func(x, y []float64) { g.Dcopy(count, x, incX, y, incY) }, func(x, y []float64) { n.Dcopy(count, x, incX, y, incY) })
	compareMut64(t, "Daxpy", x, y, func(x, y []float64) { g.Daxpy(count, 0.5, x, incX, y, incY) }, func(x, y []float64) { n.Daxpy(count, 0.5, x, incX, y, incY) })
	compareMut64(t, "Drot", x, y, func(x, y []float64) { g.Drot(count, x, incX, y, incY, 0, 1) }, func(x, y []float64) { n.Drot(count, x, incX, y, incY, 0, 1) })
	p := blas.DrotmParams{Flag: blas.Rescaling, H: [4]float64{0.5, -0.25, 0.25, 2}}
	compareMut64(t, "Drotm", x, y, func(x, y []float64) { g.Drotm(count, x, incX, y, incY, p) }, func(x, y []float64) { n.Drotm(count, x, incX, y, incY, p) })
	if incX > 0 {
		check64(t, "Dnrm2", g.Dnrm2(count, x, incX), n.Dnrm2(count, x, incX))
		check64(t, "Dasum", g.Dasum(count, x, incX), n.Dasum(count, x, incX))
		checkInt(t, "Idamax", g.Idamax(count, x, incX), n.Idamax(count, x, incX))
		compareMut64(t, "Dscal", x, nil, func(x, _ []float64) { g.Dscal(count, -1, x, incX) }, func(x, _ []float64) { n.Dscal(count, -1, x, incX) })
	}
}

func testLevel1Complex64Case(t *testing.T, count, incX, incY int) {
	t.Helper()
	g, n := Implementation{}, netlib.Implementation{}
	x, y := level1DataC64(count, incX, 1), level1DataC64(count, incY, -2)
	checkC64(t, "Cdotu", g.Cdotu(count, x, incX, y, incY), n.Cdotu(count, x, incX, y, incY))
	checkC64(t, "Cdotc", g.Cdotc(count, x, incX, y, incY), n.Cdotc(count, x, incX, y, incY))
	compareMutC64(t, "Cswap", x, y, func(x, y []complex64) { g.Cswap(count, x, incX, y, incY) }, func(x, y []complex64) { n.Cswap(count, x, incX, y, incY) })
	compareMutC64(t, "Ccopy", x, y, func(x, y []complex64) { g.Ccopy(count, x, incX, y, incY) }, func(x, y []complex64) { n.Ccopy(count, x, incX, y, incY) })
	compareMutC64(t, "Caxpy", x, y, func(x, y []complex64) { g.Caxpy(count, 0.5+0.25i, x, incX, y, incY) }, func(x, y []complex64) { n.Caxpy(count, 0.5+0.25i, x, incX, y, incY) })
	if incX > 0 {
		check32(t, "Scnrm2", g.Scnrm2(count, x, incX), n.Scnrm2(count, x, incX))
		check32(t, "Scasum", g.Scasum(count, x, incX), n.Scasum(count, x, incX))
		checkInt(t, "Icamax", g.Icamax(count, x, incX), n.Icamax(count, x, incX))
		compareMutC64(t, "Cscal", x, nil, func(x, _ []complex64) { g.Cscal(count, 0+1i, x, incX) }, func(x, _ []complex64) { n.Cscal(count, 0+1i, x, incX) })
		compareMutC64(t, "Csscal", x, nil, func(x, _ []complex64) { g.Csscal(count, -1, x, incX) }, func(x, _ []complex64) { n.Csscal(count, -1, x, incX) })
	}
}

func testLevel1Complex128Case(t *testing.T, count, incX, incY int) {
	t.Helper()
	g, n := Implementation{}, netlib.Implementation{}
	x, y := level1DataC128(count, incX, 1), level1DataC128(count, incY, -2)
	checkC128(t, "Zdotu", g.Zdotu(count, x, incX, y, incY), n.Zdotu(count, x, incX, y, incY))
	checkC128(t, "Zdotc", g.Zdotc(count, x, incX, y, incY), n.Zdotc(count, x, incX, y, incY))
	compareMutC128(t, "Zswap", x, y, func(x, y []complex128) { g.Zswap(count, x, incX, y, incY) }, func(x, y []complex128) { n.Zswap(count, x, incX, y, incY) })
	compareMutC128(t, "Zcopy", x, y, func(x, y []complex128) { g.Zcopy(count, x, incX, y, incY) }, func(x, y []complex128) { n.Zcopy(count, x, incX, y, incY) })
	compareMutC128(t, "Zaxpy", x, y, func(x, y []complex128) { g.Zaxpy(count, 0.5+0.25i, x, incX, y, incY) }, func(x, y []complex128) { n.Zaxpy(count, 0.5+0.25i, x, incX, y, incY) })
	if incX > 0 {
		check64(t, "Dznrm2", g.Dznrm2(count, x, incX), n.Dznrm2(count, x, incX))
		check64(t, "Dzasum", g.Dzasum(count, x, incX), n.Dzasum(count, x, incX))
		checkInt(t, "Izamax", g.Izamax(count, x, incX), n.Izamax(count, x, incX))
		compareMutC128(t, "Zscal", x, nil, func(x, _ []complex128) { g.Zscal(count, 0+1i, x, incX) }, func(x, _ []complex128) { n.Zscal(count, 0+1i, x, incX) })
		compareMutC128(t, "Zdscal", x, nil, func(x, _ []complex128) { g.Zdscal(count, -1, x, incX) }, func(x, _ []complex128) { n.Zdscal(count, -1, x, incX) })
	}
}

func level1Len(n, inc int) int {
	if n <= 0 {
		return 3
	}
	if inc < 0 {
		inc = -inc
	}
	return 3 + (n-1)*inc
}

func level1Data32(n, inc int, shift float32) []float32 {
	x := make([]float32, level1Len(n, inc))
	for i := range x {
		x[i] = float32(i%11-5)/4 + shift
	}
	return x
}

func level1Data64(n, inc int, shift float64) []float64 {
	x := make([]float64, level1Len(n, inc))
	for i := range x {
		x[i] = float64(i%11-5)/4 + shift
	}
	return x
}

func level1DataC64(n, inc int, shift float32) []complex64 {
	x := make([]complex64, level1Len(n, inc))
	for i := range x {
		x[i] = complex(float32(i%11-5)/4+shift, float32(i%7-3)/5-shift)
	}
	return x
}

func level1DataC128(n, inc int, shift float64) []complex128 {
	x := make([]complex128, level1Len(n, inc))
	for i := range x {
		x[i] = complex(float64(i%11-5)/4+shift, float64(i%7-3)/5-shift)
	}
	return x
}

func compareMut32(t *testing.T, name string, x, y []float32, g, n func(x, y []float32)) {
	t.Helper()
	gx, nx, gy, ny := slices.Clone(x), slices.Clone(x), slices.Clone(y), slices.Clone(y)
	g(gx, gy)
	n(nx, ny)
	check32s(t, name+" x", gx, nx)
	check32s(t, name+" y", gy, ny)
}

func compareMut64(t *testing.T, name string, x, y []float64, g, n func(x, y []float64)) {
	t.Helper()
	gx, nx, gy, ny := slices.Clone(x), slices.Clone(x), slices.Clone(y), slices.Clone(y)
	g(gx, gy)
	n(nx, ny)
	check64s(t, name+" x", gx, nx)
	check64s(t, name+" y", gy, ny)
}

func compareMutC64(t *testing.T, name string, x, y []complex64, g, n func(x, y []complex64)) {
	t.Helper()
	gx, nx, gy, ny := slices.Clone(x), slices.Clone(x), slices.Clone(y), slices.Clone(y)
	g(gx, gy)
	n(nx, ny)
	checkC64s(t, name+" x", gx, nx)
	checkC64s(t, name+" y", gy, ny)
}

func compareMutC128(t *testing.T, name string, x, y []complex128, g, n func(x, y []complex128)) {
	t.Helper()
	gx, nx, gy, ny := slices.Clone(x), slices.Clone(x), slices.Clone(y), slices.Clone(y)
	g(gx, gy)
	n(nx, ny)
	checkC128s(t, name+" x", gx, nx)
	checkC128s(t, name+" y", gy, ny)
}

func check32(t *testing.T, name string, got, want float32) {
	t.Helper()
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(got-want)) > 2e-5*math.Max(1, math.Max(math.Abs(float64(got)), math.Abs(float64(want)))) {
		t.Fatalf("%s: got %v want %v", name, got, want)
	}
}

func check64(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) || math.Abs(got-want) > 2e-13*math.Max(1, math.Max(math.Abs(got), math.Abs(want))) {
		t.Fatalf("%s: got %v want %v", name, got, want)
	}
}

func checkC64(t *testing.T, name string, got, want complex64) {
	t.Helper()
	check32(t, name+" real", real(got), real(want))
	check32(t, name+" imag", imag(got), imag(want))
}

func checkC128(t *testing.T, name string, got, want complex128) {
	t.Helper()
	check64(t, name+" real", real(got), real(want))
	check64(t, name+" imag", imag(got), imag(want))
}

func check32s(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		check32(t, name, got[i], want[i])
	}
}

func check64s(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		check64(t, name, got[i], want[i])
	}
}

func checkC64s(t *testing.T, name string, got, want []complex64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		checkC64(t, name, got[i], want[i])
	}
}

func checkC128s(t *testing.T, name string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		checkC128(t, name, got[i], want[i])
	}
}

func checkInt(t *testing.T, name string, got, want int) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", name, got, want)
	}
}

func checkSrotmg(t *testing.T, got, want blas.SrotmParams) {
	t.Helper()
	if got.Flag != want.Flag {
		t.Fatalf("Srotmg flag: got %v want %v", got.Flag, want.Flag)
	}
	for _, i := range rotmgDefined(got.Flag) {
		check32(t, "Srotmg H", got.H[i], want.H[i])
	}
}

func checkDrotmg(t *testing.T, got, want blas.DrotmParams) {
	t.Helper()
	if got.Flag != want.Flag {
		t.Fatalf("Drotmg flag: got %v want %v", got.Flag, want.Flag)
	}
	for _, i := range rotmgDefined(got.Flag) {
		check64(t, "Drotmg H", got.H[i], want.H[i])
	}
}

func rotmgDefined(flag blas.Flag) []int {
	switch flag {
	case blas.Identity:
		return nil
	case blas.OffDiagonal:
		return []int{1, 2}
	case blas.Diagonal:
		return []int{0, 3}
	case blas.Rescaling:
		return []int{0, 1, 2, 3}
	default:
		panic("unexpected rotm flag")
	}
}
