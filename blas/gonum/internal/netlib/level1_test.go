// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib_test

import (
	"math/cmplx"
	"testing"

	gonum "plantcontrol.org/v1/gonum/blas/gonum"
	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

func TestComplexDotABI(t *testing.T) {
	x64 := []complex64{1 + 2i, -3 + 0.5i, 2 - 4i}
	y64 := []complex64{-2 + 3i, 0.25 - 2i, 1 + 0.75i}
	x128 := []complex128{1 + 2i, -3 + 0.5i, 2 - 4i}
	y128 := []complex128{-2 + 3i, 0.25 - 2i, 1 + 0.75i}
	for _, tc := range []struct {
		name string
		got  complex128
		want complex128
	}{
		{"Cdotu", complex128(netlib.Implementation{}.Cdotu(3, x64, 1, y64, 1)), complex128(gonum.Implementation{}.Cdotu(3, x64, 1, y64, 1))},
		{"Cdotc", complex128(netlib.Implementation{}.Cdotc(3, x64, 1, y64, 1)), complex128(gonum.Implementation{}.Cdotc(3, x64, 1, y64, 1))},
		{"Zdotu", netlib.Implementation{}.Zdotu(3, x128, 1, y128, 1), gonum.Implementation{}.Zdotu(3, x128, 1, y128, 1)},
		{"Zdotc", netlib.Implementation{}.Zdotc(3, x128, 1, y128, 1), gonum.Implementation{}.Zdotc(3, x128, 1, y128, 1)},
	} {
		if cmplx.IsNaN(tc.got) || cmplx.IsInf(tc.got) || cmplx.IsNaN(tc.want) || cmplx.IsInf(tc.want) || cmplx.Abs(tc.got-tc.want) > 2e-6*max(1, cmplx.Abs(tc.want)) {
			t.Errorf("%s=%v, want %v", tc.name, tc.got, tc.want)
		}
	}
}

func TestBatchedComplexDotChecksums(t *testing.T) {
	x64, y64 := []complex64{1 + 2i, -3 + 0.5i}, []complex64{-2 + 3i, 0.25 - 2i}
	x128, y128 := []complex128{1 + 2i, -3 + 0.5i}, []complex128{-2 + 3i, 0.25 - 2i}
	one := netlib.Implementation{}
	for _, repeat := range []int{2, 3} {
		batch := netlib.Implementation{Repeat: repeat}
		factor := complex(float64(repeat), 0)
		for _, tc := range []struct {
			name      string
			got, want complex128
		}{
			{"Cdotu", complex128(batch.Cdotu(2, x64, 1, y64, 1)), factor * complex128(one.Cdotu(2, x64, 1, y64, 1))},
			{"Cdotc", complex128(batch.Cdotc(2, x64, 1, y64, 1)), factor * complex128(one.Cdotc(2, x64, 1, y64, 1))},
			{"Zdotu", batch.Zdotu(2, x128, 1, y128, 1), factor * one.Zdotu(2, x128, 1, y128, 1)},
			{"Zdotc", batch.Zdotc(2, x128, 1, y128, 1), factor * one.Zdotc(2, x128, 1, y128, 1)},
		} {
			if cmplx.IsNaN(tc.got) || cmplx.IsInf(tc.got) || cmplx.Abs(tc.got-tc.want) > 2e-6*max(1, cmplx.Abs(tc.want)) {
				t.Errorf("repeat=%d %s=%v, want %v", repeat, tc.name, tc.got, tc.want)
			}
		}
	}
}

func TestBatchedReductionChecksums(t *testing.T) {
	x, y := []float64{1, -2, 3, 4}, []float64{-3, 0.5, 2, -1}
	one := netlib.Implementation{}.Ddot(4, x, 1, y, 1)
	for _, repeat := range []int{2, 3} {
		got := (netlib.Implementation{Repeat: repeat}).Ddot(4, x, 1, y, 1)
		if got != float64(repeat)*one {
			t.Errorf("repeat=%d checksum=%g, want %g", repeat, got, float64(repeat)*one)
		}
	}
}

func TestBatchedAxpyAlternatesSign(t *testing.T) {
	for _, repeat := range []int{2, 3} {
		xs, gs, ws := []float32{1, -2, 3}, []float32{4, 5, 6}, []float32{4, 5, 6}
		xd, gd, wd := []float64{1, -2, 3}, []float64{4, 5, 6}, []float64{4, 5, 6}
		xc, gc, wc := []complex64{1 + 2i, -2 + 1i, 3 - 1i}, []complex64{4, 5, 6}, []complex64{4, 5, 6}
		xz, gz, wz := []complex128{1 + 2i, -2 + 1i, 3 - 1i}, []complex128{4, 5, 6}, []complex128{4, 5, 6}
		batch := netlib.Implementation{Repeat: repeat}
		batch.Saxpy(3, 0.25, xs, 1, gs, 1)
		batch.Daxpy(3, 0.25, xd, 1, gd, 1)
		batch.Caxpy(3, 0.25+0.5i, xc, 1, gc, 1)
		batch.Zaxpy(3, 0.25+0.5i, xz, 1, gz, 1)
		goImpl := gonum.Implementation{}
		for j := 0; j < repeat; j++ {
			s, d, c, z := float32(0.25), 0.25, complex64(0.25+0.5i), complex128(0.25+0.5i)
			if j%2 != 0 {
				s = -s
				d = -d
				c = -c
				z = -z
			}
			goImpl.Saxpy(3, s, xs, 1, ws, 1)
			goImpl.Daxpy(3, d, xd, 1, wd, 1)
			goImpl.Caxpy(3, c, xc, 1, wc, 1)
			goImpl.Zaxpy(3, z, xz, 1, wz, 1)
		}
		if [3]float32(gs) != [3]float32(ws) {
			t.Errorf("repeat=%d Saxpy=%v want %v", repeat, gs, ws)
		}
		if [3]float64(gd) != [3]float64(wd) {
			t.Errorf("repeat=%d Daxpy=%v want %v", repeat, gd, wd)
		}
		if [3]complex64(gc) != [3]complex64(wc) {
			t.Errorf("repeat=%d Caxpy=%v want %v", repeat, gc, wc)
		}
		if [3]complex128(gz) != [3]complex128(wz) {
			t.Errorf("repeat=%d Zaxpy=%v want %v", repeat, gz, wz)
		}
	}
}

func TestBatchedGeneratorsUseConstantInputs(t *testing.T) {
	one := netlib.Implementation{}
	for _, repeat := range []int{2, 3} {
		batched := netlib.Implementation{Repeat: repeat}
		gcf, gsf, grf, gzf := one.Srotg(3, 4)
		bcf, bsf, brf, bzf := batched.Srotg(3, 4)
		if [4]float32{bcf, bsf, brf, bzf} != [4]float32{gcf, gsf, grf, gzf} {
			t.Errorf("Srotg repeat=%d differs from constant-input call", repeat)
		}
		gc, gs, gr, gz := one.Drotg(3, 4)
		bc, bs, br, bz := batched.Drotg(3, 4)
		if [4]float64{bc, bs, br, bz} != [4]float64{gc, gs, gr, gz} {
			t.Errorf("Drotg repeat=%d differs from constant-input call", repeat)
		}
		gpf, gd1f, gd2f, gx1f := one.Srotmg(2, 3, 4, 5)
		bpf, bd1f, bd2f, bx1f := batched.Srotmg(2, 3, 4, 5)
		if bpf != gpf || bd1f != gd1f || bd2f != gd2f || bx1f != gx1f {
			t.Errorf("Srotmg repeat=%d differs from constant-input call", repeat)
		}
		gp, gd1, gd2, gx1 := one.Drotmg(2, 3, 4, 5)
		bp, bd1, bd2, bx1 := batched.Drotmg(2, 3, 4, 5)
		if bp != gp || bd1 != gd1 || bd2 != gd2 || bx1 != gx1 {
			t.Errorf("Drotmg repeat=%d differs from constant-input call", repeat)
		}
	}
}

func TestIAMAXIndexConversion(t *testing.T) {
	x := []float64{1, -7, 3}
	if got := (netlib.Implementation{}).Idamax(3, x, 1); got != 1 {
		t.Fatalf("Idamax=%d, want 1", got)
	}
	if got := (netlib.Implementation{}).Idamax(0, nil, 1); got != -1 {
		t.Fatalf("empty Idamax=%d, want -1", got)
	}
}
