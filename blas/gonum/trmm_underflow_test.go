// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// trmmUnderflowCases exercise ?trmm when alpha*B[i,k] underflows to zero
// while B[i,k] does not. The two-element row has one element whose scaled
// value underflows and one that does not.
var trmmUnderflowCases = []struct {
	uplo blas.Uplo
	diag blas.Diag
	n    int
	b    []float64
	want []float64
}{
	{blas.Upper, blas.NonUnit, 1, []float64{1e-30}, []float64{0}},
	{blas.Lower, blas.NonUnit, 1, []float64{1e-30}, []float64{0}},
	{blas.Upper, blas.NonUnit, 2, []float64{1e-30, 1}, []float64{0, 1e-300}},
	{blas.Upper, blas.Unit, 2, []float64{1e-30, 1}, []float64{0, 1e-300}},
	{blas.Lower, blas.NonUnit, 2, []float64{1e-30, 1}, []float64{1e-300, 1e-300}},
	{blas.Lower, blas.Unit, 2, []float64{1e-30, 1}, []float64{1e-300, 1e-300}},
}

const trmmUnderflowAlpha = 1e-300

func trmmOnes(n int) []float64 {
	a := make([]float64, n*n)
	for i := range a {
		a[i] = 1
	}
	return a
}

func TestDtrmmRightUnderflow(t *testing.T) {
	t.Parallel()
	for _, tc := range trmmUnderflowCases {
		b := append([]float64(nil), tc.b...)
		impl.Dtrmm(blas.Right, tc.uplo, blas.NoTrans, tc.diag, 1, tc.n, trmmUnderflowAlpha, trmmOnes(tc.n), tc.n, b, tc.n)
		for j := range b {
			if b[j] != tc.want[j] {
				t.Errorf("uplo=%c diag=%c b=%v: got %v, want %v", tc.uplo, tc.diag, tc.b, b, tc.want)
				break
			}
		}
	}
}

func TestZtrmmRightUnderflow(t *testing.T) {
	t.Parallel()
	for _, tc := range trmmUnderflowCases {
		a := make([]complex128, tc.n*tc.n)
		for i := range a {
			a[i] = 1
		}
		b := make([]complex128, tc.n)
		for j, v := range tc.b {
			b[j] = complex(v, v)
		}
		impl.Ztrmm(blas.Right, tc.uplo, blas.NoTrans, tc.diag, 1, tc.n, trmmUnderflowAlpha, a, tc.n, b, tc.n)
		for j := range b {
			if want := complex(tc.want[j], tc.want[j]); b[j] != want {
				t.Errorf("uplo=%c diag=%c b=%v: got %v, want %v", tc.uplo, tc.diag, tc.b, b, tc.want)
				break
			}
		}
	}
}

// The single precision variants are generated; scale the cases into float32 range.
func TestStrmmCtrmmRightUnderflow(t *testing.T) {
	t.Parallel()
	const alpha = 1e-30
	scale := func(v float64) float32 {
		if v == 1e-300 {
			return 1e-30
		}
		if v == 1e-30 {
			return 1e-20
		}
		return float32(v)
	}
	for _, tc := range trmmUnderflowCases {
		name := fmt.Sprintf("uplo=%c diag=%c n=%d", tc.uplo, tc.diag, tc.n)
		bs := make([]float32, tc.n)
		bc := make([]complex64, tc.n)
		as := make([]float32, tc.n*tc.n)
		ac := make([]complex64, tc.n*tc.n)
		for i := range as {
			as[i], ac[i] = 1, 1
		}
		for j, v := range tc.b {
			bs[j] = scale(v)
			bc[j] = complex(scale(v), scale(v))
		}
		impl.Strmm(blas.Right, tc.uplo, blas.NoTrans, tc.diag, 1, tc.n, alpha, as, tc.n, bs, tc.n)
		impl.Ctrmm(blas.Right, tc.uplo, blas.NoTrans, tc.diag, 1, tc.n, alpha, ac, tc.n, bc, tc.n)
		for j := range bs {
			want := scale(tc.want[j])
			if bs[j] != want {
				t.Errorf("Strmm %s: got %v, want element %d = %v", name, bs, j, want)
			}
			if bc[j] != complex(want, want) {
				t.Errorf("Ctrmm %s: got %v, want element %d = %v", name, bc, j, complex(want, want))
			}
		}
	}
}
