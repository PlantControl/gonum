// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

type Dlascler interface {
	Dlascl(kind lapack.MatrixType, kl, ku int, cfrom, cto float64, m, n int, a []float64, lda int)
}

func DlasclTest(t *testing.T, impl Dlascler) {
	const tol = 1e-15

	rnd := rand.New(rand.NewPCG(1, 1))
	for ti, test := range []struct {
		m, n int
	}{
		{0, 0},
		{1, 1},
		{1, 10},
		{10, 1},
		{2, 2},
		{2, 11},
		{11, 2},
		{3, 3},
		{3, 11},
		{11, 3},
		{11, 11},
		{11, 100},
		{100, 11},
	} {
		m := test.m
		n := test.n
		for _, extra := range []int{0, 11} {
			for _, kind := range []lapack.MatrixType{lapack.General, lapack.UpperTri, lapack.LowerTri} {
				a := randomGeneral(m, n, n+extra, rnd)
				aCopy := cloneGeneral(a)
				cfrom := rnd.NormFloat64()
				cto := rnd.NormFloat64()
				scale := cto / cfrom

				impl.Dlascl(kind, -1, -1, cfrom, cto, m, n, a.Data, a.Stride)

				prefix := fmt.Sprintf("Case #%v: kind=%v,m=%v,n=%v,extra=%v", ti, kind, m, n, extra)
				if !generalOutsideAllNaN(a) {
					t.Errorf("%v: out-of-range write to A", prefix)
				}
				switch kind {
				case lapack.UpperTri:
					var mod bool
				loopLower:
					for i := 0; i < m; i++ {
						for j := 0; j < min(i, n); j++ {
							if a.Data[i*a.Stride+j] != aCopy.Data[i*aCopy.Stride+j] {
								mod = true
								break loopLower
							}
						}
					}
					if mod {
						t.Errorf("%v: unexpected modification in lower triangle of A", prefix)
					}
				case lapack.LowerTri:
					var mod bool
				loopUpper:
					for i := 0; i < m; i++ {
						for j := i + 1; j < n; j++ {
							if a.Data[i*a.Stride+j] != aCopy.Data[i*aCopy.Stride+j] {
								mod = true
								break loopUpper
							}
						}
					}
					if mod {
						t.Errorf("%v: unexpected modification in upper triangle of A", prefix)
					}
				}

				var resid float64
				switch kind {
				case lapack.General:
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							want := scale * aCopy.Data[i*aCopy.Stride+j]
							got := a.Data[i*a.Stride+j]
							resid = math.Max(resid, math.Abs(want-got))
						}
					}
				case lapack.UpperTri:
					for i := 0; i < m; i++ {
						for j := i; j < n; j++ {
							want := scale * aCopy.Data[i*aCopy.Stride+j]
							got := a.Data[i*a.Stride+j]
							resid = math.Max(resid, math.Abs(want-got))
						}
					}
				case lapack.LowerTri:
					for i := 0; i < m; i++ {
						for j := 0; j <= min(i, n-1); j++ {
							want := scale * aCopy.Data[i*aCopy.Stride+j]
							got := a.Data[i*a.Stride+j]
							resid = math.Max(resid, math.Abs(want-got))
						}
					}
				}
				if resid > tol*float64(max(m, n)) {
					t.Errorf("%v: unexpected result; residual=%v, want<=%v", prefix, resid, tol*float64(max(m, n)))
				}
			}
		}
	}
	testDlasclUpperHessenberg(t, impl)
	testDlasclExtremeRatios(t, impl)
}

func testDlasclExtremeRatios(t *testing.T, impl Dlascler) {
	for _, tc := range []struct {
		name         string
		value, cfrom float64
		cto          float64
	}{
		{name: "Down", value: math.MaxFloat64, cfrom: math.MaxFloat64, cto: math.SmallestNonzeroFloat64},
		{name: "Up", value: math.SmallestNonzeroFloat64, cfrom: math.SmallestNonzeroFloat64, cto: math.MaxFloat64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := []float64{tc.value}
			impl.Dlascl(lapack.General, 0, 0, tc.cfrom, tc.cto, 1, 1, a, 1)
			if a[0] != tc.cto {
				t.Fatalf("scaled value=%g, want %g", a[0], tc.cto)
			}
		})
	}
}

func testDlasclUpperHessenberg(t *testing.T, impl Dlascler) {
	rnd := rand.New(rand.NewPCG(2, 2))
	for _, test := range []struct {
		m, n int
	}{
		{0, 0},
		{1, 1},
		{1, 10},
		{10, 1},
		{3, 11},
		{11, 3},
		{11, 11},
	} {
		for _, extra := range []int{0, 11} {
			a := randomGeneral(test.m, test.n, test.n+extra, rnd)
			aCopy := cloneGeneral(a)
			impl.Dlascl(lapack.UpperHessenberg, -1, -1, 3, -2, test.m, test.n, a.Data, a.Stride)
			prefix := fmt.Sprintf("m=%d,n=%d,extra=%d", test.m, test.n, extra)
			if !generalOutsideAllNaN(a) {
				t.Errorf("%s: out-of-range write to A", prefix)
			}
			for i := 0; i < test.m; i++ {
				for j := 0; j < test.n; j++ {
					want := aCopy.Data[i*aCopy.Stride+j]
					if j >= max(0, i-1) {
						want *= -2.0 / 3
					}
					got := a.Data[i*a.Stride+j]
					tol := 1e-15 * float64(max(test.m, test.n)) * math.Max(1, math.Abs(want))
					if math.Abs(got-want) > tol {
						t.Errorf("%s: A[%d,%d]=%g, want %g", prefix, i, j, got, want)
					}
				}
			}
		}
	}
}
