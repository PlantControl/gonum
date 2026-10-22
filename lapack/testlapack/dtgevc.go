// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dtgevcer interface {
	Dtgevc(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int,
		s []float64, lds int, p []float64, ldp int, vl []float64, ldvl int, vr []float64, ldvr int,
		mm int, work []float64) (m int, ok bool)
}

func DtgevcTest(t *testing.T, impl Dtgevcer) {
	// Test with 2x2 matrix pair in generalized Schur form.
	testDtgevc2x2(t, impl)

	// Test with 3x3 matrix pair.
	testDtgevc3x3(t, impl)

	testDtgevcResiduals(t, impl)
}

func testDtgevc2x2(t *testing.T, impl Dtgevcer) {
	// 2x2 quasi-triangular S and upper triangular P.
	s := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			2, 1,
			0, 3,
		},
	}
	p := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0.5,
			0, 1,
		},
	}
	vl := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: make([]float64, 4),
	}
	vr := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: make([]float64, 4),
	}

	work := make([]float64, 12)

	m, ok := impl.Dtgevc(lapack.EVBoth, lapack.EVAll, nil, 2,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		2, work)

	if !ok {
		t.Error("2x2 test: Dtgevc failed")
	}

	if m != 2 {
		t.Errorf("2x2 test: m = %d, want 2", m)
	}
}

func testDtgevc3x3(t *testing.T, impl Dtgevcer) {
	// 3x3 quasi-triangular S and upper triangular P.
	s := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.5,
			0, 3, 1,
			0, 0, 4,
		},
	}
	p := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.3, 0.1,
			0, 1, 0.2,
			0, 0, 1,
		},
	}
	vl := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	vr := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	work := make([]float64, 18)

	m, ok := impl.Dtgevc(lapack.EVBoth, lapack.EVAll, nil, 3,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		3, work)

	if !ok {
		t.Error("3x3 test: Dtgevc failed")
	}

	if m != 3 {
		t.Errorf("3x3 test: m = %d, want 3", m)
	}

	// Test with selected eigenvectors.
	selected := []bool{true, false, true}
	vl = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	vr = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	m, ok = impl.Dtgevc(lapack.EVBoth, lapack.EVSelected, selected, 3,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		2, work)

	if !ok {
		t.Error("3x3 selected test: Dtgevc failed")
	}

	if m != 2 {
		t.Errorf("3x3 selected test: m = %d, want 2", m)
	}
}

// randomGeneralizedSchurPair returns a random n×n quasi-upper triangular S
// with standardized 2×2 blocks and an upper triangular P with diagonal 2×2
// blocks, together with the eigenvalues alpha/beta of (S,P). If inf, a real
// eigenvalue is made infinite.
func randomGeneralizedSchurPair(n, stride int, inf bool, rnd *rand.Rand) (s, p blas64.General, alphar, alphai, beta []float64) {
	s = nanGeneral(n, n, stride)
	p = nanGeneral(n, n, stride)
	alphar = make([]float64, n)
	alphai = make([]float64, n)
	beta = make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s.Data[i*stride+j] = 0
			p.Data[i*stride+j] = 0
			if j > i {
				s.Data[i*stride+j] = rnd.NormFloat64()
				p.Data[i*stride+j] = rnd.NormFloat64()
			}
		}
	}
	for j := 0; j < n; {
		if j < n-1 && rnd.Float64() < 0.5 {
			for {
				s11, s22 := rnd.NormFloat64(), rnd.NormFloat64()
				s12, s21 := rnd.Float64()+0.5, -(rnd.Float64() + 0.5)
				d1, d2 := rnd.Float64()+0.5, rnd.Float64()+0.5
				tr := s11*d2 + s22*d1
				disc := tr*tr - 4*d1*d2*(s11*s22-s12*s21)
				if disc >= -1e-2 {
					continue
				}
				s.Data[j*stride+j], s.Data[j*stride+j+1] = s11, s12
				s.Data[(j+1)*stride+j], s.Data[(j+1)*stride+j+1] = s21, s22
				p.Data[j*stride+j], p.Data[j*stride+j+1] = d1, 0
				p.Data[(j+1)*stride+j+1] = d2
				wr := tr / (2 * d1 * d2)
				wi := math.Sqrt(-disc) / (2 * d1 * d2)
				alphar[j], alphai[j], beta[j] = wr, wi, 1
				alphar[j+1], alphai[j+1], beta[j+1] = wr, -wi, 1
				break
			}
			j += 2
			continue
		}
		s.Data[j*stride+j] = rnd.NormFloat64()
		p.Data[j*stride+j] = rnd.Float64() + 0.5
		if inf {
			p.Data[j*stride+j] = 0
			inf = false
		}
		alphar[j], beta[j] = s.Data[j*stride+j], p.Data[j*stride+j]
		j++
	}
	return s, p, alphar, alphai, beta
}

func testDtgevcResiduals(t *testing.T, impl Dtgevcer) {
	bi := blas64.Implementation()
	rnd := rand.New(rand.NewPCG(1, 2))
	for n := 1; n <= 10; n++ {
		for _, extra := range []int{0, 3} {
			ld := n + extra
			for trial := 0; trial < 10; trial++ {
				name := fmt.Sprintf("n=%d ld=%d trial=%d", n, ld, trial)
				s, p, alphar, alphai, beta := randomGeneralizedSchurPair(n, ld, trial == 9, rnd)
				sOrig, pOrig := cloneGeneral(s), cloneGeneral(p)
				work := nanSlice(6 * n)

				vl := nanGeneral(n, n, ld)
				vr := nanGeneral(n, n, ld)
				m, ok := impl.Dtgevc(lapack.EVBoth, lapack.EVAll, nil, n, s.Data, ld, p.Data, ld,
					vl.Data, ld, vr.Data, ld, n, work)
				if !ok || m != n {
					t.Errorf("%s EVAll: ok=%v m=%d", name, ok, m)
					continue
				}
				if !equalGeneral(s, sOrig) || !equalGeneral(p, pOrig) {
					t.Errorf("%s: S or P modified", name)
				}
				checkGeneralizedEigenvectorResiduals(t, name+" EVAll left", s, p, alphar, alphai, beta, vl.Data, ld, true)
				checkGeneralizedEigenvectorResiduals(t, name+" EVAll right", s, p, alphar, alphai, beta, vr.Data, ld, false)

				for _, side := range []lapack.EVSide{lapack.EVLeft, lapack.EVRight} {
					v := nanGeneral(n, n, ld)
					want := vl
					vlp, vrp := v.Data, []float64(nil)
					if side == lapack.EVRight {
						want = vr
						vlp, vrp = nil, v.Data
					}
					impl.Dtgevc(side, lapack.EVAll, nil, n, s.Data, ld, p.Data, ld, vlp, ld, vrp, ld, n, work)
					if !equalGeneral(v, want) {
						t.Errorf("%s: side=%c result differs from EVBoth", name, side)
					}
				}

				sel := make([]bool, n)
				var cols []int
				for j := 0; j < n; j++ {
					sel[j] = rnd.Float64() < 0.5
				}
				for j := 0; j < n; j++ {
					if alphai[j] > 0 {
						if sel[j] || sel[j+1] {
							cols = append(cols, j, j+1)
						}
						j++
						continue
					}
					if sel[j] {
						cols = append(cols, j)
					}
				}
				vls := nanGeneral(n, max(1, len(cols)), max(1, len(cols)))
				vrs := nanGeneral(n, max(1, len(cols)), max(1, len(cols)))
				m, ok = impl.Dtgevc(lapack.EVBoth, lapack.EVSelected, sel, n, s.Data, ld, p.Data, ld,
					vls.Data, vls.Stride, vrs.Data, vrs.Stride, len(cols), work)
				if !ok || m != len(cols) {
					t.Errorf("%s EVSelected: ok=%v m=%d, want %d", name, ok, m, len(cols))
				} else {
					for k, j := range cols {
						for i := 0; i < n; i++ {
							if vls.Data[i*vls.Stride+k] != vl.Data[i*ld+j] || vrs.Data[i*vrs.Stride+k] != vr.Data[i*ld+j] {
								t.Errorf("%s EVSelected: column %d differs from EVAll column %d", name, k, j)
								break
							}
						}
					}
				}

				q := randomOrthogonal(n, rnd)
				z := randomOrthogonal(n, rnd)
				a := zeros(n, n, n)
				b := zeros(n, n, n)
				tmp := zeros(n, n, n)
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, q.Data, q.Stride, s.Data, ld, 0, tmp.Data, n)
				bi.Dgemm(blas.NoTrans, blas.Trans, n, n, n, 1, tmp.Data, n, z.Data, z.Stride, 0, a.Data, n)
				bi.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, q.Data, q.Stride, p.Data, ld, 0, tmp.Data, n)
				bi.Dgemm(blas.NoTrans, blas.Trans, n, n, n, 1, tmp.Data, n, z.Data, z.Stride, 0, b.Data, n)
				vlq := nanGeneral(n, n, ld)
				vrz := nanGeneral(n, n, ld)
				copyGeneral(vlq, q)
				copyGeneral(vrz, z)
				m, ok = impl.Dtgevc(lapack.EVBoth, lapack.EVAllMulQ, nil, n, s.Data, ld, p.Data, ld,
					vlq.Data, ld, vrz.Data, ld, n, work)
				if !ok || m != n {
					t.Errorf("%s EVAllMulQ: ok=%v m=%d", name, ok, m)
					continue
				}
				checkGeneralizedEigenvectorResiduals(t, name+" EVAllMulQ left", a, b, alphar, alphai, beta, vlq.Data, ld, true)
				checkGeneralizedEigenvectorResiduals(t, name+" EVAllMulQ right", a, b, alphar, alphai, beta, vrz.Data, ld, false)
			}
		}
	}
}
