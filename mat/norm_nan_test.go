// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/lapack64"
)

// normNaNCase builds a matrix of one structure over a finite background; the
// callback writes each stored element.
type normNaNCase struct {
	name   string
	r, c   int
	stored func(i, j int) bool
	build  func(set func(put func(i, j int, v float64))) Matrix
}

func normBackground(i, j int) float64 {
	return float64(i+1) + float64(j+1)/8
}

func normNaNCases() []normNaNCase {
	always := func(int, int) bool { return true }
	var cases []normNaNCase
	for _, sz := range [][2]int{{1, 1}, {1, 3}, {3, 1}, {2, 2}, {3, 5}, {5, 3}, {4, 4}} {
		r, c := sz[0], sz[1]
		cases = append(cases, normNaNCase{
			name: fmt.Sprintf("Dense%dx%d", r, c), r: r, c: c, stored: always,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDense(r+2, c+3, nil)
				s := m.Slice(1, r+1, 1, c+1).(*Dense)
				set(s.Set)
				return s
			},
		})
		cases = append(cases, normNaNCase{
			name: fmt.Sprintf("Generic%dx%d", r, c), r: r, c: c, stored: always,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDense(r, c, nil)
				set(m.Set)
				return asBasicMatrix(m)
			},
		})
		for _, b := range [][2]int{{0, 0}, {1, 1}, {0, 2}, {2, 0}} {
			kl, ku := min(b[0], r-1), min(b[1], c-1)
			inBand := func(i, j int) bool { return j-i <= ku && i-j <= kl }
			cases = append(cases, normNaNCase{
				name: fmt.Sprintf("Band%dx%d_kl%d_ku%d", r, c, kl, ku), r: r, c: c, stored: inBand,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewBandDense(r, c, kl, ku, nil)
					set(m.SetBand)
					return m
				},
			})
			cases = append(cases, normNaNCase{
				name: fmt.Sprintf("GenericBand%dx%d_kl%d_ku%d", r, c, kl, ku), r: r, c: c, stored: inBand,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewBandDense(r, c, kl, ku, nil)
					set(m.SetBand)
					return asBasicBanded(m)
				},
			})
		}
	}
	for _, n := range []int{1, 2, 3, 4, 5, 8, 17} {
		cases = append(cases, normNaNCase{
			name: fmt.Sprintf("Diag%d", n), r: n, c: n,
			stored: func(i, j int) bool { return i == j },
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDiagDense(n, nil)
				set(func(i, _ int, v float64) { m.SetDiag(i, v) })
				return m
			},
		}, normNaNCase{
			name: fmt.Sprintf("DiagStrided%d", n), r: n, c: n,
			stored: func(i, j int) bool { return i == j },
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDense(n, n, nil)
				d := m.DiagView().(*DiagDense)
				set(func(i, _ int, v float64) { d.SetDiag(i, v) })
				return d
			},
		}, normNaNCase{
			name: fmt.Sprintf("GenericDiag%d", n), r: n, c: n,
			stored: func(i, j int) bool { return i == j },
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDiagDense(n, nil)
				set(func(i, _ int, v float64) { m.SetDiag(i, v) })
				return asBasicDiagonal(m)
			},
		}, normNaNCase{
			name: fmt.Sprintf("Vec%d", n), r: n, c: 1, stored: always,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewVecDense(n, nil)
				set(func(i, _ int, v float64) { m.SetVec(i, v) })
				return m
			},
		}, normNaNCase{
			name: fmt.Sprintf("VecInc%d", n), r: n, c: 1, stored: always,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewDense(n, 3, nil)
				v := m.ColView(1).(*VecDense)
				set(func(i, _ int, x float64) { v.SetVec(i, x) })
				return v
			},
		}, normNaNCase{
			name: fmt.Sprintf("GenericVec%d", n), r: n, c: 1, stored: always,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewVecDense(n, nil)
				set(func(i, _ int, v float64) { m.SetVec(i, v) })
				return asBasicVector(m)
			},
		})
		upper := func(i, j int) bool { return i <= j }
		lower := func(i, j int) bool { return i >= j }
		cases = append(cases, normNaNCase{
			name: fmt.Sprintf("Sym%d", n), r: n, c: n, stored: upper,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewSymDense(n, nil)
				set(m.SetSym)
				return m
			},
		}, normNaNCase{
			name: fmt.Sprintf("GenericSym%d", n), r: n, c: n, stored: upper,
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewSymDense(n, nil)
				set(m.SetSym)
				return asBasicSymmetric(m)
			},
		})
		for _, tc := range []struct {
			kind   TriKind
			name   string
			stored func(i, j int) bool
		}{{Upper, "Upper", upper}, {Lower, "Lower", lower}} {
			cases = append(cases, normNaNCase{
				name: fmt.Sprintf("Tri%s%d", tc.name, n), r: n, c: n, stored: tc.stored,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewTriDense(n, tc.kind, nil)
					set(m.SetTri)
					return m
				},
			}, normNaNCase{
				name: fmt.Sprintf("GenericTri%s%d", tc.name, n), r: n, c: n, stored: tc.stored,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewTriDense(n, tc.kind, nil)
					set(m.SetTri)
					return asBasicTriangular(m)
				},
			})
			for _, k := range []int{0, 1, 2} {
				k := min(k, n-1)
				inBand := func(i, j int) bool {
					return tc.stored(i, j) && j-i <= k && i-j <= k
				}
				cases = append(cases, normNaNCase{
					name: fmt.Sprintf("TriBand%s%d_k%d", tc.name, n, k), r: n, c: n, stored: inBand,
					build: func(set func(func(i, j int, v float64))) Matrix {
						m := NewTriBandDense(n, k, tc.kind, nil)
						set(m.SetTriBand)
						return m
					},
				}, normNaNCase{
					name: fmt.Sprintf("GenericTriBand%s%d_k%d", tc.name, n, k), r: n, c: n, stored: inBand,
					build: func(set func(func(i, j int, v float64))) Matrix {
						m := NewTriBandDense(n, k, tc.kind, nil)
						set(m.SetTriBand)
						return asBasicTriBanded(m)
					},
				})
			}
		}
		for _, k := range []int{0, 1, 2} {
			k := min(k, n-1)
			inBand := func(i, j int) bool { return i <= j && j-i <= k }
			cases = append(cases, normNaNCase{
				name: fmt.Sprintf("SymBand%d_k%d", n, k), r: n, c: n, stored: inBand,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewSymBandDense(n, k, nil)
					set(m.SetSymBand)
					return m
				},
			}, normNaNCase{
				name: fmt.Sprintf("GenericSymBand%d_k%d", n, k), r: n, c: n, stored: inBand,
				build: func(set func(func(i, j int, v float64))) Matrix {
					m := NewSymBandDense(n, k, nil)
					set(m.SetSymBand)
					return asBasicSymBanded(m)
				},
			})
		}
		cases = append(cases, normNaNCase{
			name: fmt.Sprintf("Tridiag%d", n), r: n, c: n,
			stored: func(i, j int) bool { return i-j <= 1 && j-i <= 1 },
			build: func(set func(func(i, j int, v float64))) Matrix {
				m := NewTridiag(n, nil, nil, nil)
				set(m.SetBand)
				return m
			},
		})
	}
	return cases
}

// normNaNWrappers lists the ways a matrix reaches Norm.
var normNaNWrappers = []struct {
	name string
	wrap func(Matrix) Matrix
}{
	{"direct", func(m Matrix) Matrix { return m }},
	{"T", func(m Matrix) Matrix { return m.T() }},
	{"Transpose", func(m Matrix) Matrix { return Transpose{m} }},
	{"TT", func(m Matrix) Matrix { return m.T().T() }},
}

func normNaNBuild(c normNaNCase, puts ...[3]float64) Matrix {
	return c.build(func(put func(i, j int, v float64)) {
		for i := 0; i < c.r; i++ {
			for j := 0; j < c.c; j++ {
				if c.stored(i, j) {
					put(i, j, normBackground(i, j))
				}
			}
		}
		for _, p := range puts {
			put(int(p[0]), int(p[1]), p[2])
		}
	})
}

func TestNormPropagatesNaN(t *testing.T) {
	t.Parallel()
	for _, c := range normNaNCases() {
		for i := 0; i < c.r; i++ {
			for j := 0; j < c.c; j++ {
				if !c.stored(i, j) {
					continue
				}
				for _, w := range normNaNWrappers {
					for _, norm := range []float64{1, 2, math.Inf(1)} {
						got := Norm(w.wrap(normNaNBuild(c, [3]float64{float64(i), float64(j), math.NaN()})), norm)
						if !math.IsNaN(got) {
							t.Errorf("%s %s NaN at (%d,%d) norm %v: got %v", c.name, w.name, i, j, norm, got)
						}
					}
				}
			}
		}
	}
}

func TestNormPropagatesNaNBesideInf(t *testing.T) {
	t.Parallel()
	for _, c := range normNaNCases() {
		var stored [][2]int
		for i := 0; i < c.r; i++ {
			for j := 0; j < c.c; j++ {
				if c.stored(i, j) {
					stored = append(stored, [2]int{i, j})
				}
			}
		}
		for _, p := range stored {
			for _, q := range stored {
				if p == q {
					continue
				}
				a := normNaNBuild(c,
					[3]float64{float64(p[0]), float64(p[1]), math.NaN()},
					[3]float64{float64(q[0]), float64(q[1]), math.Inf(-1)})
				for _, norm := range []float64{1, 2, math.Inf(1)} {
					if got := Norm(a, norm); !math.IsNaN(got) {
						t.Errorf("%s NaN at %v Inf at %v norm %v: got %v", c.name, p, q, norm, got)
					}
				}
			}
		}
	}
}

// TestNormIgnoresUnstoredNaN pins that storage outside the structure is
// unreferenced, as for reference Dlansy and Dlantr.
func TestNormIgnoresUnstoredNaN(t *testing.T) {
	t.Parallel()
	nan := math.NaN()
	for _, norm := range []float64{1, 2, math.Inf(1)} {
		sym := NewSymDense(2, []float64{1, 2, nan, 3})
		if got := Norm(sym, norm); math.IsNaN(got) {
			t.Errorf("SymDense norm %v: got NaN from unreferenced triangle", norm)
		}
		upper := NewTriDense(2, Upper, []float64{1, 2, nan, 3})
		if got := Norm(upper, norm); math.IsNaN(got) {
			t.Errorf("upper TriDense norm %v: got NaN from unreferenced triangle", norm)
		}
		lower := NewTriDense(2, Lower, []float64{1, nan, 2, 3})
		if got := Norm(lower, norm); math.IsNaN(got) {
			t.Errorf("lower TriDense norm %v: got NaN from unreferenced triangle", norm)
		}
	}
}

// TestLapackNormsPropagateNaN covers lapack.MaxAbs, which Norm does not expose,
// on the routines behind the Normer implementations.
func TestLapackNormsPropagateNaN(t *testing.T) {
	t.Parallel()
	nan := math.NaN()
	norms := []lapack.MatrixNorm{lapack.MaxAbs, lapack.MaxColumnSum, lapack.MaxRowSum, lapack.Frobenius}
	const n = 4
	for _, norm := range norms {
		work := make([]float64, n)
		for pos := 0; pos < n*n; pos++ {
			i, j := pos/n, pos%n

			ge := NewDense(n, n, nil)
			fillNormBackground(ge)
			ge.Set(i, j, nan)
			if got := lapack64.Lange(norm, ge.RawMatrix(), work); !math.IsNaN(got) {
				t.Errorf("Lange %c NaN at (%d,%d): got %v", norm, i, j, got)
			}
			if i <= j {
				sy := NewSymDense(n, nil)
				fillNormBackgroundFunc(n, func(a, b int, v float64) {
					if a <= b {
						sy.SetSym(a, b, v)
					}
				})
				sy.SetSym(i, j, nan)
				if got := lapack64.Lansy(norm, sy.RawSymmetric(), work); !math.IsNaN(got) {
					t.Errorf("Lansy %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
				tr := NewTriDense(n, Upper, nil)
				fillNormBackgroundFunc(n, func(a, b int, v float64) {
					if a <= b {
						tr.SetTri(a, b, v)
					}
				})
				tr.SetTri(i, j, nan)
				if got := lapack64.Lantr(norm, tr.RawTriangular(), work); !math.IsNaN(got) {
					t.Errorf("Lantr %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
			}
			if i-j <= 1 && j-i <= 1 {
				gb := NewBandDense(n, n, 1, 1, nil)
				fillNormBackgroundFunc(n, func(a, b int, v float64) {
					if a-b <= 1 && b-a <= 1 {
						gb.SetBand(a, b, v)
					}
				})
				gb.SetBand(i, j, nan)
				if got := lapack64.Langb(norm, gb.RawBand()); !math.IsNaN(got) {
					t.Errorf("Langb %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
				gt := NewTridiag(n, nil, nil, nil)
				fillNormBackgroundFunc(n, func(a, b int, v float64) {
					if a-b <= 1 && b-a <= 1 {
						gt.SetBand(a, b, v)
					}
				})
				gt.SetBand(i, j, nan)
				if got := lapack64.Langt(norm, gt.RawTridiagonal()); !math.IsNaN(got) {
					t.Errorf("Langt %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
			}
			if i <= j && j-i <= 1 {
				sb := NewSymBandDense(n, 1, nil)
				tb := NewTriBandDense(n, 1, Upper, nil)
				fillNormBackgroundFunc(n, func(a, b int, v float64) {
					if a <= b && b-a <= 1 {
						sb.SetSymBand(a, b, v)
						tb.SetTriBand(a, b, v)
					}
				})
				sb.SetSymBand(i, j, nan)
				tb.SetTriBand(i, j, nan)
				if got := lapack64.Lansb(norm, sb.RawSymBand(), work); !math.IsNaN(got) {
					t.Errorf("Lansb %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
				if got := lapack64.Lantb(norm, tb.RawTriBand(), work); !math.IsNaN(got) {
					t.Errorf("Lantb %c NaN at (%d,%d): got %v", norm, i, j, got)
				}
			}
		}
	}
}

func fillNormBackground(m *Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, normBackground(i, j))
		}
	}
}

func fillNormBackgroundFunc(n int, put func(i, j int, v float64)) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			put(i, j, normBackground(i, j))
		}
	}
}

// nanSkippingNormer reports a fixed norm whatever the elements, standing in for
// a Normer outside this package that skips NaN.
type nanSkippingNormer struct {
	*Dense
	norm float64
}

func (n nanSkippingNormer) Norm(float64) float64 { return n.norm }

func expAllNaN(a Matrix) bool {
	var e Dense
	e.Exp(a)
	r, c := e.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if !math.IsNaN(e.At(i, j)) {
				return false
			}
		}
	}
	return true
}

func TestDenseExpNaNEveryMatrixType(t *testing.T) {
	t.Parallel()
	for _, c := range normNaNCases() {
		if c.r != c.c || c.r < 2 {
			continue
		}
		for i := 0; i < c.r; i++ {
			for j := 0; j < c.c; j++ {
				if !c.stored(i, j) {
					continue
				}
				if !expAllNaN(normNaNBuild(c, [3]float64{float64(i), float64(j), math.NaN()})) {
					t.Errorf("%s NaN at (%d,%d): Exp not all NaN", c.name, i, j)
				}
			}
		}
	}
}

func TestDenseExpExternalNormerSkippingNaN(t *testing.T) {
	t.Parallel()
	nan := math.NaN()
	for _, data := range [][]float64{
		{0, nan, 0, 0},
		{0, 0, nan, 0},
		{nan, 0, 0, 0},
		{0, 0, 0, nan},
		{4.5, 0, 0, nan},
		{nan, 0, 0, 4.5},
		{1, nan, 0, 1},
	} {
		for _, norm := range []float64{0, 0.2, 0.5, 2, 100} {
			if !expAllNaN(nanSkippingNormer{NewDense(2, 2, data), norm}) {
				t.Errorf("a=%v norm %v: Exp not all NaN", data, norm)
			}
		}
	}
}
