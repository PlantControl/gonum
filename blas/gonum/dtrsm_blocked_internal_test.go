// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func TestDtrsmBlocked(t *testing.T) {
	for _, m := range []int{63, 64, 65, 127, 128, 129} {
		for _, n := range []int{15, 16, 17} {
			for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
				for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
					for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
						name := fmt.Sprintf("m%d/n%d/%c/%c/%c", m, n, ul, trans, diag)
						t.Run(name, func(t *testing.T) {
							lda, ldb := m+3, n+5
							a := makeDtrsmCandidateA(m, lda, ul, diag)
							for i := 0; i < m; i++ {
								if diag == blas.Unit {
									a[i*lda+i] = math.NaN()
								}
								for j := 0; j < m; j++ {
									if (ul == blas.Upper && j < i) || (ul == blas.Lower && j > i) {
										a[i*lda+j] = math.NaN()
									}
								}
							}
							aOrig := slices.Clone(a)
							orig := make([]float64, (m-1)*ldb+n)
							for i := range orig {
								orig[i] = math.Float64frombits(0x3ff0000000000000 + uint64(i%31))
							}
							for i := 0; i < m; i++ {
								for j := 0; j < n; j++ {
									orig[i*ldb+j] = float64((i*7+j*3)%23-11) / 8
								}
							}
							got, want := slices.Clone(orig), slices.Clone(orig)
							Implementation{}.Dtrsm(blas.Left, ul, trans, diag, m, n, 1, a, lda, got, ldb)
							for j := 0; j < n; j++ {
								Implementation{}.Dtrsm(blas.Left, ul, trans, diag, m, 1, 1, a, lda, want[j:], ldb)
							}
							for i := range got {
								if i%ldb >= n {
									if math.Float64bits(got[i]) != math.Float64bits(orig[i]) {
										t.Fatalf("padding changed at %d", i)
									}
									continue
								}
								if !dtrsmCandidateClose(got[i], want[i]) {
									t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
								}
							}
							if !slices.EqualFunc(a, aOrig, func(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) }) {
								t.Fatal("A changed")
							}
							if residual := dtrsmCandidateResidual(ul, trans, diag, m, n, a, lda, got, ldb, orig); math.IsNaN(residual) || math.IsInf(residual, 0) || residual > 2e-11 {
								t.Fatalf("relative residual %g", residual)
							}
						})
					}
				}
			}
		}
	}
}

func dtrsmCandidateClose(got, want float64) bool {
	if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		return false
	}
	return math.Abs(got-want) <= 2e-12*(1+math.Abs(want))
}

func makeDtrsmCandidateA(n, lda int, ul blas.Uplo, diag blas.Diag) []float64 {
	a := make([]float64, (n-1)*lda+n)
	for i := 0; i < n; i++ {
		if diag == blas.NonUnit {
			a[i*lda+i] = 2 + float64(i%7)/16
		}
		for j := 0; j < n; j++ {
			if (ul == blas.Upper && j > i) || (ul == blas.Lower && j < i) {
				a[i*lda+j] = float64((i+j)%11-5) / float64(16*n)
			}
		}
	}
	return a
}

func dtrsmCandidateResidual(ul blas.Uplo, trans blas.Transpose, diag blas.Diag, m, n int, a []float64, lda int, x []float64, ldx int, rhs []float64) float64 {
	var maxResidual, scale float64
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < m; k++ {
				row, col := i, k
				if trans != blas.NoTrans {
					row, col = k, i
				}
				if (ul == blas.Upper && col < row) || (ul == blas.Lower && col > row) {
					continue
				}
				av := a[row*lda+col]
				if diag == blas.Unit && row == col {
					av = 1
				}
				sum += av * x[k*ldx+j]
			}
			maxResidual = max(maxResidual, math.Abs(sum-rhs[i*ldx+j]))
			scale = max(scale, math.Abs(rhs[i*ldx+j]))
		}
	}
	return maxResidual / max(1, scale)
}

func TestDtrsmBlockedNonfiniteZero(t *testing.T) {
	const m, n = 128, 16
	for _, orient := range []struct {
		name              string
		ul                blas.Uplo
		trans             blas.Transpose
		source, target    int
		coefficientOffset int
	}{
		{name: "lower-no-trans", ul: blas.Lower, trans: blas.NoTrans, source: 0, target: 64, coefficientOffset: 64 * m},
		{name: "upper-trans", ul: blas.Upper, trans: blas.Trans, source: 0, target: 64, coefficientOffset: 64},
		{name: "upper-no-trans", ul: blas.Upper, trans: blas.NoTrans, source: 64, target: 63, coefficientOffset: 63*m + 64},
		{name: "lower-trans", ul: blas.Lower, trans: blas.Trans, source: 64, target: 63, coefficientOffset: 64*m + 63},
	} {
		for _, coefficient := range []float64{0, math.Copysign(0, -1), 1} {
			t.Run(fmt.Sprintf("%s/coefficient=%g", orient.name, coefficient), func(t *testing.T) {
				a := makeDtrsmCandidateA(m, m, orient.ul, blas.Unit)
				for i := range a {
					if i%m != i/m {
						a[i] = 0
					}
				}
				for i := 0; i < m; i++ {
					a[i*m+i] = math.NaN()
				}
				a[orient.coefficientOffset] = coefficient
				b := make([]float64, m*n)
				for i := range b {
					b[i] = 1
				}
				for j := 0; j < n; j++ {
					b[orient.source*n+j] = math.NaN()
				}
				Implementation{}.Dtrsm(blas.Left, orient.ul, orient.trans, blas.Unit, m, n, 1, a, m, b, n)
				for j := 0; j < n; j++ {
					got := b[orient.target*n+j]
					if coefficient == 0 && got != 1 {
						t.Fatalf("zero coefficient contaminated column %d: %g", j, got)
					}
					if coefficient != 0 && !math.IsNaN(got) {
						t.Fatalf("active coefficient did not propagate NaN in column %d: %g", j, got)
					}
				}
			})
		}
	}
}

func TestDtrsmBlockedDiagonalNonfinite(t *testing.T) {
	const m, n = 128, 16
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
				t.Run(fmt.Sprintf("%c/%c/%c", ul, trans, diag), func(t *testing.T) {
					a := make([]float64, m*m)
					for i := 0; i < m; i++ {
						a[i*m+i] = 1
					}
					a[64*m+64] = math.NaN()
					b := make([]float64, m*n)
					for i := range b {
						b[i] = 1
					}
					Implementation{}.Dtrsm(blas.Left, ul, trans, diag, m, n, 1, a, m, b, n)
					for j := 0; j < n; j++ {
						got := b[64*n+j]
						if diag == blas.Unit && got != 1 {
							t.Fatalf("stored unit diagonal used in column %d: %g", j, got)
						}
						if diag == blas.NonUnit && !math.IsNaN(got) {
							t.Fatalf("non-unit NaN diagonal did not propagate in column %d: %g", j, got)
						}
					}
				})
			}
		}
	}
}

func TestDtrsmBlockedAlphaFallback(t *testing.T) {
	const m, n, alpha = 128, 16, 0.75
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			t.Run(fmt.Sprintf("%c/%c", ul, trans), func(t *testing.T) {
				a := makeDtrsmCandidateA(m, m, ul, blas.NonUnit)
				orig := make([]float64, m*n)
				for i := range orig {
					orig[i] = float64(i%23-11) / 8
				}
				got, want := slices.Clone(orig), slices.Clone(orig)
				Implementation{}.Dtrsm(blas.Left, ul, trans, blas.NonUnit, m, n, alpha, a, m, got, n)
				for j := 0; j < n; j++ {
					Implementation{}.Dtrsm(blas.Left, ul, trans, blas.NonUnit, m, 1, alpha, a, m, want[j:], n)
				}
				for i := range got {
					if !dtrsmCandidateClose(got[i], want[i]) {
						t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
					}
				}
			})
		}
	}
}

func TestDtrsmBlockedBackwardOverflow(t *testing.T) {
	const m, n = 128, 16
	for _, tc := range []struct {
		name  string
		ul    blas.Uplo
		trans blas.Transpose
	}{
		{name: "upper-no-trans", ul: blas.Upper, trans: blas.NoTrans},
		{name: "lower-trans", ul: blas.Lower, trans: blas.Trans},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float64, m*m)
			rows := []int{0, 1, 64}
			if tc.trans == blas.NoTrans {
				a[1] = 1
				a[64] = -1
			} else {
				a[64*m] = -1
				a[65*m] = 1
				rows = []int{0, 64, 65}
			}
			b := make([]float64, m*n)
			for _, row := range rows {
				for j := 0; j < n; j++ {
					b[row*n+j] = 1e308
				}
			}
			Implementation{}.Dtrsm(blas.Left, tc.ul, tc.trans, blas.Unit, m, n, 1, a, m, b, n)
			for j := 0; j < n; j++ {
				if got := b[j]; got != 1e308 {
					t.Fatalf("column %d: got %g want %g", j, got, 1e308)
				}
			}
		})
	}
}

func TestStrsmBlockedBackwardOverflow(t *testing.T) {
	const m, n = 128, 16
	for _, tc := range []struct {
		name  string
		ul    blas.Uplo
		trans blas.Transpose
	}{
		{name: "upper-no-trans", ul: blas.Upper, trans: blas.NoTrans},
		{name: "lower-trans", ul: blas.Lower, trans: blas.Trans},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := make([]float32, m*m)
			rows := []int{0, 1, 64}
			if tc.trans == blas.NoTrans {
				a[1] = 1
				a[64] = -1
			} else {
				a[64*m] = -1
				a[65*m] = 1
				rows = []int{0, 64, 65}
			}
			b := make([]float32, m*n)
			for _, row := range rows {
				for j := 0; j < n; j++ {
					b[row*n+j] = 2e38
				}
			}
			Implementation{}.Strsm(blas.Left, tc.ul, tc.trans, blas.Unit, m, n, 1, a, m, b, n)
			for j := 0; j < n; j++ {
				if got := b[j]; got != 2e38 {
					t.Fatalf("column %d: got %g want %g", j, got, float32(2e38))
				}
			}
		})
	}
}

func TestStrsmBlocked(t *testing.T) {
	const m, n = 128, 16
	for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			t.Run(fmt.Sprintf("%c/%c", ul, trans), func(t *testing.T) {
				const lda, ldb = m + 3, n + 5
				a64 := makeDtrsmCandidateA(m, lda, ul, blas.NonUnit)
				a := make([]float32, len(a64))
				for i, v := range a64 {
					a[i] = float32(v)
				}
				orig := make([]float32, (m-1)*ldb+n)
				for i := range orig {
					orig[i] = float32(i%23-11) / 8
				}
				got, want := slices.Clone(orig), slices.Clone(orig)
				Implementation{}.Strsm(blas.Left, ul, trans, blas.NonUnit, m, n, 1, a, lda, got, ldb)
				for j := 0; j < n; j++ {
					Implementation{}.Strsm(blas.Left, ul, trans, blas.NonUnit, m, 1, 1, a, lda, want[j:], ldb)
				}
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Abs(float64(got[i]-want[i])) > 3e-4*(1+math.Abs(float64(want[i]))) {
						t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
					}
				}
			})
		}
	}
}

func BenchmarkDtrsmBlockedSizes(b *testing.B) {
	cases := [][2]int{
		{32, 1}, {128, 1}, {256, 1}, {128, 2}, {128, 3}, {128, 4},
		{127, 15}, {127, 16}, {127, 17},
		{128, 15}, {128, 16}, {128, 17},
		{129, 15}, {129, 16}, {129, 17},
		{256, 16}, {256, 64}, {512, 16}, {512, 64},
	}
	for _, dims := range cases {
		m, n := dims[0], dims[1]
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				name := fmt.Sprintf("m%d/n%d/%c/%c", m, n, ul, trans)
				b.Run(name, func(b *testing.B) {
					a := makeDtrsmCandidateA(m, m, ul, blas.NonUnit)
					orig := make([]float64, m*n)
					for i := range orig {
						orig[i] = float64(i%17-8) / 8
					}
					data := slices.Clone(orig)
					b.ReportAllocs()
					for b.Loop() {
						Implementation{}.Dtrsm(blas.Left, ul, trans, blas.NonUnit, m, n, 1, a, m, data, n)
						b.StopTimer()
						copy(data, orig)
						b.StartTimer()
					}
				})
			}
		}
	}
}

func BenchmarkStrsmBlockedSizes(b *testing.B) {
	cases := [][2]int{
		{32, 1}, {128, 1}, {256, 1}, {128, 2}, {128, 3}, {128, 4},
		{127, 15}, {127, 16}, {127, 17},
		{128, 15}, {128, 16}, {128, 17},
		{129, 15}, {129, 16}, {129, 17},
		{256, 16}, {256, 64}, {512, 16}, {512, 64},
	}
	for _, dims := range cases {
		m, n := dims[0], dims[1]
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				name := fmt.Sprintf("m%d/n%d/%c/%c", m, n, ul, trans)
				b.Run(name, func(b *testing.B) {
					a64 := makeDtrsmCandidateA(m, m, ul, blas.NonUnit)
					a := make([]float32, len(a64))
					for i, v := range a64 {
						a[i] = float32(v)
					}
					orig := make([]float32, m*n)
					for i := range orig {
						orig[i] = float32(i%17-8) / 8
					}
					data := slices.Clone(orig)
					b.ReportAllocs()
					for b.Loop() {
						Implementation{}.Strsm(blas.Left, ul, trans, blas.NonUnit, m, n, 1, a, m, data, n)
						b.StopTimer()
						copy(data, orig)
						b.StartTimer()
					}
				})
			}
		}
	}
}
