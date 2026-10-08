// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// dggesRejectedSwapFixture is a finite pencil whose eigenvalue reordering is
// rejected by the swap kernel in both Gonum and Reference LAPACK (checked
// against the Homebrew build, ILAVER 3.12.0).
//
// Construction: the input is already in generalized real Schur form, so
// QZ leaves it structurally intact. The leading 2×2 block is
// 2⁻⁶⁰·([1 1; -1 1], I) with eigenvalues 1±i, followed by the blocks
// listed in eigs, coupled by O(1) entries. All entries are exact binary
// fractions. The selector beta != 0 && alphar < 0 picks every block with
// a negative real part, so a selected block must be swapped past the
// 2⁻⁶⁰-scaled block. Its A and B parts both sit below eps relative to the
// neighbour, so the Kronecker system in Dtgsy2 has pivots below
// eps·max|Z| and Dgetc2 perturbs them; Dtgex2 rejects the swap.
// Dhgeqz first flushes the tiny B diagonal (|T(j,j)| ≤ ulp·‖B‖) so Dgges
// sees a finite real eigenvalue and an infinite one there, and the rejected
// swap is 1×1 against 2×2; Dtgsen called on the input directly rejects a 2×2
// against 2×2 swap. The margin is about 2.5 orders of magnitude below
// threshold.
//
// No random orthogonal back-transform is applied because any O(1)
// orthogonal mixing buries the 2⁻⁶⁰ block in roundoff. Swaps of blocks at
// comparable scale were not rejected in Reference LAPACK for eigenvalue
// gaps down to 1e-16 with coupling up to 1e14; only exactly equal
// eigenvalues or a scale split below eps were rejected. Such pencils are
// within O(eps) of one whose ordering is undefined, so a back-transformed
// fixture would depend on roundoff.
//
// Reference DGGES reports these as INFO=N+2, not N+3: after DTGSEN sets
// N+3 the selection recheck finds an unselected eigenvalue ahead of a
// selected one and overwrites INFO with N+2 (dgges.f lines 566-567 and
// 651-664 at v3.12.1). With a selector that depends only on the eigenvalue,
// a rejected swap always leaves such an inversion, so N+3 cannot be pinned
// deterministically through the driver; Dtgsen reports the rejection.
type dggesRejectedSwapFixture struct {
	name string
	n    int
	a, b []float64
	// selected marks the input eigenvalues chosen by the selector, for Dtgsen.
	selected []bool
	sdim     int
	// eigs are the finite eigenvalues expected at indices 2..n-1 of the
	// Dgges output after the partial reordering.
	eigs []complex128
	// splits lists the indices i with S[i,i-1] == 0 in the Dgges output.
	splits []int
}

func dggesRejectedSwapFixtures() []dggesRejectedSwapFixture {
	const s = 0x1p-60
	return []dggesRejectedSwapFixture{
		{
			name: "2x2-2x2",
			n:    4,
			a: []float64{
				s, s, 1, 0.5,
				-s, s, -0.25, 1,
				0, 0, -1, 1,
				0, 0, -1, -1,
			},
			b: []float64{
				s, 0, 0.5, 1,
				0, s, 1, -0.5,
				0, 0, 1, 0,
				0, 0, 0, 1,
			},
			selected: []bool{false, false, true, true},
			sdim:     2,
			eigs:     []complex128{complex(-1, 1), complex(-1, -1)},
			splits:   []int{1, 2},
		},
		{
			// The -1±i block first moves past the eigenvalue 2, then is
			// rejected at the tiny block; -3 is never moved.
			name: "partial-move",
			n:    6,
			a: []float64{
				s, s, 1, 0.5, 0.25, -1,
				-s, s, -0.25, 1, 0.5, 0.75,
				0, 0, 2, 1, -1, 0.5,
				0, 0, 0, -1, 1, 0.5,
				0, 0, 0, -1, -1, 1,
				0, 0, 0, 0, 0, -3,
			},
			b: []float64{
				s, 0, 0.5, 1, 0.25, 1,
				0, s, 1, -0.5, 0.5, -1,
				0, 0, 1, 0.5, 0.25, 0.5,
				0, 0, 0, 1, 0, 0.25,
				0, 0, 0, 0, 1, 0.5,
				0, 0, 0, 0, 0, 1,
			},
			selected: []bool{false, false, false, true, true, true},
			sdim:     3,
			eigs:     []complex128{complex(-1, 1), complex(-1, -1), 2, -3},
			splits:   []int{1, 2, 4, 5},
		},
	}
}

func TestDggesRejectedSwapFixtures(t *testing.T) {
	selector := func(ar, _, beta float64) bool { return beta != 0 && ar < 0 }
	for _, tc := range dggesRejectedSwapFixtures() {
		for _, vectors := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/vectors=%t", tc.name, vectors), func(t *testing.T) {
				n := tc.n
				jobvs := lapack.SchurNone
				if vectors {
					jobvs = lapack.SchurHess
				}
				a, b := append([]float64(nil), tc.a...), append([]float64(nil), tc.b...)
				ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
				vsl, vsr := make([]float64, n*n), make([]float64, n*n)
				query := make([]float64, 1)
				impl := Implementation{}
				impl.Dgges(jobvs, jobvs, lapack.SortSelected, selector, n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, query, -1, nil)
				work := make([]float64, int(query[0]))
				sdim, ok := impl.Dgges(jobvs, jobvs, lapack.SortSelected, selector, n, a, n, b, n,
					ar, ai, beta, vsl, n, vsr, n, work, len(work), make([]bool, n))
				if ok {
					t.Fatal("rejected reordering reported success")
				}
				if sdim != tc.sdim {
					t.Errorf("sdim=%d, want %d", sdim, tc.sdim)
				}
				if ai[0] != 0 || ai[1] != 0 || beta[0] <= 0 || ar[0] <= 0 || beta[1] != 0 {
					t.Errorf("tiny block eigenvalues (%g,%g,%g), (%g,%g,%g): want finite positive then infinite",
						ar[0], ai[0], beta[0], ar[1], ai[1], beta[1])
				}
				for k, want := range tc.eigs {
					i := k + 2
					got := complex(ar[i], ai[i]) / complex(beta[i], 0)
					if math.Abs(real(got-want))+math.Abs(imag(got-want)) > 1e-13 {
						t.Errorf("eigenvalue %d=%v, want %v", i, got, want)
					}
				}
				checkRejectedSwapStructure(t, a, b, n, tc.splits)
				if vectors {
					checkRejectedSwapResidual(t, tc.a, a, vsl, vsr, n)
					checkRejectedSwapResidual(t, tc.b, b, vsl, vsr, n)
				}
			})
		}
	}
}

func TestDtgsenRejectedSwapFixtures(t *testing.T) {
	for _, tc := range dggesRejectedSwapFixtures() {
		for _, vectors := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/vectors=%t", tc.name, vectors), func(t *testing.T) {
				n := tc.n
				a, b := append([]float64(nil), tc.a...), append([]float64(nil), tc.b...)
				ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
				q, z := make([]float64, n*n), make([]float64, n*n)
				for i := range n {
					q[i*n+i], z[i*n+i] = 1, 1
				}
				work := make([]float64, 4*n+16)
				iwork := make([]int, 1)
				m, _, _, _, ok := Implementation{}.Dtgsen(0, vectors, vectors, tc.selected, n,
					a, n, b, n, ar, ai, beta, q, n, z, n, work, len(work), iwork, len(iwork))
				if ok {
					t.Fatal("rejected reordering reported success")
				}
				if m != tc.sdim {
					t.Errorf("m=%d, want %d", m, tc.sdim)
				}
				for i := range 2 {
					if a[i*n+i] != tc.a[i*n+i] || b[i*n+i] != tc.b[i*n+i] {
						t.Errorf("tiny block diagonal modified at %d", i)
					}
				}
				checkRejectedSwapStructure(t, a, b, n, []int{2})
				if vectors {
					checkRejectedSwapResidual(t, tc.a, a, q, z, n)
					checkRejectedSwapResidual(t, tc.b, b, q, z, n)
				}
			})
		}
	}
}

func checkRejectedSwapStructure(t *testing.T, a, b []float64, n int, splits []int) {
	t.Helper()
	for i := range n {
		for j := range i {
			if b[i*n+j] != 0 {
				t.Errorf("B[%d,%d]=%g, want 0", i, j, b[i*n+j])
			}
			if j < i-1 && a[i*n+j] != 0 {
				t.Errorf("A[%d,%d]=%g, want 0", i, j, a[i*n+j])
			}
		}
	}
	for _, i := range splits {
		if a[i*n+i-1] != 0 {
			t.Errorf("A[%d,%d]=%g, want block boundary", i, i-1, a[i*n+i-1])
		}
	}
}

// checkRejectedSwapResidual checks ‖orig - Q·S·Zᵀ‖_max ≤ 100·eps·‖orig‖_max.
func checkRejectedSwapResidual(t *testing.T, orig, s, q, z []float64, n int) {
	t.Helper()
	norm := 0.0
	for _, v := range orig {
		norm = math.Max(norm, math.Abs(v))
	}
	for i := range n {
		for j := range n {
			sum := orig[i*n+j]
			for k := range n {
				for l := range n {
					sum -= q[i*n+k] * s[k*n+l] * z[j*n+l]
				}
			}
			if math.Abs(sum) > 100*dlamchE*norm {
				t.Fatalf("residual[%d,%d]=%g exceeds %g", i, j, sum, 100*dlamchE*norm)
			}
		}
	}
}
