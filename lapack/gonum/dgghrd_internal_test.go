// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

func TestDgghrdBatchedVectors(t *testing.T) {
	comps := []lapack.OrthoComp{lapack.OrthoNone, lapack.OrthoExplicit, lapack.OrthoPostmul}
	for _, n := range []int{31, 32, 33, 63, 64, 65, 128} {
		for _, extra := range []int{0, 3} {
			stride := n + extra
			for _, compq := range comps {
				for _, compz := range comps {
					name := string([]byte{byte(compq), byte(compz)})
					t.Run(fmt.Sprintf("%s/n=%d/extra=%d", name, n, extra), func(t *testing.T) {
						a, b, q, z := dgghrdBatchedInputs(n, stride, compq, compz)
						wantA := append([]float64(nil), a...)
						wantB := append([]float64(nil), b...)
						wantQ := append([]float64(nil), q...)
						wantZ := append([]float64(nil), z...)
						dgghrdImmediate(compq, compz, n, 0, n-1, wantA, stride, wantB, stride, wantQ, stride, wantZ, stride)
						Implementation{}.Dgghrd(compq, compz, n, 0, n-1, a, stride, b, stride, q, stride, z, stride)
						checkDgghrdBatchedMatrix(t, "A", n, stride, a, wantA)
						checkDgghrdBatchedMatrix(t, "B", n, stride, b, wantB)
						if compq != lapack.OrthoNone {
							checkDgghrdBatchedMatrix(t, "Q", n, stride, q, wantQ)
						}
						if compz != lapack.OrthoNone {
							checkDgghrdBatchedMatrix(t, "Z", n, stride, z, wantZ)
						}
					})
				}
			}
		}
	}
}

func TestDgghrdBatchedPartialRange(t *testing.T) {
	const n, stride = 65, 68
	a, b, q, z := dgghrdBatchedInputs(n, stride, lapack.OrthoPostmul, lapack.OrthoPostmul)
	wantA := append([]float64(nil), a...)
	wantB := append([]float64(nil), b...)
	wantQ := append([]float64(nil), q...)
	wantZ := append([]float64(nil), z...)
	dgghrdImmediate(lapack.OrthoPostmul, lapack.OrthoPostmul, n, 7, 58, wantA, stride, wantB, stride, wantQ, stride, wantZ, stride)
	Implementation{}.Dgghrd(lapack.OrthoPostmul, lapack.OrthoPostmul, n, 7, 58, a, stride, b, stride, q, stride, z, stride)
	checkDgghrdBatchedMatrix(t, "A", n, stride, a, wantA)
	checkDgghrdBatchedMatrix(t, "B", n, stride, b, wantB)
	checkDgghrdBatchedMatrix(t, "Q", n, stride, q, wantQ)
	checkDgghrdBatchedMatrix(t, "Z", n, stride, z, wantZ)
}

func TestDgghrdCustomBLASUsesImmediateRotations(t *testing.T) {
	const n = 32
	old := blas64.Implementation()
	recorder := &dgghrdRecordingBLAS{Float64: old}
	blas64.Use(recorder)
	defer blas64.Use(old)

	a, b, q, z := dgghrdBatchedInputs(n, n, lapack.OrthoExplicit, lapack.OrthoExplicit)
	Implementation{}.Dgghrd(lapack.OrthoExplicit, lapack.OrthoExplicit, n, 0, n-1, a, n, b, n, q, n, z, n)
	rotations := (n - 1) * (n - 2) / 2
	if want := 6 * rotations; recorder.calls != want {
		t.Fatalf("custom BLAS received %d Drot calls, want %d", recorder.calls, want)
	}
}

func TestDgghrdReplayVectorsExceptional(t *testing.T) {
	const first, count, stride = 5, 4, 6
	values := []float64{
		0, math.Copysign(0, -1), 2, -3, math.Inf(1), math.NaN(),
		-1, 4, math.SmallestNonzeroFloat64, 0, math.Inf(-1), 7,
		2, -5, 8, math.Copysign(0, -1), 1, -2,
	}
	for _, n := range []int{3, 4, 5, 8} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			q := make([]float64, n*stride)
			for i := range q {
				q[i] = values[i%len(values)]
			}
			z := append([]float64(nil), q...)
			wantQ := append([]float64(nil), q...)
			wantZ := append([]float64(nil), z...)
			c := []float64{1, 0, math.Inf(1), math.NaN()}
			s := []float64{math.Copysign(0, -1), 1, math.Inf(-1), 0}
			bi := blas64.Implementation()
			for k := 0; k < count; k++ {
				j := first - k
				bi.Drot(n, wantQ[j-1:], stride, wantQ[j:], stride, c[k], s[k])
				bi.Drot(n, wantZ[j:], stride, wantZ[j-1:], stride, c[k], s[k])
			}
			dgghrdReplayVectors(n, first, count,
				lapack.OrthoPostmul, q, stride, c, s,
				lapack.OrthoPostmul, z, stride, c, s)
			checkDgghrdReplayBits(t, "Q", q, wantQ)
			checkDgghrdReplayBits(t, "Z", z, wantZ)
		})
	}
}

func checkDgghrdReplayBits(t *testing.T, name string, got, want []float64) {
	t.Helper()
	for i, w := range want {
		g := got[i]
		if math.IsNaN(g) && math.IsNaN(w) {
			continue
		}
		if math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("%s[%d]=%v (%#x) want %v (%#x)", name, i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

type dgghrdRecordingBLAS struct {
	blas.Float64
	calls int
}

func (r *dgghrdRecordingBLAS) Drot(n int, x []float64, incX int, y []float64, incY int, c, s float64) {
	r.calls++
	r.Float64.Drot(n, x, incX, y, incY, c, s)
}

func dgghrdBatchedInputs(n, stride int, compq, compz lapack.OrthoComp) (a, b, q, z []float64) {
	a = make([]float64, n*stride)
	b = make([]float64, n*stride)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*stride+j] = float64((i*19+j*7)%31-15) / 8
			if j >= i {
				b[i*stride+j] = float64((i*5+j*13)%23-11) / 8
			}
		}
		b[i*stride+i] += float64(n)
		for j := n; j < stride; j++ {
			a[i*stride+j] = math.NaN()
			b[i*stride+j] = math.NaN()
		}
	}
	if compq != lapack.OrthoNone {
		q = make([]float64, n*stride)
		for i := 0; i < n; i++ {
			for j := n; j < stride; j++ {
				q[i*stride+j] = math.NaN()
			}
		}
	}
	if compz != lapack.OrthoNone {
		z = make([]float64, n*stride)
		for i := 0; i < n; i++ {
			for j := n; j < stride; j++ {
				z[i*stride+j] = math.NaN()
			}
		}
	}
	if compq == lapack.OrthoPostmul {
		for i := 0; i < n; i++ {
			q[i*stride+i] = 1
		}
	}
	if compz == lapack.OrthoPostmul {
		for i := 0; i < n; i++ {
			z[i*stride+i] = 1
		}
	}
	return a, b, q, z
}

func checkDgghrdBatchedMatrix(t *testing.T, name string, n, stride int, got, want []float64) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g, w := got[i*stride+j], want[i*stride+j]
			if math.IsNaN(g) != math.IsNaN(w) || !math.IsNaN(g) && math.Abs(g-w) > 2e-14*max(1, math.Abs(w)) {
				t.Fatalf("%s[%d,%d]=%v want %v", name, i, j, g, w)
			}
		}
		for j := n; j < stride; j++ {
			if !math.IsNaN(got[i*stride+j]) {
				t.Fatalf("%s padding modified at [%d,%d]", name, i, j)
			}
		}
	}
}

func dgghrdImmediate(compq, compz lapack.OrthoComp, n, ilo, ihi int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int) {
	impl := Implementation{}
	if compq == lapack.OrthoExplicit {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if compz == lapack.OrthoExplicit {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			b[i*ldb+j] = 0
		}
	}
	bi := blas64.Implementation()
	for jcol := ilo; jcol <= ihi-2; jcol++ {
		for jrow := ihi; jrow >= jcol+2; jrow-- {
			c, s, r := impl.Dlartg(a[(jrow-1)*lda+jcol], a[jrow*lda+jcol])
			a[(jrow-1)*lda+jcol] = r
			a[jrow*lda+jcol] = 0
			bi.Drot(n-jcol-1, a[(jrow-1)*lda+jcol+1:], 1, a[jrow*lda+jcol+1:], 1, c, s)
			bi.Drot(n+1-jrow, b[(jrow-1)*ldb+jrow-1:], 1, b[jrow*ldb+jrow-1:], 1, c, s)
			if compq != lapack.OrthoNone {
				bi.Drot(n, q[jrow-1:], ldq, q[jrow:], ldq, c, s)
			}
			c, s, r = impl.Dlartg(b[jrow*ldb+jrow], b[jrow*ldb+jrow-1])
			b[jrow*ldb+jrow] = r
			b[jrow*ldb+jrow-1] = 0
			bi.Drot(ihi+1, a[jrow:], lda, a[jrow-1:], lda, c, s)
			bi.Drot(jrow, b[jrow:], ldb, b[jrow-1:], ldb, c, s)
			if compz != lapack.OrthoNone {
				bi.Drot(n, z[jrow:], ldz, z[jrow-1:], ldz, c, s)
			}
		}
	}
}
