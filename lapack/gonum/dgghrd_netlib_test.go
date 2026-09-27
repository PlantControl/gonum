// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDgghrdNetlibDifferential(t *testing.T) {
	const n = 6
	rnd := rand.New(rand.NewPCG(17, 17))
	aOrig := make([]float64, n*n)
	bOrig := make([]float64, n*n)
	for i := range aOrig {
		aOrig[i] = rnd.NormFloat64()
		bOrig[i] = rnd.NormFloat64()
	}
	comps := []struct {
		g lapack.OrthoComp
		n byte
	}{
		{g: lapack.OrthoNone, n: 'N'},
		{g: lapack.OrthoExplicit, n: 'I'},
		{g: lapack.OrthoPostmul, n: 'V'},
	}
	for _, active := range []struct {
		name     string
		ilo, ihi int
	}{
		{name: "Full", ilo: 0, ihi: n - 1},
		{name: "Interior", ilo: 1, ihi: n - 2},
	} {
		for _, compq := range comps {
			for _, compz := range comps {
				name := fmt.Sprintf("%s/Q=%c/Z=%c", active.name, compq.n, compz.n)
				t.Run(name, func(t *testing.T) {
					ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					gq, gz := identityData(n), identityData(n)
					nq, nz := identityData(n), identityData(n)
					Implementation{}.Dgghrd(compq.g, compz.g, n, active.ilo, active.ihi,
						ga, n, gb, n, gq, n, gz, n)
					info := netlib.Dgghrd(compq.n, compz.n, n, active.ilo, active.ihi,
						na, nb, nq, nz)
					if info != 0 {
						t.Fatalf("Netlib info=%d", info)
					}
					for i := range ga {
						checkCloseNetlib(t, fmt.Sprintf("A[%d]", i), ga[i], na[i])
						checkCloseNetlib(t, fmt.Sprintf("B[%d]", i), gb[i], nb[i])
						if compq.g != lapack.OrthoNone {
							checkCloseNetlib(t, fmt.Sprintf("Q[%d]", i), gq[i], nq[i])
						}
						if compz.g != lapack.OrthoNone {
							checkCloseNetlib(t, fmt.Sprintf("Z[%d]", i), gz[i], nz[i])
						}
					}
				})
			}
		}
	}
}

func TestDgghrdNetlibBatchBoundaries(t *testing.T) {
	const guard = 0x1.23456789abcdefp+100
	tests := []struct {
		n            int
		interior     bool
		compq, compz lapack.OrthoComp
		nq, nz       byte
	}{
		{31, false, lapack.OrthoExplicit, lapack.OrthoExplicit, 'I', 'I'},
		{32, true, lapack.OrthoPostmul, lapack.OrthoPostmul, 'V', 'V'},
		{33, false, lapack.OrthoNone, lapack.OrthoNone, 'N', 'N'},
		{63, true, lapack.OrthoPostmul, lapack.OrthoNone, 'V', 'N'},
		{64, false, lapack.OrthoNone, lapack.OrthoPostmul, 'N', 'V'},
		{65, true, lapack.OrthoExplicit, lapack.OrthoNone, 'I', 'N'},
		{128, true, lapack.OrthoNone, lapack.OrthoExplicit, 'N', 'I'},
	}
	for _, test := range tests {
		name := fmt.Sprintf("n=%d/interior=%t/Q=%c/Z=%c", test.n, test.interior, test.nq, test.nz)
		t.Run(name, func(t *testing.T) {
			n := test.n
			lda, ldb, ldq, ldz := n+3, n+5, n+2, n+4
			a := guardedMatrix(n, lda, guard)
			b := guardedMatrix(n, ldb, guard)
			q := guardedMatrix(n, ldq, guard)
			z := guardedMatrix(n, ldz, guard)
			rnd := rand.New(rand.NewPCG(uint64(n), 0xd66d687264))
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					a[i*lda+j] = rnd.NormFloat64()
					if j >= i {
						b[i*ldb+j] = rnd.NormFloat64()
					} else {
						b[i*ldb+j] = 0
					}
				}
			}
			qSeed := dgghrdOrthogonalSeed(n, 0.19)
			zSeed := dgghrdOrthogonalSeed(n, -0.27)
			setCompactMatrix(q, ldq, qSeed, n)
			setCompactMatrix(z, ldz, zSeed, n)

			na := compactMatrix(a, lda, n)
			nb := compactMatrix(b, ldb, n)
			nq := append([]float64(nil), qSeed...)
			nz := append([]float64(nil), zSeed...)
			ilo, ihi := 0, n-1
			if test.interior {
				ilo, ihi = 2, n-3
			}
			Implementation{}.Dgghrd(test.compq, test.compz, n, ilo, ihi,
				a, lda, b, ldb, q, ldq, z, ldz)
			info := netlib.Dgghrd(test.nq, test.nz, n, ilo, ihi, na, nb, nq, nz)
			if info != 0 {
				t.Fatalf("Netlib info=%d", info)
			}
			checkDgghrdMatrix(t, "A", a, lda, na, n, guard)
			checkDgghrdMatrix(t, "B", b, ldb, nb, n, guard)
			checkDgghrdMatrix(t, "Q", q, ldq, nq, n, guard)
			checkDgghrdMatrix(t, "Z", z, ldz, nz, n, guard)
		})
	}
}

func guardedMatrix(n, stride int, guard float64) []float64 {
	a := make([]float64, n*stride)
	for i := range a {
		a[i] = guard
	}
	return a
}

func compactMatrix(a []float64, stride, n int) []float64 {
	dst := make([]float64, n*n)
	for i := 0; i < n; i++ {
		copy(dst[i*n:(i+1)*n], a[i*stride:i*stride+n])
	}
	return dst
}

func setCompactMatrix(dst []float64, stride int, src []float64, n int) {
	for i := 0; i < n; i++ {
		copy(dst[i*stride:i*stride+n], src[i*n:(i+1)*n])
	}
}

func dgghrdOrthogonalSeed(n int, angle float64) []float64 {
	q := identityData(n)
	for i := 0; i < n-1; i++ {
		c, s := math.Cos(angle+float64(i%5)*0.03), math.Sin(angle+float64(i%5)*0.03)
		for row := 0; row < n; row++ {
			x, y := q[row*n+i], q[row*n+i+1]
			q[row*n+i] = c*x + s*y
			q[row*n+i+1] = c*y - s*x
		}
	}
	return q
}

func checkDgghrdMatrix(t *testing.T, name string, got []float64, stride int, want []float64, n int, guard float64) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := got[i*stride+j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s[%d,%d]=%v", name, i, j, v)
			}
			checkCloseNetlib(t, fmt.Sprintf("%s[%d,%d]", name, i, j), v, want[i*n+j])
		}
		for j := n; j < stride; j++ {
			if got[i*stride+j] != guard {
				t.Fatalf("%s padding[%d,%d]=%g, want guard %g", name, i, j, got[i*stride+j], guard)
			}
		}
	}
}
