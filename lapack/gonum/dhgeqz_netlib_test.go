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

func TestDhgeqzNetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(13, 13))
	for n := 2; n <= 10; n++ {
		for k := 0; k < 40; k++ {
			h := make([]float64, n*n)
			tt := make([]float64, n*n)
			for i := range n {
				for j := max(0, i-1); j < n; j++ {
					h[i*n+j] = rnd.NormFloat64()
				}
				for j := i; j < n; j++ {
					tt[i*n+j] = rnd.NormFloat64()
				}
				tt[i*n+i] += float64(n)
			}
			t.Run(fmt.Sprintf("n=%d/case=%d", n, k), func(t *testing.T) {
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
					n, 0, n-1, h, tt, true)
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone,
					n, 0, n-1, h, tt, false)
			})
		}
	}
}

func TestDhgeqzNetlibFailureLeavesLeadingEigenvalues(t *testing.T) {
	const n = 4
	h := []float64{
		2, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 2, 1,
		0, 0, math.NaN(), 3,
	}
	tt := []float64{
		-1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar := []float64{11, 12, 13, 14}
	gai := []float64{21, 22, 23, 24}
	gbeta := []float64{31, 32, 33, 34}
	nar := append([]float64(nil), gar...)
	nai := append([]float64(nil), gai...)
	nbeta := append([]float64(nil), gbeta...)
	work := make([]float64, n)
	gok := Implementation{}.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone,
		n, 1, n-1, gh, n, gt, n, gar, gai, gbeta, nil, 1, nil, 1, work, len(work))
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	info := netlib.Dhgeqz(byte(lapack.EigenvaluesOnly), byte(lapack.SchurNone), byte(lapack.SchurNone),
		n, 1, n-1, nh, nt, nar, nai, nbeta, nq, nz)
	if gok || info <= 0 {
		t.Fatalf("failure mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if gh[0] != nh[0] || gt[0] != nt[0] || gar[0] != nar[0] || gai[0] != nai[0] || gbeta[0] != nbeta[0] {
		t.Fatalf("leading failure output mismatch: Gonum H=%g T=%g alpha=(%g,%g) beta=%g; Netlib H=%g T=%g alpha=(%g,%g) beta=%g",
			gh[0], gt[0], gar[0], gai[0], gbeta[0], nh[0], nt[0], nar[0], nai[0], nbeta[0])
	}
}

func TestDhgeqzNetlibComplexRepresentation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		h, tt []float64
	}{
		{
			name: "NonDiagonalT",
			h:    []float64{0, -1, 1, 0},
			tt:   []float64{2, 1, 0, 3},
		},
		{
			name: "SeparatedDiagonals",
			h:    []float64{4, -7, 3, 2},
			tt:   []float64{1e-100, 2e-100, 0, 4e-100},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const n = 2
			gh, gt := append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...)
			nh, nt := append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...)
			gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
			nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
			gq, gz := make([]float64, n*n), make([]float64, n*n)
			nq, nz := make([]float64, n*n), make([]float64, n*n)
			work := make([]float64, n)
			gok := Implementation{}.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
				n, 0, n-1, gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
			info := netlib.Dhgeqz(byte(lapack.EigenvaluesAndSchur), byte(lapack.SchurHess), byte(lapack.SchurHess),
				n, 0, n-1, nh, nt, nar, nai, nbeta, nq, nz)
			if !gok || info != 0 {
				t.Fatalf("Gonum success=%v Netlib info=%d", gok, info)
			}
			for i := range n {
				for _, values := range [][2]float64{{gar[i], nar[i]}, {gai[i], nai[i]}, {gbeta[i], nbeta[i]}} {
					tol := 2e-13*math.Abs(values[1]) + 1e-300
					if math.Abs(values[0]-values[1]) > tol {
						t.Fatalf("triplet representation mismatch at %d: Gonum=(%v,%v)/%v Netlib=(%v,%v)/%v",
							i, gar, gai, gbeta, nar, nai, nbeta)
					}
				}
			}
		})
	}
}

func TestDhgeqzNetlibActiveSubrange(t *testing.T) {
	const n = 6
	h := []float64{
		-2, 1, 0, 0, 0, 0,
		0, 1, 2, 0.5, 0, 0,
		0, -1, 3, 1, 0.25, 0,
		0, 0, 0.75, -1, 2, 0.5,
		0, 0, 0, -0.5, 4, 1,
		0, 0, 0, 0, 0, 7,
	}
	tt := []float64{
		2, 0, 0, 0, 0, 0,
		0, 3, 0.5, 0, 0, 0,
		0, 0, 4, 0.25, 0, 0,
		0, 0, 0, 5, 0.5, 0,
		0, 0, 0, 0, 6, 0,
		0, 0, 0, 0, 0, 8,
	}
	compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
		n, 1, 4, h, tt, true)
}

func TestDhgeqzNetlibAccumulationOptions(t *testing.T) {
	const n = 5
	rnd := rand.New(rand.NewPCG(19, 19))
	hOrig := make([]float64, n*n)
	tOrig := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := max(0, i-1); j < n; j++ {
			hOrig[i*n+j] = rnd.NormFloat64()
		}
		for j := i; j < n; j++ {
			tOrig[i*n+j] = rnd.NormFloat64()
		}
		tOrig[i*n+i] += n
	}
	qOrig := identityData(n)
	qOrig[0], qOrig[1], qOrig[n], qOrig[n+1] = 0.6, -0.8, 0.8, 0.6
	zOrig := identityData(n)
	zOrig[2*n+2], zOrig[2*n+3], zOrig[3*n+2], zOrig[3*n+3] = 0.8, -0.6, 0.6, 0.8
	for _, compq := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess, lapack.SchurOrig} {
		for _, compz := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess, lapack.SchurOrig} {
			t.Run(fmt.Sprintf("Q=%c/Z=%c", compq, compz), func(t *testing.T) {
				gh, gt := append([]float64(nil), hOrig...), append([]float64(nil), tOrig...)
				nh, nt := append([]float64(nil), hOrig...), append([]float64(nil), tOrig...)
				gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				gq, nq := append([]float64(nil), qOrig...), append([]float64(nil), qOrig...)
				gz, nz := append([]float64(nil), zOrig...), append([]float64(nil), zOrig...)
				if compq == lapack.SchurHess {
					for i := range gq {
						gq[i], nq[i] = math.NaN(), math.NaN()
					}
				}
				if compz == lapack.SchurHess {
					for i := range gz {
						gz[i], nz[i] = math.NaN(), math.NaN()
					}
				}
				gok := Implementation{}.Dhgeqz(lapack.EigenvaluesAndSchur, compq, compz,
					n, 0, n-1, gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n,
					make([]float64, n), n)
				info := netlib.Dhgeqz(byte(lapack.EigenvaluesAndSchur), byte(compq), byte(compz),
					n, 0, n-1, nh, nt, nar, nai, nbeta, nq, nz)
				if gok != (info == 0) {
					t.Fatalf("Gonum success=%v, Netlib info=%d", gok, info)
				}
				if !gok {
					return
				}
				compareGeneralizedEigenvalues(t, gar, gai, gbeta, nar, nai, nbeta)
				for i := range gh {
					checkCloseNetlib(t, fmt.Sprintf("H[%d]", i), gh[i], nh[i])
					checkCloseNetlib(t, fmt.Sprintf("T[%d]", i), gt[i], nt[i])
					if compq != lapack.SchurNone {
						checkCloseNetlib(t, fmt.Sprintf("Q[%d]", i), gq[i], nq[i])
					}
					if compz != lapack.SchurNone {
						checkCloseNetlib(t, fmt.Sprintf("Z[%d]", i), gz[i], nz[i])
					}
				}
			})
		}
	}
}

func compareDhgeqzWithNetlib(t *testing.T, job lapack.SchurJob, compq, compz lapack.SchurComp,
	n, ilo, ihi int, h, tt []float64, checkSchur bool) {
	t.Helper()
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	gq, gz := make([]float64, n*n), make([]float64, n*n)
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, n)
	gok := Implementation{}.Dhgeqz(job, compq, compz, n, ilo, ihi,
		gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
	info := netlib.Dhgeqz(byte(job), byte(compq), byte(compz), n, ilo, ihi,
		nh, nt, nar, nai, nbeta, nq, nz)
	if gok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if !gok {
		return
	}
	compareGeneralizedEigenvalues(t, gar, gai, gbeta, nar, nai, nbeta)
	if checkSchur {
		checkGeneralizedSchurResult(t, "Gonum DHGEQZ", h, tt, gh, gt, gq, gz, n)
		checkGeneralizedSchurResult(t, "Netlib DHGEQZ", h, tt, nh, nt, nq, nz, n)
	}
}

// TestDhgeqzNetlibZeroDiagonalT compares Dhgeqz with Netlib on pencils with
// exactly zero or negative diagonal elements of T, which exercise the
// infinite-eigenvalue chases (DHGEQZ labels 40, 50 and 70), the ILAZR2 test
// (without taking its chase; see TestDhgeqzNetlibIlazr2Chase), and the sign
// normalization of T outside and inside the active block.
func TestDhgeqzNetlibZeroDiagonalT(t *testing.T) {
	rnd := rand.New(rand.NewPCG(31, 31))
	for n := 2; n <= 8; n++ {
		for k := 0; k < 60; k++ {
			h := make([]float64, n*n)
			tt := make([]float64, n*n)
			for i := range n {
				for j := max(0, i-1); j < n; j++ {
					h[i*n+j] = rnd.NormFloat64()
				}
				for j := i; j < n; j++ {
					tt[i*n+j] = rnd.NormFloat64()
				}
			}
			ilo, ihi := 0, n-1
			if k%4 == 3 && n > 2 {
				ilo, ihi = 1, n-2
				h[ilo*n+ilo-1] = 0
				h[(ihi+1)*n+ihi] = 0
			}
			j := ilo + rnd.IntN(ihi-ilo+1)
			switch k % 6 {
			case 0:
				tt[ilo*n+ilo] = 0
			case 1:
				tt[ihi*n+ihi] = 0
			case 2:
				tt[j*n+j] = 0
			case 3:
				tt[j*n+j] = 0
				if j < ihi {
					tt[(j+1)*n+j+1] = 0
				}
			case 4:
				if j < ihi {
					tt[(j+1)*n+j+1] = 0
					if j+2 <= ihi {
						h[(j+2)*n+j+1] = 1e-18
					}
				}
			case 5:
				tt[j*n+j] = 0
				tt[ilo*n+ilo] = 0
			}
			t.Run(fmt.Sprintf("n=%d/case=%d", n, k), func(t *testing.T) {
				for _, c := range []struct {
					job          lapack.SchurJob
					compq, compz lapack.SchurComp
					schur        bool
				}{
					{lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, true},
					{lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, false},
				} {
					compareDhgeqzWithNetlib(t, c.job, c.compq, c.compz, n, ilo, ihi, h, tt, c.schur)
					compareDhgeqzInOrder(t, c.job, c.compq, c.compz, n, ilo, ihi, h, tt)
				}
			})
		}
	}
}

// TestDhgeqzNetlibExceptionalShift compares Dhgeqz with Netlib on cyclic
// pencils whose standard shifts stall, which forces the exceptional shift.
func TestDhgeqzNetlibExceptionalShift(t *testing.T) {
	for n := 3; n <= 6; n++ {
		for _, tscale := range []float64{1, 1e-300} {
			h := make([]float64, n*n)
			tt := make([]float64, n*n)
			for i := 1; i < n; i++ {
				h[i*n+i-1] = 1
			}
			h[n-1] = 1
			for i := range n {
				tt[i*n+i] = 1
			}
			tt[(n-2)*n+n-2] = tscale
			t.Run(fmt.Sprintf("n=%d/tscale=%g", n, tscale), func(t *testing.T) {
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1, h, tt, true)
				compareDhgeqzInOrder(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1, h, tt)
			})
		}
	}
}

// compareDhgeqzInOrder compares alphar, alphai, beta and, for job S, the
// diagonals of S and T elementwise with Netlib.
func compareDhgeqzInOrder(t *testing.T, job lapack.SchurJob, compq, compz lapack.SchurComp,
	n, ilo, ihi int, h, tt []float64) {
	t.Helper()
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	gq, gz := make([]float64, n*n), make([]float64, n*n)
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, n)
	gok := Implementation{}.Dhgeqz(job, compq, compz, n, ilo, ihi,
		gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
	info := netlib.Dhgeqz(byte(job), byte(compq), byte(compz), n, ilo, ihi,
		nh, nt, nar, nai, nbeta, nq, nz)
	if !gok || info != 0 {
		return
	}
	for j := range n {
		checkCloseNetlib(t, fmt.Sprintf("job=%c alphar[%d]", job, j), gar[j], nar[j])
		checkCloseNetlib(t, fmt.Sprintf("job=%c alphai[%d]", job, j), gai[j], nai[j])
		checkCloseNetlib(t, fmt.Sprintf("job=%c beta[%d]", job, j), gbeta[j], nbeta[j])
		if job == lapack.EigenvaluesAndSchur {
			checkCloseNetlib(t, fmt.Sprintf("S[%d,%d]", j, j), gh[j*n+j], nh[j*n+j])
			checkCloseNetlib(t, fmt.Sprintf("T[%d,%d]", j, j), gt[j*n+j], nt[j*n+j])
		}
	}
}

// TestDhgeqzNetlibNaNTrailingDiagonal checks that a NaN T(j,j) outside the
// active block is not sign-normalized, since DHGEQZ negates only when
// T(j,j) < 0.
func TestDhgeqzNetlibNaNTrailingDiagonal(t *testing.T) {
	const n = 3
	h := []float64{1, 2, 3, 4, 5, 6, 0, 0, 7}
	tt := []float64{2, 1, -1, 0, 3, 2, 0, 0, math.NaN()}
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	gq, gz := make([]float64, n*n), make([]float64, n*n)
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, n)
	Implementation{}.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, 1,
		gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
	netlib.Dhgeqz(byte(lapack.EigenvaluesAndSchur), byte(lapack.SchurHess), byte(lapack.SchurHess), n, 0, 1,
		nh, nt, nar, nai, nbeta, nq, nz)
	for i := range n {
		checkCloseNetlib(t, fmt.Sprintf("H[%d,2]", i), gh[i*n+2], nh[i*n+2])
		checkCloseNetlib(t, fmt.Sprintf("Z[%d,2]", i), gz[i*n+2], nz[i*n+2])
	}
}

// TestDhgeqzNetlibIlazr2Chase compares Dhgeqz with Netlib on pencils where
// T(j,j) = 0 for an interior j and H(j,j-1)·H(j+1,j) is negligible although
// neither factor is, so DHGEQZ takes the ILAZR2 chase and scales H(j,j-1) by
// the first rotation's cosine.
func TestDhgeqzNetlibIlazr2Chase(t *testing.T) {
	for n := 4; n <= 6; n++ {
		for j := 1; j < n-1; j++ {
			rnd := rand.New(rand.NewPCG(uint64(n), uint64(j)))
			h := make([]float64, n*n)
			tt := make([]float64, n*n)
			for r := range n {
				for c := max(0, r-1); c < n; c++ {
					h[r*n+c] = rnd.NormFloat64()
				}
				for c := r; c < n; c++ {
					tt[r*n+c] = rnd.NormFloat64()
				}
			}
			h[j*n+j] = 1
			h[j*n+j-1] = 1e-9
			h[(j+1)*n+j] = 1e-9
			tt[j*n+j] = 0
			for _, job := range []lapack.SchurJob{lapack.EigenvaluesOnly, lapack.EigenvaluesAndSchur} {
				t.Run(fmt.Sprintf("n=%d/j=%d/job=%c", n, j, job), func(t *testing.T) {
					compq, compz := lapack.SchurNone, lapack.SchurNone
					if job == lapack.EigenvaluesAndSchur {
						compq, compz = lapack.SchurHess, lapack.SchurHess
					}
					compareDhgeqzWithNetlib(t, job, compq, compz, n, 0, n-1, h, tt, job == lapack.EigenvaluesAndSchur)
					compareDhgeqzInOrder(t, job, compq, compz, n, 0, n-1, h, tt)
				})
			}
		}
	}
}

// TestDhgeqzNetlibExceptionalShiftAccumulates compares Dhgeqz with Netlib on
// pencils with T below 1e-299, where at every tenth iteration
// MAXIT·SAFMIN·|H(ilast,ilast-1)| >= |T(ilast-1,ilast-1)| and DHGEQZ adds
// 1/(SAFMIN·MAXIT) to a nonzero exceptional shift instead of resetting it.
// Both pencils, found by search, converge only with the accumulated shift.
func TestDhgeqzNetlibExceptionalShiftAccumulates(t *testing.T) {
	for i, tc := range []struct {
		n     int
		h, tt []float64
	}{
		{
			n: 6,
			h: []float64{
				0, 0, -0.3339221563805084, 0, 0, 0,
				-0.270437872829134, -0.363382047715767, 0, 0, 0.6573028634910769, 0,
				0, 0.7686298848642166, 0.35030660177222694, 1.1088609654208472, 0.39244671047755264, 0.3162493535093063,
				0, 0, -0.09044119156898656, 0, 0, 0,
				0, 0, 0, 1.4247991405436191, 0, 0,
				0, 0, 0, 0, 0.3478946666536997, 1.6628024715335878,
			},
			tt: []float64{
				5.475741653041993e-306, 2.2100263633282568e-301, 0, 0, 2.438052272012528e-301, 0,
				0, 4.39230735885567e-303, 0, 0, -1.448971790859472e-300, 1.3684408849496923e-300,
				0, 0, 9.348023413066877e-308, 0, 0, 0,
				0, 0, 0, 4.4161506371172714e-306, -6.281984711514761e-301, 0,
				0, 0, 0, 0, 1.0723566230310677e-302, 0,
				0, 0, 0, 0, 0, 4.3396527351449675e-304,
			},
		},
		{
			n: 7,
			h: []float64{
				0, 0, 0, 0.08660867690912477, 0, 0.9761290937818051, 0,
				-0.07889349703556148, 0, 0, 0, 0.5139935474199059, 0, 0,
				0, 0.27555035937322014, 0, 0, 1.0088645367725235, 2.0388056301148283, -0.19269250492734802,
				0, 0, 0.038040129283957724, 0, 0.5216291563002307, 0, 0,
				0, 0, 0, 0.055779621209777835, 0, 0, 0,
				0, 0, 0, 0, -1.9199455779038987, 0.23043801737604053, 0.8110876276247365,
				0, 0, 0, 0, 0, -0.5227962976602188, 0,
			},
			tt: []float64{
				4.4941279880815924e-304, 0, 0, 0, 0, -1.9462925265433537e-300, 0,
				0, 3.1548064250429e-307, 0, 0, 0, 6.484629733303438e-302, 0,
				0, 0, 1.9759246659979776e-304, 0, 0, -6.0759929658947725e-301, -5.458220956477567e-301,
				0, 0, 0, 1.3683924429140707e-304, 1.7932417749156881e-301, 0, 0,
				0, 0, 0, 0, 7.999005893078747e-304, 0, 0,
				0, 0, 0, 0, 0, 7.882711061902595e-307, 0,
				0, 0, 0, 0, 0, 0, 1.5227284604606824e-307,
			},
		},
	} {
		for _, job := range []lapack.SchurJob{lapack.EigenvaluesOnly, lapack.EigenvaluesAndSchur} {
			t.Run(fmt.Sprintf("case=%d/job=%c", i, job), func(t *testing.T) {
				compq, compz := lapack.SchurNone, lapack.SchurNone
				if job == lapack.EigenvaluesAndSchur {
					compq, compz = lapack.SchurHess, lapack.SchurHess
				}
				n := tc.n
				gh, gt := append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...)
				gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
				gok := Implementation{}.Dhgeqz(job, compq, compz, n, 0, n-1, gh, n, gt, n,
					gar, gai, gbeta, make([]float64, n*n), n, make([]float64, n*n), n, make([]float64, n), n)
				info := netlib.Dhgeqz(byte(job), byte(compq), byte(compz), n, 0, n-1,
					append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...),
					nar, nai, nbeta, make([]float64, n*n), make([]float64, n*n))
				if !gok || info != 0 {
					t.Fatalf("ok=%v, Netlib info=%d, want both to converge", gok, info)
				}
				// The pencils have eigenvalues near 0/0, so compare alpha and
				// beta to backward-error accuracy rather than their ratios.
				hnorm := Implementation{}.Dlange(lapack.Frobenius, n, n, tc.h, n, nil)
				tnorm := Implementation{}.Dlange(lapack.Frobenius, n, n, tc.tt, n, nil)
				for j := range n {
					if math.Abs(gar[j]-nar[j]) > 1e-12*hnorm || math.Abs(gai[j]-nai[j]) > 1e-12*hnorm ||
						math.Abs(gbeta[j]-nbeta[j]) > 1e-12*tnorm {
						t.Errorf("eigenvalue %d: (%g,%g)/%g, want Netlib (%g,%g)/%g",
							j, gar[j], gai[j], gbeta[j], nar[j], nai[j], nbeta[j])
					}
				}
			})
		}
	}
}
