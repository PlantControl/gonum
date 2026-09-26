// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dtgevc computes some or all of the right and/or left eigenvectors of a pair
// of real matrices (S, P), where S is a quasi-triangular matrix and P is upper
// triangular.
//
// The right eigenvector x and the left eigenvector y of (S, P) corresponding to
// an eigenvalue λ are defined by:
//
//	S * x = λ * P * x
//	y^H * S = λ * y^H * P
//
// The eigenvalues are not input to this routine but are determined from the
// diagonal blocks of S and P. For real eigenvalues (1x1 blocks), the eigenvector
// is real. For complex eigenvalue pairs (2x2 blocks), the eigenvectors are
// complex conjugate pairs stored as two consecutive real vectors.
//
// side specifies which eigenvectors to compute:
//
//	lapack.EVRight: right eigenvectors only
//	lapack.EVLeft: left eigenvectors only
//	lapack.EVBoth: both right and left eigenvectors
//
// howmny specifies how many eigenvectors to compute:
//
//	lapack.EVAll: all eigenvectors
//	lapack.EVAllMulQ: all eigenvectors, multiplied by input matrices
//	lapack.EVSelected: selected eigenvectors (indicated by selected[])
//
// selected must have length n when howmny is lapack.EVSelected. If selected[j]
// is true, the eigenvector corresponding to the j-th eigenvalue is computed.
// For complex pairs, only the first element of the pair needs to be selected.
//
// n is the order of the matrices S and P.
//
// S is an n×n quasi-upper triangular matrix. P is an n×n upper triangular matrix.
//
// vl and vr are the computed left and right eigenvector matrices.
// If howmny is lapack.EVAllMulQ, on entry vl/vr should contain matrices to
// multiply the eigenvectors by.
//
// mm is the number of columns in vl and/or vr. mm >= m (the number of
// eigenvectors to compute).
//
// work must have length at least 6*n.
//
// The return value m is the number of eigenvectors computed.
// ok is false if the eigenvector computation failed.
//
// Dtgevc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgevc(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int,
	s []float64, lds int, p []float64, ldp int, vl []float64, ldvl int, vr []float64, ldvr int,
	mm int, work []float64) (m int, ok bool) {

	compl := side == lapack.EVLeft || side == lapack.EVBoth
	compr := side == lapack.EVRight || side == lapack.EVBoth
	over := howmny == lapack.EVAllMulQ
	somev := howmny == lapack.EVSelected

	switch {
	case side != lapack.EVRight && side != lapack.EVLeft && side != lapack.EVBoth:
		panic(badEVSide)
	case howmny != lapack.EVAll && howmny != lapack.EVAllMulQ && howmny != lapack.EVSelected:
		panic(badEVHowMany)
	case n < 0:
		panic(nLT0)
	case lds < max(1, n):
		panic(badLdA)
	case ldp < max(1, n):
		panic(badLdB)
	case compl && ldvl < max(1, n):
		panic(badLdVL)
	case compr && ldvr < max(1, n):
		panic(badLdVR)
	}

	if n == 0 {
		return 0, true
	}

	switch {
	case len(s) < (n-1)*lds+n:
		panic(shortA)
	case len(p) < (n-1)*ldp+n:
		panic(shortB)
	case somev && len(selected) < n:
		panic("lapack: insufficient length of selected")
	case len(work) < 6*n:
		panic(shortWork)
	}

	// Count the number of eigenvectors to compute.
	if somev {
		m = 0
		for j := 0; j < n; j++ {
			if selected[j] {
				m++
			}
		}
	} else {
		m = n
	}

	switch {
	case mm < m:
		panic("lapack: mm < m")
	case compl && len(vl) < (n-1)*ldvl+mm:
		panic(shortVL)
	case compr && len(vr) < (n-1)*ldvr+mm:
		panic(shortVR)
	}

	if m == 0 {
		return 0, true
	}

	bi := blas64.Implementation()

	// Machine constants.
	safmin := dlamchS
	ulp := dlamchE
	small := safmin * float64(n) / ulp
	big := 1 / small
	bignum := 1 / (safmin * float64(n))

	// Compute right eigenvectors.
	if compr {
		im := m - 1
		for je := n - 1; je >= 0; {
			if somev && !selected[je] {
				je--
				continue
			}

			// Detect 2x2 block.
			nw := 1
			if je > 0 && s[je*lds+je-1] != 0 {
				nw = 2
			}

			if nw == 1 {
				// Real eigenvalue.
				// Compute coefficients a and b in (aS - bP)x = 0.
				acoef := 1.0
				bcoef := 1.0
				if math.Abs(s[je*lds+je]) > safmin {
					acoef = 1.0
					bcoef = s[je*lds+je] / p[je*ldp+je]
				} else if math.Abs(p[je*ldp+je]) > safmin {
					acoef = p[je*ldp+je] / s[je*lds+je]
					bcoef = 1.0
				}

				// Initialize eigenvector.
				for i := 0; i < n; i++ {
					work[i] = 0
					work[n+i] = 0
				}
				work[je] = 1

				// Back-substitute.
				for i := je - 1; i >= 0; i-- {
					if i > 0 && s[i*lds+i-1] != 0 {
						// Skip first row of 2x2 block (handle together).
						continue
					}

					na := 1
					if i < je-1 && i < n-1 && s[(i+1)*lds+i] != 0 {
						na = 2
					}

					// Compute right-hand side.
					if na == 1 {
						work[i] = -acoef * bi.Ddot(je-i, s[i*lds+i+1:], 1, work[i+1:], 1)
						work[i] += bcoef * bi.Ddot(je-i, p[i*ldp+i+1:], 1, work[i+1:], 1)
					} else {
						work[i] = -acoef * bi.Ddot(je-i, s[i*lds+i+1:], 1, work[i+1:], 1)
						work[i] += bcoef * bi.Ddot(je-i, p[i*ldp+i+1:], 1, work[i+1:], 1)
						work[i+1] = -acoef * bi.Ddot(je-i-1, s[(i+1)*lds+i+2:], 1, work[i+2:], 1)
						work[i+1] += bcoef * bi.Ddot(je-i-1, p[(i+1)*ldp+i+2:], 1, work[i+2:], 1)
					}

					// Solve the system.
					if na == 1 {
						denom := acoef*s[i*lds+i] - bcoef*p[i*ldp+i]
						if math.Abs(denom) < safmin {
							denom = safmin
						}
						work[i] /= denom
					} else {
						// 2x2 system: use Dlaln2.
						a2x2 := [4]float64{s[i*lds+i], s[i*lds+i+1], s[(i+1)*lds+i], s[(i+1)*lds+i+1]}
						b2x2 := [2]float64{work[i], work[i+1]}
						var x2x2 [2]float64
						scale, _, _ := impl.Dlaln2(false, 2, 1, small, acoef, a2x2[:], 2,
							bcoef*p[i*ldp+i], bcoef*p[(i+1)*ldp+i+1], b2x2[:], 1, bcoef, 0, x2x2[:], 1)
						work[i] = x2x2[0] / scale
						work[i+1] = x2x2[1] / scale
						i--
					}
				}

				// Copy to output and normalize.
				if over {
					bi.Dgemv(blas.NoTrans, n, je+1, 1, vr, ldvr, work[:je+1], 1, 0, work[2*n:], 1)
					for i := 0; i < n; i++ {
						vr[i*ldvr+im] = work[2*n+i]
					}
				} else {
					for i := 0; i < n; i++ {
						vr[i*ldvr+im] = work[i]
					}
				}

				// Normalize.
				xmax := 0.0
				for i := 0; i < n; i++ {
					xmax = math.Max(xmax, math.Abs(vr[i*ldvr+im]))
				}
				if xmax > safmin {
					bi.Dscal(n, 1/xmax, vr[im:], ldvr)
				}

				im--
				je--
			} else {
				// Complex eigenvalue pair.
				// Get eigenvalue from 2x2 block using Dlag2.
				_, _, acoefr, _, bcoefi := impl.Dlag2(s[(je-1)*lds+je-1:], lds, p[(je-1)*ldp+je-1:], ldp)
				acoef := 1.0
				bcoefr := acoefr
				if math.Abs(acoefr) > safmin || math.Abs(bcoefi) > safmin {
					scale := 1 / math.Max(math.Abs(acoefr), math.Abs(bcoefi))
					acoef = scale
					bcoefr = acoefr * scale
					bcoefi = bcoefi * scale
				}

				// Initialize eigenvector.
				for i := 0; i < n; i++ {
					work[i] = 0
					work[n+i] = 0
				}
				work[je-1] = 1
				work[n+je] = 1

				// Back-substitute for complex pair.
				for i := je - 2; i >= 0; i-- {
					if i > 0 && s[i*lds+i-1] != 0 {
						continue
					}

					na := 1
					if i < je-2 && i < n-1 && s[(i+1)*lds+i] != 0 {
						na = 2
					}

					// Compute RHS for real and imaginary parts.
					sumr := -acoef * bi.Ddot(je-1-i, s[i*lds+i+1:], 1, work[i+1:], 1)
					sumr += bcoefr * bi.Ddot(je-1-i, p[i*ldp+i+1:], 1, work[i+1:], 1)
					sumr -= bcoefi * bi.Ddot(je-1-i, p[i*ldp+i+1:], 1, work[n+i+1:], 1)

					sumi := -acoef * bi.Ddot(je-1-i, s[i*lds+i+1:], 1, work[n+i+1:], 1)
					sumi += bcoefr * bi.Ddot(je-1-i, p[i*ldp+i+1:], 1, work[n+i+1:], 1)
					sumi += bcoefi * bi.Ddot(je-1-i, p[i*ldp+i+1:], 1, work[i+1:], 1)

					if na == 1 {
						// 1x1 block with complex shift.
						a1x1 := [1]float64{s[i*lds+i]}
						b1x1 := [2]float64{sumr, sumi}
						var x1x1 [2]float64
						scale, _, _ := impl.Dlaln2(false, 1, 2, small, acoef, a1x1[:], 1,
							bcoefr*p[i*ldp+i], 0, b1x1[:], 2, bcoefr, bcoefi, x1x1[:], 2)
						work[i] = x1x1[0] / scale
						work[n+i] = x1x1[1] / scale
					} else {
						// 2x2 block with complex shift.
						sumr2 := -acoef * bi.Ddot(je-2-i, s[(i+1)*lds+i+2:], 1, work[i+2:], 1)
						sumr2 += bcoefr * bi.Ddot(je-2-i, p[(i+1)*ldp+i+2:], 1, work[i+2:], 1)
						sumr2 -= bcoefi * bi.Ddot(je-2-i, p[(i+1)*ldp+i+2:], 1, work[n+i+2:], 1)

						sumi2 := -acoef * bi.Ddot(je-2-i, s[(i+1)*lds+i+2:], 1, work[n+i+2:], 1)
						sumi2 += bcoefr * bi.Ddot(je-2-i, p[(i+1)*ldp+i+2:], 1, work[n+i+2:], 1)
						sumi2 += bcoefi * bi.Ddot(je-2-i, p[(i+1)*ldp+i+2:], 1, work[i+2:], 1)

						a2x2 := [4]float64{s[i*lds+i], s[i*lds+i+1], s[(i+1)*lds+i], s[(i+1)*lds+i+1]}
						b2x2 := [4]float64{sumr, sumi, sumr2, sumi2}
						var x2x2 [4]float64
						scale, _, _ := impl.Dlaln2(false, 2, 2, small, acoef, a2x2[:], 2,
							bcoefr*p[i*ldp+i], bcoefr*p[(i+1)*ldp+i+1], b2x2[:], 2, bcoefr, bcoefi, x2x2[:], 2)
						work[i] = x2x2[0] / scale
						work[n+i] = x2x2[1] / scale
						work[i+1] = x2x2[2] / scale
						work[n+i+1] = x2x2[3] / scale
						i--
					}
				}

				// Copy to output.
				if over {
					bi.Dgemv(blas.NoTrans, n, je, 1, vr, ldvr, work[:je], 1, 0, work[2*n:], 1)
					bi.Dgemv(blas.NoTrans, n, je, 1, vr, ldvr, work[n:n+je], 1, 0, work[3*n:], 1)
					for i := 0; i < n; i++ {
						vr[i*ldvr+im-1] = work[2*n+i]
						vr[i*ldvr+im] = work[3*n+i]
					}
				} else {
					for i := 0; i < n; i++ {
						vr[i*ldvr+im-1] = work[i]
						vr[i*ldvr+im] = work[n+i]
					}
				}

				// Normalize.
				xmax := 0.0
				for i := 0; i < n; i++ {
					xmax = math.Max(xmax, math.Abs(vr[i*ldvr+im-1])+math.Abs(vr[i*ldvr+im]))
				}
				if xmax > safmin {
					bi.Dscal(n, 1/xmax, vr[im-1:], ldvr)
					bi.Dscal(n, 1/xmax, vr[im:], ldvr)
				}

				im -= 2
				je -= 2
			}
		}
	}

	// Compute left eigenvectors.
	if compl {
		im := 0
		for je := 0; je < n; {
			if somev && !selected[je] {
				je++
				continue
			}

			// Detect 2x2 block.
			nw := 1
			if je < n-1 && s[(je+1)*lds+je] != 0 {
				nw = 2
			}

			if nw == 1 {
				// Real eigenvalue.
				acoef := 1.0
				bcoef := 1.0
				if math.Abs(s[je*lds+je]) > safmin {
					acoef = 1.0
					bcoef = s[je*lds+je] / p[je*ldp+je]
				} else if math.Abs(p[je*ldp+je]) > safmin {
					acoef = p[je*ldp+je] / s[je*lds+je]
					bcoef = 1.0
				}

				// Initialize eigenvector.
				for i := 0; i < n; i++ {
					work[i] = 0
				}
				work[je] = 1

				// Forward-substitute.
				for i := je + 1; i < n; i++ {
					if i < n-1 && s[(i+1)*lds+i] != 0 {
						continue
					}

					na := 1
					if i > je+1 && s[i*lds+i-1] != 0 {
						na = 2
						i--
					}

					if na == 1 {
						work[i] = -acoef * bi.Ddot(i-je, s[je*lds+je:], 1, work[je:], 1)
						work[i] += bcoef * bi.Ddot(i-je, p[je*ldp+je:], 1, work[je:], 1)

						denom := acoef*s[i*lds+i] - bcoef*p[i*ldp+i]
						if math.Abs(denom) < safmin {
							denom = safmin
						}
						work[i] /= denom
					} else {
						work[i] = -acoef * bi.Ddot(i-je, s[je*lds+je:], 1, work[je:], 1)
						work[i] += bcoef * bi.Ddot(i-je, p[je*ldp+je:], 1, work[je:], 1)
						work[i+1] = -acoef * bi.Ddot(i+1-je, s[je*lds+je:], 1, work[je:], 1)
						work[i+1] += bcoef * bi.Ddot(i+1-je, p[je*ldp+je:], 1, work[je:], 1)

						a2x2 := [4]float64{s[i*lds+i], s[(i+1)*lds+i], s[i*lds+i+1], s[(i+1)*lds+i+1]}
						b2x2 := [2]float64{work[i], work[i+1]}
						var x2x2 [2]float64
						scale, _, _ := impl.Dlaln2(true, 2, 1, small, acoef, a2x2[:], 2,
							bcoef*p[i*ldp+i], bcoef*p[(i+1)*ldp+i+1], b2x2[:], 1, bcoef, 0, x2x2[:], 1)
						work[i] = x2x2[0] / scale
						work[i+1] = x2x2[1] / scale
						i++
					}
				}

				// Copy to output.
				if over {
					bi.Dgemv(blas.NoTrans, n, n-je, 1, vl[je:], ldvl, work[je:], 1, 0, work[2*n:], 1)
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = work[2*n+i]
					}
				} else {
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = work[i]
					}
				}

				// Normalize.
				xmax := 0.0
				for i := 0; i < n; i++ {
					xmax = math.Max(xmax, math.Abs(vl[i*ldvl+im]))
				}
				if xmax > safmin {
					bi.Dscal(n, 1/xmax, vl[im:], ldvl)
				}

				im++
				je++
			} else {
				// Complex eigenvalue pair.
				_, _, acoefr, _, bcoefi := impl.Dlag2(s[je*lds+je:], lds, p[je*ldp+je:], ldp)
				acoef := 1.0
				bcoefr := acoefr
				if math.Abs(acoefr) > safmin || math.Abs(bcoefi) > safmin {
					scale := 1 / math.Max(math.Abs(acoefr), math.Abs(bcoefi))
					acoef = scale
					bcoefr = acoefr * scale
					bcoefi = bcoefi * scale
				}

				// Initialize eigenvector.
				for i := 0; i < n; i++ {
					work[i] = 0
					work[n+i] = 0
				}
				work[je] = 1
				work[n+je+1] = 1

				// Forward-substitute for complex pair.
				for i := je + 2; i < n; i++ {
					if i < n-1 && s[(i+1)*lds+i] != 0 {
						continue
					}

					na := 1
					if i > je+2 && s[i*lds+i-1] != 0 {
						na = 2
						i--
					}

					sumr := -acoef * bi.Ddot(i-je, s[je*lds+je:], 1, work[je:], 1)
					sumr += bcoefr * bi.Ddot(i-je, p[je*ldp+je:], 1, work[je:], 1)
					sumr += bcoefi * bi.Ddot(i-je, p[je*ldp+je:], 1, work[n+je:], 1)

					sumi := -acoef * bi.Ddot(i-je, s[je*lds+je:], 1, work[n+je:], 1)
					sumi += bcoefr * bi.Ddot(i-je, p[je*ldp+je:], 1, work[n+je:], 1)
					sumi -= bcoefi * bi.Ddot(i-je, p[je*ldp+je:], 1, work[je:], 1)

					if na == 1 {
						a1x1 := [1]float64{s[i*lds+i]}
						b1x1 := [2]float64{sumr, sumi}
						var x1x1 [2]float64
						scale, _, _ := impl.Dlaln2(true, 1, 2, small, acoef, a1x1[:], 1,
							bcoefr*p[i*ldp+i], 0, b1x1[:], 2, bcoefr, bcoefi, x1x1[:], 2)
						work[i] = x1x1[0] / scale
						work[n+i] = x1x1[1] / scale
					} else {
						sumr2 := -acoef * bi.Ddot(i+1-je, s[je*lds+je:], 1, work[je:], 1)
						sumr2 += bcoefr * bi.Ddot(i+1-je, p[je*ldp+je:], 1, work[je:], 1)
						sumr2 += bcoefi * bi.Ddot(i+1-je, p[je*ldp+je:], 1, work[n+je:], 1)

						sumi2 := -acoef * bi.Ddot(i+1-je, s[je*lds+je:], 1, work[n+je:], 1)
						sumi2 += bcoefr * bi.Ddot(i+1-je, p[je*ldp+je:], 1, work[n+je:], 1)
						sumi2 -= bcoefi * bi.Ddot(i+1-je, p[je*ldp+je:], 1, work[je:], 1)

						a2x2 := [4]float64{s[i*lds+i], s[(i+1)*lds+i], s[i*lds+i+1], s[(i+1)*lds+i+1]}
						b2x2 := [4]float64{sumr, sumi, sumr2, sumi2}
						var x2x2 [4]float64
						scale, _, _ := impl.Dlaln2(true, 2, 2, small, acoef, a2x2[:], 2,
							bcoefr*p[i*ldp+i], bcoefr*p[(i+1)*ldp+i+1], b2x2[:], 2, bcoefr, bcoefi, x2x2[:], 2)
						work[i] = x2x2[0] / scale
						work[n+i] = x2x2[1] / scale
						work[i+1] = x2x2[2] / scale
						work[n+i+1] = x2x2[3] / scale
						i++
					}
				}

				// Copy to output.
				if over {
					bi.Dgemv(blas.NoTrans, n, n-je, 1, vl[je:], ldvl, work[je:], 1, 0, work[2*n:], 1)
					bi.Dgemv(blas.NoTrans, n, n-je, 1, vl[je:], ldvl, work[n+je:], 1, 0, work[3*n:], 1)
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = work[2*n+i]
						vl[i*ldvl+im+1] = work[3*n+i]
					}
				} else {
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = work[i]
						vl[i*ldvl+im+1] = work[n+i]
					}
				}

				// Normalize.
				xmax := 0.0
				for i := 0; i < n; i++ {
					xmax = math.Max(xmax, math.Abs(vl[i*ldvl+im])+math.Abs(vl[i*ldvl+im+1]))
				}
				if xmax > safmin {
					bi.Dscal(n, 1/xmax, vl[im:], ldvl)
					bi.Dscal(n, 1/xmax, vl[im+1:], ldvl)
				}

				im += 2
				je += 2
			}
		}
	}

	_ = big
	_ = bignum
	_ = ulp

	return m, true
}
