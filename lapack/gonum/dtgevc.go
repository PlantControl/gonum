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
// triangular. Matrix pairs of this type are produced by the generalized Schur
// factorization of a matrix pair (A,B):
//
//	A = Q * S * Zᵀ,  B = Q * P * Zᵀ
//
// as computed by Dgghrd + Dhgeqz.
//
// The right eigenvector x and the left eigenvector y of (S, P) corresponding to
// an eigenvalue w are defined by:
//
//	S * x = w * P * x
//	yᴴ * S = w * yᴴ * P
//
// The eigenvalues are not input to this routine but are determined from the
// diagonal blocks of S and P. For real eigenvalues (1×1 blocks), the
// eigenvector is real. For a complex conjugate pair (2×2 block), the
// eigenvector of the eigenvalue with positive imaginary part is stored as two
// consecutive columns holding its real and imaginary parts.
//
// side specifies which eigenvectors to compute:
//
//	lapack.EVRight: right eigenvectors only
//	lapack.EVLeft: left eigenvectors only
//	lapack.EVBoth: both right and left eigenvectors
//
// howmny specifies how many eigenvectors to compute:
//
//	lapack.EVAll: all eigenvectors of (S, P)
//	lapack.EVAllMulQ: all eigenvectors, back-transformed by the matrices
//	                  supplied in vl (Q) and vr (Z) on entry
//	lapack.EVSelected: selected eigenvectors (indicated by selected[])
//
// selected must have length n when howmny is lapack.EVSelected. If selected[j]
// is true, the eigenvector corresponding to the j-th eigenvalue is computed.
// A complex pair is computed if either of its two elements is selected.
//
// S must have standardized 2×2 diagonal blocks (no adjacent blocks) and, for
// each 2×2 block of S, the corresponding diagonal block of P must be diagonal
// with nonzero diagonal, otherwise Dtgevc will panic.
//
// vl and vr are n×mm matrices. mm must be at least m, the number of columns
// required to store the requested eigenvectors.
//
// Each eigenvector is scaled so the largest component has
// abs(real part)+abs(imag part) = 1.
//
// work must have length at least 6*n.
//
// The return value m is the number of columns used to store the eigenvectors.
// ok is false if a 2×2 block of (S, P) has real eigenvalues.
//
// Dtgevc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgevc(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int,
	s []float64, lds int, p []float64, ldp int, vl []float64, ldvl int, vr []float64, ldvr int,
	mm int, work []float64) (m int, ok bool) {

	compl := side == lapack.EVLeft || side == lapack.EVBoth
	compr := side == lapack.EVRight || side == lapack.EVBoth
	ilback := howmny == lapack.EVAllMulQ
	ilall := howmny != lapack.EVSelected

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
	case ldvl < 1:
		panic(badLdVL)
	case ldvr < 1:
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
	case !ilall && len(selected) < n:
		panic(badLenSelected)
	case len(work) < 6*n:
		panic(shortWork)
	}

	if ilall {
		m = n
	} else {
		m = 0
		for j := 0; j < n; j++ {
			if j < n-1 && s[(j+1)*lds+j] != 0 {
				if selected[j] || selected[j+1] {
					m += 2
				}
				j++
				continue
			}
			if selected[j] {
				m++
			}
		}
	}

	for j := 0; j < n-1; j++ {
		if s[(j+1)*lds+j] == 0 {
			continue
		}
		if p[j*ldp+j] == 0 || p[(j+1)*ldp+j+1] == 0 || p[j*ldp+j+1] != 0 {
			panic("lapack: P has a non-diagonal block under a 2×2 block of S")
		}
		if j < n-2 && s[(j+2)*lds+j+1] != 0 {
			panic("lapack: S has adjacent 2×2 blocks")
		}
	}

	switch {
	case mm < m:
		panic(badMm)
	case compl && ldvl < mm:
		panic(badLdVL)
	case compr && ldvr < mm:
		panic(badLdVR)
	case compl && len(vl) < (n-1)*ldvl+mm:
		panic(shortVL)
	case compr && len(vr) < (n-1)*ldvr+mm:
		panic(shortVR)
	}

	const safety = 100
	safmin := dlamchS
	ulp := dlamchP
	small := safmin * float64(n) / ulp
	big := 1 / small
	bignum := 1 / (safmin * float64(n))

	// work[0:n] and work[n:2n] hold the 1-norms of the strictly upper
	// triangular parts of the columns of S and P, excluding diagonal blocks.
	anorm := math.Abs(s[0])
	if n > 1 {
		anorm += math.Abs(s[lds])
	}
	bnorm := math.Abs(p[0])
	work[0] = 0
	work[n] = 0
	for j := 1; j < n; j++ {
		var temp, temp2 float64
		iend := j
		if s[j*lds+j-1] != 0 {
			iend = j - 1
		}
		for i := 0; i < iend; i++ {
			temp += math.Abs(s[i*lds+j])
			temp2 += math.Abs(p[i*ldp+j])
		}
		work[j] = temp
		work[n+j] = temp2
		for i := iend; i < min(j+2, n); i++ {
			temp += math.Abs(s[i*lds+j])
			temp2 += math.Abs(p[i*ldp+j])
		}
		anorm = math.Max(anorm, temp)
		bnorm = math.Max(bnorm, temp2)
	}
	ascale := 1 / math.Max(anorm, safmin)
	bscale := 1 / math.Max(bnorm, safmin)

	bi := blas64.Implementation()
	var (
		bdiag [2]float64
		sum   [4]float64
		sums  [4]float64
		sump  [4]float64
		rhs   [4]float64
	)

	realCoef := func(je int) (acoef, bcoefr float64) {
		temp := 1 / math.Max(math.Max(math.Abs(s[je*lds+je])*ascale, math.Abs(p[je*ldp+je])*bscale), safmin)
		salfar := (temp * s[je*lds+je]) * ascale
		sbeta := (temp * p[je*ldp+je]) * bscale
		acoef = sbeta * ascale
		bcoefr = salfar * bscale
		scale := 1.0
		lsa := math.Abs(sbeta) >= safmin && math.Abs(acoef) < small
		lsb := math.Abs(salfar) >= safmin && math.Abs(bcoefr) < small
		if lsa {
			scale = (small / math.Abs(sbeta)) * math.Min(anorm, big)
		}
		if lsb {
			scale = math.Max(scale, (small/math.Abs(salfar))*math.Min(bnorm, big))
		}
		if lsa || lsb {
			scale = math.Min(scale, 1/(safmin*math.Max(1, math.Max(math.Abs(acoef), math.Abs(bcoefr)))))
			if lsa {
				acoef = ascale * (scale * sbeta)
			} else {
				acoef *= scale
			}
			if lsb {
				bcoefr = bscale * (scale * salfar)
			} else {
				bcoefr *= scale
			}
		}
		return acoef, bcoefr
	}

	scaleComplexCoef := func(acoef, bcoefr, bcoefi float64) (float64, float64, float64) {
		acoefa := math.Abs(acoef)
		bcoefa := math.Abs(bcoefr) + math.Abs(bcoefi)
		scale := 1.0
		if acoefa*ulp < safmin && acoefa >= safmin {
			scale = (safmin / ulp) / acoefa
		}
		if bcoefa*ulp < safmin && bcoefa >= safmin {
			scale = math.Max(scale, (safmin/ulp)/bcoefa)
		}
		if safmin*acoefa > ascale {
			scale = ascale / (safmin * acoefa)
		}
		if safmin*bcoefa > bscale {
			scale = math.Min(scale, bscale/(safmin*bcoefa))
		}
		if scale != 1 {
			acoef *= scale
			bcoefr *= scale
			bcoefi *= scale
		}
		return acoef, bcoefr, bcoefi
	}

	if compl {
		ieig := 0
		for je := 0; je < n; je++ {
			nw := 1
			ilcplx := je < n-1 && s[(je+1)*lds+je] != 0
			if ilcplx {
				nw = 2
			}
			ilcomp := ilall
			if !ilall {
				ilcomp = selected[je] || (ilcplx && selected[je+1])
			}
			if !ilcomp {
				if ilcplx {
					je++
				}
				continue
			}

			if !ilcplx && math.Abs(s[je*lds+je]) <= safmin && math.Abs(p[je*ldp+je]) <= safmin {
				// Singular pencil: unit eigenvector, not back-transformed (as Netlib).
				for jr := 0; jr < n; jr++ {
					vl[jr*ldvl+ieig] = 0
				}
				vl[ieig*ldvl+ieig] = 1
				ieig++
				continue
			}

			for jr := 0; jr < nw*n; jr++ {
				work[2*n+jr] = 0
			}

			var acoef, bcoefr, bcoefi, xmax float64
			if !ilcplx {
				acoef, bcoefr = realCoef(je)
				work[2*n+je] = 1
				xmax = 1
			} else {
				acoef, _, bcoefr, _, bcoefi = impl.dlag2(s[je*lds+je:], lds, p[je*ldp+je:], ldp, safmin*safety)
				bcoefi = -bcoefi
				if bcoefi == 0 {
					return m, false
				}
				acoef, bcoefr, bcoefi = scaleComplexCoef(acoef, bcoefr, bcoefi)

				temp := acoef * s[(je+1)*lds+je]
				temp2r := acoef*s[je*lds+je] - bcoefr*p[je*ldp+je]
				temp2i := -bcoefi * p[je*ldp+je]
				if math.Abs(temp) > math.Abs(temp2r)+math.Abs(temp2i) {
					work[2*n+je] = 1
					work[3*n+je] = 0
					work[2*n+je+1] = -temp2r / temp
					work[3*n+je+1] = -temp2i / temp
				} else {
					work[2*n+je+1] = 1
					work[3*n+je+1] = 0
					temp = acoef * s[je*lds+je+1]
					work[2*n+je] = (bcoefr*p[(je+1)*ldp+je+1] - acoef*s[(je+1)*lds+je+1]) / temp
					work[3*n+je] = bcoefi * p[(je+1)*ldp+je+1] / temp
				}
				xmax = math.Max(math.Abs(work[2*n+je])+math.Abs(work[3*n+je]),
					math.Abs(work[2*n+je+1])+math.Abs(work[3*n+je+1]))
			}
			acoefa := math.Abs(acoef)
			bcoefa := math.Abs(bcoefr) + math.Abs(bcoefi)
			dmin := math.Max(math.Max(ulp*acoefa*anorm, ulp*bcoefa*bnorm), safmin)

			// Triangular solve of (a S - b P)ᵀ y = 0, rowwise in (a S - b P)ᵀ.
			for j := je + nw; j < n; j++ {
				na := 1
				bdiag[0] = p[j*ldp+j]
				if j < n-1 && s[(j+1)*lds+j] != 0 {
					bdiag[1] = p[(j+1)*ldp+j+1]
					na = 2
				}

				xscale := 1 / math.Max(1, xmax)
				temp := math.Max(math.Max(work[j], work[n+j]), acoefa*work[j]+bcoefa*work[n+j])
				if na == 2 {
					temp = math.Max(temp, math.Max(math.Max(work[j+1], work[n+j+1]), acoefa*work[j+1]+bcoefa*work[n+j+1]))
				}
				if temp > bignum*xscale {
					for jw := 0; jw < nw; jw++ {
						for jr := je; jr < j; jr++ {
							work[(jw+2)*n+jr] *= xscale
						}
					}
					xmax *= xscale
				}

				for jw := 0; jw < nw; jw++ {
					for ja := 0; ja < na; ja++ {
						var ss, sp float64
						for jr := je; jr < j; jr++ {
							x := work[(jw+2)*n+jr]
							ss += s[jr*lds+j+ja] * x
							sp += p[jr*ldp+j+ja] * x
						}
						sums[ja*2+jw] = ss
						sump[ja*2+jw] = sp
					}
				}
				for ja := 0; ja < na; ja++ {
					if ilcplx {
						sum[ja*2] = -acoef*sums[ja*2] + bcoefr*sump[ja*2] - bcoefi*sump[ja*2+1]
						sum[ja*2+1] = -acoef*sums[ja*2+1] + bcoefr*sump[ja*2+1] + bcoefi*sump[ja*2]
					} else {
						sum[ja*2] = -acoef*sums[ja*2] + bcoefr*sump[ja*2]
					}
				}

				scale, xnorm, _ := impl.Dlaln2(true, na, nw, dmin, acoef, s[j*lds+j:], lds,
					bdiag[0], bdiag[1], sum[:], 2, bcoefr, bcoefi, rhs[:], 2)
				for jw := 0; jw < nw; jw++ {
					for ja := 0; ja < na; ja++ {
						work[(jw+2)*n+j+ja] = rhs[ja*2+jw]
					}
				}
				if scale < 1 {
					for jw := 0; jw < nw; jw++ {
						for jr := je; jr < j; jr++ {
							work[(jw+2)*n+jr] *= scale
						}
					}
					xmax *= scale
				}
				xmax = math.Max(xmax, xnorm)
				if na == 2 {
					j++
				}
			}

			ibeg := je
			if ilback {
				for jw := 0; jw < nw; jw++ {
					bi.Dgemv(blas.NoTrans, n, n-je, 1, vl[je:], ldvl,
						work[(jw+2)*n+je:], 1, 0, work[(jw+4)*n:], 1)
				}
				for jw := 0; jw < nw; jw++ {
					for jr := 0; jr < n; jr++ {
						vl[jr*ldvl+ieig+jw] = work[(jw+4)*n+jr]
					}
				}
				ibeg = 0
			} else {
				for jw := 0; jw < nw; jw++ {
					for jr := 0; jr < n; jr++ {
						vl[jr*ldvl+ieig+jw] = work[(jw+2)*n+jr]
					}
				}
			}

			xmax = 0
			for jr := ibeg; jr < n; jr++ {
				v := math.Abs(vl[jr*ldvl+ieig])
				if ilcplx {
					v += math.Abs(vl[jr*ldvl+ieig+1])
				}
				xmax = math.Max(xmax, v)
			}
			if xmax > safmin {
				xscale := 1 / xmax
				for jw := 0; jw < nw; jw++ {
					for jr := ibeg; jr < n; jr++ {
						vl[jr*ldvl+ieig+jw] *= xscale
					}
				}
			}
			ieig += nw
			if ilcplx {
				je++
			}
		}
	}

	if compr {
		ieig := m
		for je := n - 1; je >= 0; je-- {
			nw := 1
			ilcplx := je > 0 && s[je*lds+je-1] != 0
			if ilcplx {
				nw = 2
			}
			ilcomp := ilall
			if !ilall {
				ilcomp = selected[je] || (ilcplx && selected[je-1])
			}
			if !ilcomp {
				if ilcplx {
					je--
				}
				continue
			}

			if !ilcplx && math.Abs(s[je*lds+je]) <= safmin && math.Abs(p[je*ldp+je]) <= safmin {
				// Singular pencil: unit eigenvector, not back-transformed (as Netlib).
				ieig--
				for jr := 0; jr < n; jr++ {
					vr[jr*ldvr+ieig] = 0
				}
				vr[ieig*ldvr+ieig] = 1
				continue
			}

			for jr := 0; jr < nw*n; jr++ {
				work[2*n+jr] = 0
			}

			var acoef, bcoefr, bcoefi, xmax float64
			if !ilcplx {
				acoef, bcoefr = realCoef(je)
				work[2*n+je] = 1
				xmax = 1
				for jr := 0; jr < je; jr++ {
					work[2*n+jr] = bcoefr*p[jr*ldp+je] - acoef*s[jr*lds+je]
				}
			} else {
				acoef, _, bcoefr, _, bcoefi = impl.dlag2(s[(je-1)*lds+je-1:], lds, p[(je-1)*ldp+je-1:], ldp, safmin*safety)
				if bcoefi == 0 {
					return m, false
				}
				acoef, bcoefr, bcoefi = scaleComplexCoef(acoef, bcoefr, bcoefi)

				temp := acoef * s[je*lds+je-1]
				temp2r := acoef*s[je*lds+je] - bcoefr*p[je*ldp+je]
				temp2i := -bcoefi * p[je*ldp+je]
				if math.Abs(temp) >= math.Abs(temp2r)+math.Abs(temp2i) {
					work[2*n+je] = 1
					work[3*n+je] = 0
					work[2*n+je-1] = -temp2r / temp
					work[3*n+je-1] = -temp2i / temp
				} else {
					work[2*n+je-1] = 1
					work[3*n+je-1] = 0
					temp = acoef * s[(je-1)*lds+je]
					work[2*n+je] = (bcoefr*p[(je-1)*ldp+je-1] - acoef*s[(je-1)*lds+je-1]) / temp
					work[3*n+je] = bcoefi * p[(je-1)*ldp+je-1] / temp
				}
				xmax = math.Max(math.Abs(work[2*n+je])+math.Abs(work[3*n+je]),
					math.Abs(work[2*n+je-1])+math.Abs(work[3*n+je-1]))

				creala := acoef * work[2*n+je-1]
				cimaga := acoef * work[3*n+je-1]
				crealb := bcoefr*work[2*n+je-1] - bcoefi*work[3*n+je-1]
				cimagb := bcoefi*work[2*n+je-1] + bcoefr*work[3*n+je-1]
				cre2a := acoef * work[2*n+je]
				cim2a := acoef * work[3*n+je]
				cre2b := bcoefr*work[2*n+je] - bcoefi*work[3*n+je]
				cim2b := bcoefi*work[2*n+je] + bcoefr*work[3*n+je]
				for jr := 0; jr < je-1; jr++ {
					work[2*n+jr] = -creala*s[jr*lds+je-1] + crealb*p[jr*ldp+je-1] -
						cre2a*s[jr*lds+je] + cre2b*p[jr*ldp+je]
					work[3*n+jr] = -cimaga*s[jr*lds+je-1] + cimagb*p[jr*ldp+je-1] -
						cim2a*s[jr*lds+je] + cim2b*p[jr*ldp+je]
				}
			}
			acoefa := math.Abs(acoef)
			bcoefa := math.Abs(bcoefr) + math.Abs(bcoefi)
			dmin := math.Max(math.Max(ulp*acoefa*anorm, ulp*bcoefa*bnorm), safmin)

			// Columnwise triangular solve of (a S - b P) x = 0.
			il2by2 := false
			for j := je - nw; j >= 0; j-- {
				// A 2×2 block in rows j-1:j is solved on the next
				// iteration, as rows j:j+1.
				if !il2by2 && j > 0 && s[j*lds+j-1] != 0 {
					il2by2 = true
					continue
				}
				bdiag[0] = p[j*ldp+j]
				na := 1
				if il2by2 {
					na = 2
					bdiag[1] = p[(j+1)*ldp+j+1]
				}

				for jw := 0; jw < nw; jw++ {
					for ja := 0; ja < na; ja++ {
						rhs[ja*2+jw] = work[(jw+2)*n+j+ja]
					}
				}
				scale, xnorm, _ := impl.Dlaln2(false, na, nw, dmin, acoef, s[j*lds+j:], lds,
					bdiag[0], bdiag[1], rhs[:], 2, bcoefr, bcoefi, sum[:], 2)
				if scale < 1 {
					for jw := 0; jw < nw; jw++ {
						for jr := 0; jr <= je; jr++ {
							work[(jw+2)*n+jr] *= scale
						}
					}
				}
				xmax = math.Max(scale*xmax, xnorm)
				for jw := 0; jw < nw; jw++ {
					for ja := 0; ja < na; ja++ {
						work[(jw+2)*n+j+ja] = sum[ja*2+jw]
					}
				}

				if j > 0 {
					xscale := 1 / math.Max(1, xmax)
					temp := acoefa*work[j] + bcoefa*work[n+j]
					if il2by2 {
						temp = math.Max(temp, acoefa*work[j+1]+bcoefa*work[n+j+1])
					}
					temp = math.Max(temp, math.Max(acoefa, bcoefa))
					if temp > bignum*xscale {
						for jw := 0; jw < nw; jw++ {
							for jr := 0; jr <= je; jr++ {
								work[(jw+2)*n+jr] *= xscale
							}
						}
						xmax *= xscale
					}

					for ja := 0; ja < na; ja++ {
						if ilcplx {
							creala := acoef * work[2*n+j+ja]
							cimaga := acoef * work[3*n+j+ja]
							crealb := bcoefr*work[2*n+j+ja] - bcoefi*work[3*n+j+ja]
							cimagb := bcoefi*work[2*n+j+ja] + bcoefr*work[3*n+j+ja]
							for jr := 0; jr < j; jr++ {
								work[2*n+jr] += -creala*s[jr*lds+j+ja] + crealb*p[jr*ldp+j+ja]
								work[3*n+jr] += -cimaga*s[jr*lds+j+ja] + cimagb*p[jr*ldp+j+ja]
							}
						} else {
							creala := acoef * work[2*n+j+ja]
							crealb := bcoefr * work[2*n+j+ja]
							for jr := 0; jr < j; jr++ {
								work[2*n+jr] += -creala*s[jr*lds+j+ja] + crealb*p[jr*ldp+j+ja]
							}
						}
					}
				}
				il2by2 = false
			}

			ieig -= nw
			iend := je + 1
			if ilback {
				for jw := 0; jw < nw; jw++ {
					bi.Dgemv(blas.NoTrans, n, je+1, 1, vr, ldvr,
						work[(jw+2)*n:], 1, 0, work[(jw+4)*n:], 1)
				}
				for jw := 0; jw < nw; jw++ {
					for jr := 0; jr < n; jr++ {
						vr[jr*ldvr+ieig+jw] = work[(jw+4)*n+jr]
					}
				}
				iend = n
			} else {
				for jw := 0; jw < nw; jw++ {
					for jr := 0; jr < n; jr++ {
						vr[jr*ldvr+ieig+jw] = work[(jw+2)*n+jr]
					}
				}
			}

			xmax = 0
			for jr := 0; jr < iend; jr++ {
				v := math.Abs(vr[jr*ldvr+ieig])
				if ilcplx {
					v += math.Abs(vr[jr*ldvr+ieig+1])
				}
				xmax = math.Max(xmax, v)
			}
			if xmax > safmin {
				xscale := 1 / xmax
				for jw := 0; jw < nw; jw++ {
					for jr := 0; jr < iend; jr++ {
						vr[jr*ldvr+ieig+jw] *= xscale
					}
				}
			}
			if ilcplx {
				je--
			}
		}
	}

	return m, true
}
