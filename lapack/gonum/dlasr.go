// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// Dlasr applies a sequence of plane rotations to the m×n matrix A. This series
// of plane rotations is implicitly represented by a matrix P. P is multiplied
// by a depending on the value of side -- A = P * A if side == lapack.Left,
// A = A * Pᵀ if side == lapack.Right.
//
// The exact value of P depends on the value of pivot, but in all cases P is
// implicitly represented by a series of 2×2 rotation matrices. The entries of
// rotation matrix k are defined by s[k] and c[k]
//
//	R(k) = [ c[k] s[k]]
//	       [-s[k] c[k]]
//
// If direct == lapack.Forward, the rotation matrices are applied as
// P = P(z-1) * ... * P(2) * P(1), while if direct == lapack.Backward they are
// applied as P = P(1) * P(2) * ... * P(n).
//
// pivot defines the mapping of the elements in R(k) to P(k).
// If pivot == lapack.Variable, the rotation is performed for the (k, k+1) plane.
//
//	P(k) = [1                    ]
//	       [    ...              ]
//	       [     1               ]
//	       [       c[k] s[k]     ]
//	       [      -s[k] c[k]     ]
//	       [                 1   ]
//	       [                ...  ]
//	       [                    1]
//
// if pivot == lapack.Top, the rotation is performed for the (1, k+1) plane,
//
//	P(k) = [c[k]        s[k]     ]
//	       [    1                ]
//	       [     ...             ]
//	       [         1           ]
//	       [-s[k]       c[k]     ]
//	       [                 1   ]
//	       [                ...  ]
//	       [                    1]
//
// and if pivot == lapack.Bottom, the rotation is performed for the (k, z) plane.
//
//	P(k) = [1                    ]
//	       [  ...                ]
//	       [      1              ]
//	       [        c[k]     s[k]]
//	       [           1         ]
//	       [            ...      ]
//	       [              1      ]
//	       [       -s[k]     c[k]]
//
// s and c have length m - 1 if side == blas.Left, and n - 1 if side == blas.Right.
//
// Dlasr is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlasr(side blas.Side, pivot lapack.Pivot, direct lapack.Direct, m, n int, c, s, a []float64, lda int) {
	switch {
	case side != blas.Left && side != blas.Right:
		panic(badSide)
	case pivot != lapack.Variable && pivot != lapack.Top && pivot != lapack.Bottom:
		panic(badPivot)
	case direct != lapack.Forward && direct != lapack.Backward:
		panic(badDirect)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	if side == blas.Left {
		if len(c) < m-1 {
			panic(shortC)
		}
		if len(s) < m-1 {
			panic(shortS)
		}
	} else {
		if len(c) < n-1 {
			panic(shortC)
		}
		if len(s) < n-1 {
			panic(shortS)
		}
	}
	if len(a) < (m-1)*lda+n {
		panic(shortA)
	}

	if side == blas.Left {
		if pivot == lapack.Variable {
			if m >= 2 && n >= 16 && (m > 2 || n >= 32) && dlasrLeftVariableSIMD(direct, m, n, c, s, a, lda) {
				return
			}
			if direct == lapack.Forward {
				for j := 0; j < m-1; j++ {
					ctmp := c[j]
					stmp := s[j]
					if ctmp != 1 || stmp != 0 {
						for i := 0; i < n; i++ {
							tmp2 := a[j*lda+i]
							tmp := a[(j+1)*lda+i]
							a[(j+1)*lda+i] = ctmp*tmp - stmp*tmp2
							a[j*lda+i] = stmp*tmp + ctmp*tmp2
						}
					}
				}
				return
			}
			for j := m - 2; j >= 0; j-- {
				ctmp := c[j]
				stmp := s[j]
				if ctmp != 1 || stmp != 0 {
					for i := 0; i < n; i++ {
						tmp2 := a[j*lda+i]
						tmp := a[(j+1)*lda+i]
						a[(j+1)*lda+i] = ctmp*tmp - stmp*tmp2
						a[j*lda+i] = stmp*tmp + ctmp*tmp2
					}
				}
			}
			return
		} else if pivot == lapack.Top {
			if direct == lapack.Forward {
				for j := 1; j < m; j++ {
					ctmp := c[j-1]
					stmp := s[j-1]
					if ctmp != 1 || stmp != 0 {
						for i := 0; i < n; i++ {
							tmp := a[j*lda+i]
							tmp2 := a[i]
							a[j*lda+i] = ctmp*tmp - stmp*tmp2
							a[i] = stmp*tmp + ctmp*tmp2
						}
					}
				}
				return
			}
			for j := m - 1; j >= 1; j-- {
				ctmp := c[j-1]
				stmp := s[j-1]
				if ctmp != 1 || stmp != 0 {
					for i := 0; i < n; i++ {
						tmp := a[j*lda+i]
						tmp2 := a[i]
						a[j*lda+i] = ctmp*tmp - stmp*tmp2
						a[i] = stmp*tmp + ctmp*tmp2
					}
				}
			}
			return
		}
		if direct == lapack.Forward {
			for j := 0; j < m-1; j++ {
				ctmp := c[j]
				stmp := s[j]
				if ctmp != 1 || stmp != 0 {
					for i := 0; i < n; i++ {
						tmp := a[j*lda+i]
						tmp2 := a[(m-1)*lda+i]
						a[j*lda+i] = stmp*tmp2 + ctmp*tmp
						a[(m-1)*lda+i] = ctmp*tmp2 - stmp*tmp
					}
				}
			}
			return
		}
		for j := m - 2; j >= 0; j-- {
			ctmp := c[j]
			stmp := s[j]
			if ctmp != 1 || stmp != 0 {
				for i := 0; i < n; i++ {
					tmp := a[j*lda+i]
					tmp2 := a[(m-1)*lda+i]
					a[j*lda+i] = stmp*tmp2 + ctmp*tmp
					a[(m-1)*lda+i] = ctmp*tmp2 - stmp*tmp
				}
			}
		}
		return
	}
	if pivot == lapack.Variable {
		if n == 1 {
			return
		}
		if m >= 64 && n >= 64 && dlasrRightVariableBlocked(direct, m, n, c, s, a, lda) {
			return
		}
		if m >= 32 && n >= 16 && dlasrRightVariableSequential(direct, m, n, c, s, a, lda) {
			return
		}
		if direct == lapack.Forward {
			for j := 0; j < n-1; j++ {
				ctmp := c[j]
				stmp := s[j]
				if ctmp != 1 || stmp != 0 {
					for i := 0; i < m; i++ {
						tmp := a[i*lda+j+1]
						tmp2 := a[i*lda+j]
						a[i*lda+j+1] = ctmp*tmp - stmp*tmp2
						a[i*lda+j] = stmp*tmp + ctmp*tmp2
					}
				}
			}
			return
		}
		for j := n - 2; j >= 0; j-- {
			ctmp := c[j]
			stmp := s[j]
			if ctmp != 1 || stmp != 0 {
				for i := 0; i < m; i++ {
					tmp := a[i*lda+j+1]
					tmp2 := a[i*lda+j]
					a[i*lda+j+1] = ctmp*tmp - stmp*tmp2
					a[i*lda+j] = stmp*tmp + ctmp*tmp2
				}
			}
		}
		return
	} else if pivot == lapack.Top {
		if direct == lapack.Forward {
			for j := 1; j < n; j++ {
				ctmp := c[j-1]
				stmp := s[j-1]
				if ctmp != 1 || stmp != 0 {
					for i := 0; i < m; i++ {
						tmp := a[i*lda+j]
						tmp2 := a[i*lda]
						a[i*lda+j] = ctmp*tmp - stmp*tmp2
						a[i*lda] = stmp*tmp + ctmp*tmp2
					}
				}
			}
			return
		}
		for j := n - 1; j >= 1; j-- {
			ctmp := c[j-1]
			stmp := s[j-1]
			if ctmp != 1 || stmp != 0 {
				for i := 0; i < m; i++ {
					tmp := a[i*lda+j]
					tmp2 := a[i*lda]
					a[i*lda+j] = ctmp*tmp - stmp*tmp2
					a[i*lda] = stmp*tmp + ctmp*tmp2
				}
			}
		}
		return
	}
	if direct == lapack.Forward {
		for j := 0; j < n-1; j++ {
			ctmp := c[j]
			stmp := s[j]
			if ctmp != 1 || stmp != 0 {
				for i := 0; i < m; i++ {
					tmp := a[i*lda+j]
					tmp2 := a[i*lda+n-1]
					a[i*lda+j] = stmp*tmp2 + ctmp*tmp
					a[i*lda+n-1] = ctmp*tmp2 - stmp*tmp
				}

			}
		}
		return
	}
	for j := n - 2; j >= 0; j-- {
		ctmp := c[j]
		stmp := s[j]
		if ctmp != 1 || stmp != 0 {
			for i := 0; i < m; i++ {
				tmp := a[i*lda+j]
				tmp2 := a[i*lda+n-1]
				a[i*lda+j] = stmp*tmp2 + ctmp*tmp
				a[i*lda+n-1] = ctmp*tmp2 - stmp*tmp
			}
		}
	}
}

func dlasrRightVariableBlocked(direct lapack.Direct, m, n int, c, s, a []float64, lda int) bool {
	for j := 0; j < n-1; j++ {
		if c[j] == 1 && s[j] == 0 {
			return false
		}
	}
	if dlasrRightVariableCarry4(direct, m, n, c, s, a, lda) {
		return true
	}
	if direct == lapack.Forward {
		for ib := 0; ib < m; {
			iend := min(ib+32, m)
			if m-iend < 16 {
				iend = m
			}
			for j := 0; j < n-1; j++ {
				ctmp := c[j]
				stmp := s[j]
				k := ib*lda + j
				for i := ib; i < iend; i++ {
					tmp := a[k+1]
					tmp2 := a[k]
					a[k+1] = ctmp*tmp - stmp*tmp2
					a[k] = stmp*tmp + ctmp*tmp2
					k += lda
				}
			}
			ib = iend
		}
		return true
	}
	for ib := 0; ib < m; {
		iend := min(ib+32, m)
		if m-iend < 16 {
			iend = m
		}
		for j := n - 2; j >= 0; j-- {
			ctmp := c[j]
			stmp := s[j]
			k := ib*lda + j
			for i := ib; i < iend; i++ {
				tmp := a[k+1]
				tmp2 := a[k]
				a[k+1] = ctmp*tmp - stmp*tmp2
				a[k] = stmp*tmp + ctmp*tmp2
				k += lda
			}
		}
		ib = iend
	}
	return true
}
