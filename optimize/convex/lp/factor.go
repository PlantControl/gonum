// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack/lapack64"
)

// columns is a matrix in compressed sparse column form.
type columns struct {
	start []int
	index []int
	value []float64
}

// basisFactor solves with a simplex basis B whose column i is column head[i]
// of a matrix. Vectors indexed by basis position i and by matrix row share
// one length, m.
type basisFactor interface {
	// factorize factorizes the basis and discards earlier updates.
	factorize(a *columns, head []int) error
	// ftran overwrites z with B⁻¹z.
	ftran(z []float64)
	// btran overwrites v with B⁻ᵀv.
	btran(v []float64)
	// update replaces basis column row by the column whose ftran is alpha,
	// and reports whether a refactorization is due.
	update(row int, alpha []float64) bool
}

// sparseFrom is the basis size from which BoundedSimplex factorizes
// sparsely. It is a variable so tests can exercise both factorizations.
var sparseFrom = 0

func newFactor(m int) basisFactor {
	if m >= sparseFrom {
		return newSparseLU(m)
	}
	return newDenseLU(m)
}

// etaFile holds product-form updates B = B₀ E₁ ⋯ E_k, where E_k is the
// identity with column row replaced by alpha. Only nonzero off-diagonal
// entries are stored.
type etaFile struct {
	row   []int
	pivot []float64
	start []int
	index []int
	value []float64
}

func (e *etaFile) reset() {
	e.row, e.pivot = e.row[:0], e.pivot[:0]
	e.start = append(e.start[:0], 0)
	e.index, e.value = e.index[:0], e.value[:0]
}

func (e *etaFile) len() int { return len(e.row) }

func (e *etaFile) add(row int, alpha []float64) {
	e.row = append(e.row, row)
	e.pivot = append(e.pivot, alpha[row])
	for i, a := range alpha {
		if i != row && a != 0 {
			e.index = append(e.index, i)
			e.value = append(e.value, a)
		}
	}
	e.start = append(e.start, len(e.index))
}

// ftran overwrites z with E_k⁻¹ ⋯ E₁⁻¹ z.
func (e *etaFile) ftran(z []float64) {
	for k, r := range e.row {
		zr := z[r] / e.pivot[k]
		if zr != 0 {
			for p := e.start[k]; p < e.start[k+1]; p++ {
				z[e.index[p]] -= e.value[p] * zr
			}
		}
		z[r] = zr
	}
}

// btran overwrites v with E₁⁻ᵀ ⋯ E_k⁻ᵀ v.
func (e *etaFile) btran(v []float64) {
	for k := len(e.row) - 1; k >= 0; k-- {
		r := e.row[k]
		sum := v[r]
		for p := e.start[k]; p < e.start[k+1]; p++ {
			sum -= e.value[p] * v[e.index[p]]
		}
		v[r] = sum / e.pivot[k]
	}
}

// denseLU factorizes the basis with LAPACK's dense LU, so each solve costs
// O(m²) whatever the sparsity.
type denseLU struct {
	m      int
	factor blas64.General // P B₀ = L U, row-major
	ipiv   []int
	etas   etaFile
}

func newDenseLU(m int) *denseLU {
	return &denseLU{
		m:      m,
		factor: blas64.General{Rows: m, Cols: m, Stride: max(m, 1), Data: make([]float64, m*m)},
		ipiv:   make([]int, m),
	}
}

func (d *denseLU) factorize(a *columns, head []int) error {
	d.etas.reset()
	if d.m == 0 {
		return nil
	}
	f := d.factor.Data
	clear(f)
	for i, j := range head {
		for k := a.start[j]; k < a.start[j+1]; k++ {
			f[a.index[k]*d.m+i] = a.value[k]
		}
	}
	if !lapack64.Getrf(d.factor, d.ipiv) {
		return ErrSingular
	}
	return nil
}

func (d *denseLU) solve(v []float64, trans bool) {
	lower := blas64.Triangular{Uplo: blas.Lower, Diag: blas.Unit, N: d.m, Stride: d.factor.Stride, Data: d.factor.Data}
	upper := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: d.m, Stride: d.factor.Stride, Data: d.factor.Data}
	x := blas64.Vector{N: d.m, Inc: 1, Data: v}
	if !trans {
		for i, p := range d.ipiv {
			v[i], v[p] = v[p], v[i]
		}
		blas64.Trsv(blas.NoTrans, lower, x)
		blas64.Trsv(blas.NoTrans, upper, x)
		return
	}
	blas64.Trsv(blas.Trans, upper, x)
	blas64.Trsv(blas.Trans, lower, x)
	for i := d.m - 1; i >= 0; i-- {
		p := d.ipiv[i]
		v[i], v[p] = v[p], v[i]
	}
}

func (d *denseLU) ftran(z []float64) {
	if d.m == 0 {
		return
	}
	d.solve(z, false)
	d.etas.ftran(z)
}

func (d *denseLU) btran(v []float64) {
	if d.m == 0 {
		return
	}
	d.etas.btran(v)
	d.solve(v, true)
}

func (d *denseLU) update(row int, alpha []float64) bool {
	d.etas.add(row, alpha)
	return d.etas.len() >= refactorEvery
}
