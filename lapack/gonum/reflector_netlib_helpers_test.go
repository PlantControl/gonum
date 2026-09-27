// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func reflectorData(n, k int, direct lapack.Direct, store lapack.StoreV) ([]float64, int) {
	rows, cols := n, k
	if store == lapack.RowWise {
		rows, cols = k, n
	}
	ld := cols + 2
	v := make([]float64, (rows-1)*ld+cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r, c := i, j
			if store == lapack.RowWise {
				r, c = j, i
			}
			diag := c
			if direct == lapack.Backward {
				diag = n - k + c
			}
			switch {
			case r == diag:
				v[i*ld+j] = 1
			case direct == lapack.Forward && r > diag || direct == lapack.Backward && r < diag:
				v[i*ld+j] = float64((r+2)*(c+3)%11-5) / 24
			}
		}
	}
	return v, ld
}

func reflectorColMajor(n, k int, store lapack.StoreV, row []float64, ld int) ([]float64, int) {
	rows, cols := n, k
	if store == lapack.RowWise {
		rows, cols = k, n
	}
	cld := rows + 2
	return rowToCol(rows, cols, row, ld, cld), cld
}

func rowToCol(m, n int, row []float64, ldr, ldc int) []float64 {
	col := make([]float64, (n-1)*ldc+m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			col[j*ldc+i] = row[i*ldr+j]
		}
	}
	return col
}

func checkClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Errorf("%s: got %g want %g", name, got, want)
		return
	}
	d := math.Abs(got - want)
	if d > tol*math.Max(1, math.Max(math.Abs(got), math.Abs(want))) {
		t.Errorf("%s: got %g want %g", name, got, want)
	}
}

func setRowPadding(a []float64, rows, cols, ld int, value float64) {
	for i := 0; i < rows; i++ {
		for j := cols; j < ld && i*ld+j < len(a); j++ {
			a[i*ld+j] = value
		}
	}
}

func setColPadding(a []float64, rows, cols, ld int, value float64) {
	for j := 0; j < cols; j++ {
		for i := rows; i < ld && j*ld+i < len(a); i++ {
			a[j*ld+i] = value
		}
	}
}

func checkRowPadding(t *testing.T, got, want []float64, rows, cols, ld int) {
	t.Helper()
	for i := 0; i < rows; i++ {
		for j := cols; j < ld && i*ld+j < len(got); j++ {
			if math.Float64bits(got[i*ld+j]) != math.Float64bits(want[i*ld+j]) {
				t.Errorf("row padding [%d,%d] modified", i, j)
			}
		}
	}
}

func checkColPadding(t *testing.T, got, want []float64, rows, cols, ld int) {
	t.Helper()
	for j := 0; j < cols; j++ {
		for i := rows; i < ld && j*ld+i < len(got); i++ {
			if math.Float64bits(got[j*ld+i]) != math.Float64bits(want[j*ld+i]) {
				t.Errorf("column padding [%d,%d] modified", i, j)
			}
		}
	}
}
