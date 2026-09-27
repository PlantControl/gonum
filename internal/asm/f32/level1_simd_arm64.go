// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"unsafe"

	"plantcontrol.org/v1/gonum/internal/math32"
)

// SasumUnitary returns the sum of the absolute values of x.
func SasumUnitary(x []float32) float32 {
	if len(x) < 16 {
		return sasumUnitaryScalar(x)
	}
	data := unsafe.Pointer(unsafe.SliceData(x))
	var sum0, sum1, sum2, sum3 archsimd.Float32x4
	i := 0
	for ; i+16 <= len(x); i += 16 {
		sum0 = sum0.Add(loadFloat32x4(data, i).Abs())
		sum1 = sum1.Add(loadFloat32x4(data, i+4).Abs())
		sum2 = sum2.Add(loadFloat32x4(data, i+8).Abs())
		sum3 = sum3.Add(loadFloat32x4(data, i+12).Abs())
	}
	sum0 = sum0.Add(sum1).Add(sum2.Add(sum3))
	pairs := sum0.ConcatAddPairs(sum0)
	sum := pairs.ConcatAddPairs(pairs).GetElem(0)
	for ; i < len(x); i++ {
		sum += math32.Abs(x[i])
	}
	return sum
}

func sasumUnitaryScalar(x []float32) (sum float32) {
	for _, v := range x {
		sum += math32.Abs(v)
	}
	return sum
}

// SwapUnitary exchanges corresponding elements of x and y.
func SwapUnitary(x, y []float32) {
	if len(x) < 16 || len(y) < len(x) || !simdSlicesIndependent(x, y) {
		swapUnitaryScalar(x, y)
		return
	}
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+16 <= len(x); i += 16 {
		x0 := loadFloat32x4(xData, i)
		x1 := loadFloat32x4(xData, i+4)
		x2 := loadFloat32x4(xData, i+8)
		x3 := loadFloat32x4(xData, i+12)
		y0 := loadFloat32x4(yData, i)
		y1 := loadFloat32x4(yData, i+4)
		y2 := loadFloat32x4(yData, i+8)
		y3 := loadFloat32x4(yData, i+12)
		storeFloat32x4(xData, i, y0)
		storeFloat32x4(xData, i+4, y1)
		storeFloat32x4(xData, i+8, y2)
		storeFloat32x4(xData, i+12, y3)
		storeFloat32x4(yData, i, x0)
		storeFloat32x4(yData, i+4, x1)
		storeFloat32x4(yData, i+8, x2)
		storeFloat32x4(yData, i+12, x3)
	}
	for ; i < len(x); i++ {
		x[i], y[i] = y[i], x[i]
	}
}

func swapUnitaryScalar(x, y []float32) {
	for i, v := range x {
		x[i], y[i] = y[i], v
	}
}

// RotUnitary applies a plane rotation to x and y.
func RotUnitary(x, y []float32, c, s float32) {
	if len(x) < 16 || len(y) < len(x) || !simdSlicesIndependent(x, y) {
		rotUnitaryScalar(x, y, c, s)
		return
	}
	transformRotUnitary(x, y, archsimd.BroadcastFloat32x4(c), archsimd.BroadcastFloat32x4(s))
}

func rotUnitaryScalar(x, y []float32, c, s float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = c*vx+s*vy, c*vy-s*vx
	}
}

// RotmUnitaryRescaling applies a full modified plane rotation to x and y.
func RotmUnitaryRescaling(x, y []float32, h11, h12, h21, h22 float32) {
	if len(x) < 16 || len(y) < len(x) || !simdSlicesIndependent(x, y) {
		rotmUnitaryRescalingScalar(x, y, h11, h12, h21, h22)
		return
	}
	transformUnitary(
		x, y,
		archsimd.BroadcastFloat32x4(h11),
		archsimd.BroadcastFloat32x4(h12),
		archsimd.BroadcastFloat32x4(h21),
		archsimd.BroadcastFloat32x4(h22),
	)
}

func rotmUnitaryRescalingScalar(x, y []float32, h11, h12, h21, h22 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx*h11+vy*h12, vx*h21+vy*h22
	}
}

// RotmUnitaryOffDiagonal applies an off-diagonal modified plane rotation.
func RotmUnitaryOffDiagonal(x, y []float32, h12, h21 float32) {
	if len(x) < 16 || len(y) < len(x) || !simdSlicesIndependent(x, y) {
		rotmUnitaryOffDiagonalScalar(x, y, h12, h21)
		return
	}
	transformOffDiagonalUnitary(x, y, archsimd.BroadcastFloat32x4(h12), archsimd.BroadcastFloat32x4(h21))
}

func rotmUnitaryOffDiagonalScalar(x, y []float32, h12, h21 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx+vy*h12, vx*h21+vy
	}
}

// RotmUnitaryDiagonal applies a diagonal modified plane rotation.
func RotmUnitaryDiagonal(x, y []float32, h11, h22 float32) {
	if len(x) < 16 || len(y) < len(x) || !simdSlicesIndependent(x, y) {
		rotmUnitaryDiagonalScalar(x, y, h11, h22)
		return
	}
	transformDiagonalUnitary(x, y, archsimd.BroadcastFloat32x4(h11), archsimd.BroadcastFloat32x4(h22))
}

func rotmUnitaryDiagonalScalar(x, y []float32, h11, h22 float32) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx*h11+vy, -vx+vy*h22
	}
}

func transformRotUnitary(x, y []float32, c, s archsimd.Float32x4) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+8 <= len(x); i += 8 {
		x0 := loadFloat32x4(xData, i)
		x1 := loadFloat32x4(xData, i+4)
		y0 := loadFloat32x4(yData, i)
		y1 := loadFloat32x4(yData, i+4)
		storeFloat32x4(xData, i, x0.Mul(c).Add(y0.Mul(s)))
		storeFloat32x4(xData, i+4, x1.Mul(c).Add(y1.Mul(s)))
		storeFloat32x4(yData, i, y0.Mul(c).Sub(x0.Mul(s)))
		storeFloat32x4(yData, i+4, y1.Mul(c).Sub(x1.Mul(s)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx*c.GetElem(0)+vy*s.GetElem(0), vy*c.GetElem(0)-vx*s.GetElem(0)
	}
}

func transformUnitary(x, y []float32, h11, h12, h21, h22 archsimd.Float32x4) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+8 <= len(x); i += 8 {
		x0 := loadFloat32x4(xData, i)
		x1 := loadFloat32x4(xData, i+4)
		y0 := loadFloat32x4(yData, i)
		y1 := loadFloat32x4(yData, i+4)
		storeFloat32x4(xData, i, x0.Mul(h11).Add(y0.Mul(h12)))
		storeFloat32x4(xData, i+4, x1.Mul(h11).Add(y1.Mul(h12)))
		storeFloat32x4(yData, i, x0.Mul(h21).Add(y0.Mul(h22)))
		storeFloat32x4(yData, i+4, x1.Mul(h21).Add(y1.Mul(h22)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx*h11.GetElem(0)+vy*h12.GetElem(0), vx*h21.GetElem(0)+vy*h22.GetElem(0)
	}
}

func transformOffDiagonalUnitary(x, y []float32, h12, h21 archsimd.Float32x4) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+8 <= len(x); i += 8 {
		x0 := loadFloat32x4(xData, i)
		x1 := loadFloat32x4(xData, i+4)
		y0 := loadFloat32x4(yData, i)
		y1 := loadFloat32x4(yData, i+4)
		storeFloat32x4(xData, i, x0.Add(y0.Mul(h12)))
		storeFloat32x4(xData, i+4, x1.Add(y1.Mul(h12)))
		storeFloat32x4(yData, i, x0.Mul(h21).Add(y0))
		storeFloat32x4(yData, i+4, x1.Mul(h21).Add(y1))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx+vy*h12.GetElem(0), vx*h21.GetElem(0)+vy
	}
}

func transformDiagonalUnitary(x, y []float32, h11, h22 archsimd.Float32x4) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+8 <= len(x); i += 8 {
		x0 := loadFloat32x4(xData, i)
		x1 := loadFloat32x4(xData, i+4)
		y0 := loadFloat32x4(yData, i)
		y1 := loadFloat32x4(yData, i+4)
		storeFloat32x4(xData, i, x0.Mul(h11).Add(y0))
		storeFloat32x4(xData, i+4, x1.Mul(h11).Add(y1))
		storeFloat32x4(yData, i, x0.Neg().Add(y0.Mul(h22)))
		storeFloat32x4(yData, i+4, x1.Neg().Add(y1.Mul(h22)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx*h11.GetElem(0)+vy, -vx+vy*h22.GetElem(0)
	}
}
