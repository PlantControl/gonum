// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -lblas
#include <lapack.h>
#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "GEMM bridge requires LP64");
extern void dgemm_(char*, char*, lapack_int*, lapack_int*, lapack_int*,
 double*, double*, lapack_int*, double*, lapack_int*, double*, double*, lapack_int*, size_t, size_t);
extern void sgemm_(char*, char*, lapack_int*, lapack_int*, lapack_int*,
 float*, float*, lapack_int*, float*, lapack_int*, float*, float*, lapack_int*, size_t, size_t);
static void run_dgemm(char ta, char tb, lapack_int m, lapack_int n, lapack_int k,
 double alpha, double *a, lapack_int lda, double *b, lapack_int ldb,
 double beta, double *c, lapack_int ldc) {
 dgemm_(&tb,&ta,&n,&m,&k,&alpha,b,&ldb,a,&lda,&beta,c,&ldc,1,1);
}
static void run_sgemm(char ta, char tb, lapack_int m, lapack_int n, lapack_int k,
 float alpha, float *a, lapack_int lda, float *b, lapack_int ldb,
 float beta, float *c, lapack_int ldc) {
 sgemm_(&tb,&ta,&n,&m,&k,&alpha,b,&ldb,a,&lda,&beta,c,&ldc,1,1);
}
*/
import "C"

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/blas"
)

// Dgemm computes a row-major matrix product using the reference BLAS.
func (Implementation) Dgemm(ta, tb blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	C.run_dgemm(C.char(ta), C.char(tb), C.lapack_int(m), C.lapack_int(n), C.lapack_int(k), C.double(alpha),
		(*C.double)(unsafe.Pointer(unsafe.SliceData(a))), C.lapack_int(lda),
		(*C.double)(unsafe.Pointer(unsafe.SliceData(b))), C.lapack_int(ldb), C.double(beta),
		(*C.double)(unsafe.Pointer(unsafe.SliceData(c))), C.lapack_int(ldc))
}

// Sgemm computes a row-major matrix product using the reference BLAS.
func (Implementation) Sgemm(ta, tb blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	C.run_sgemm(C.char(ta), C.char(tb), C.lapack_int(m), C.lapack_int(n), C.lapack_int(k), C.float(alpha),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(a))), C.lapack_int(lda),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(b))), C.lapack_int(ldb), C.float(beta),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(c))), C.lapack_int(ldc))
}
