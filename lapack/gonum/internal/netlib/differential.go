// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas
#include <lapacke.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static lapack_logical select_negative(const double *ar, const double *ai, const double *beta) {
	(void)ai;
	return *beta != 0 && *ar < 0;
}

static lapack_logical select_unit_circle(const double *ar, const double *ai, const double *beta) {
 return hypot(*ar, *ai) < fabs(*beta);
}

static lapack_int run_dgges_work(char jobvsl, char jobvsr, lapack_int selection, lapack_int n,
 double *a, double *b, lapack_int *sdim, double *ar, double *ai,
 double *beta, double *vsl, double *vsr, double *work, lapack_int lwork, lapack_logical *bwork) {
 return LAPACKE_dgges_work(LAPACK_COL_MAJOR, jobvsl, jobvsr, selection ? 'S' : 'N',
  selection == 2 ? select_unit_circle : select_negative,
  n, a, n, b, n, sdim, ar, ai, beta, vsl, n, vsr, n, work, lwork, bwork);
}

static lapack_logical select_large_alpha(const double *ar, const double *ai, const double *beta) {
	(void)ai;
	(void)beta;
	return *ar > 1e-299;
}

static void row_to_col(int n, const double *src, double *dst);
static void col_to_row(int n, const double *src, double *dst);

static lapack_int run_dgges(char jobvsl, char jobvsr, lapack_int n,
	double *a, double *b, lapack_int dosort,
	lapack_int *sdim, double *ar, double *ai, double *beta, double *vsl, double *vsr) {
	lapack_int info = LAPACKE_dgges(LAPACK_ROW_MAJOR, jobvsl, jobvsr, dosort ? 'S' : 'N',
		dosort ? select_negative : NULL, n, a, n, b, n, sdim, ar, ai, beta,
		vsl, n, vsr, n);
	return info;
}

static lapack_int run_dgges_large_alpha(lapack_int n, double *a, double *b,
		lapack_int *sdim, double *ar, double *ai, double *beta, double *vsl, double *vsr) {
	return LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', 'S', select_large_alpha,
		n, a, n, b, n, sdim, ar, ai, beta, vsl, n, vsr, n);
}

static lapack_int query_dtgsen(lapack_int ijob, lapack_int n, lapack_int lwork,
		lapack_int liwork, double *work_out, lapack_int *iwork_out) {
	lapack_logical select[2] = {1, 0};
	double a[4] = {1, 0, 0, 2};
	double b[4] = {1, 0, 0, 1};
	double ar[2] = {0, 0}, ai[2] = {0, 0}, beta[2] = {0, 0};
	double q[4] = {1, 0, 0, 1}, z[4] = {1, 0, 0, 1};
	double dif[2] = {0, 0}, work[1] = {-1}, pl = 0, pr = 0;
	lapack_int iwork[1] = {-1}, m = 0;
	lapack_int ld = n > 0 ? n : 1;
	lapack_int info = LAPACKE_dtgsen_work(LAPACK_ROW_MAJOR, ijob, 0, 0,
		select, n, a, ld, b, ld, ar, ai, beta, q, ld, z, ld, &m, &pl, &pr,
		dif, work, lwork, iwork, liwork);
	*work_out = work[0];
	*iwork_out = iwork[0];
	return info;
}

static lapack_int run_dhgeqz(char job, char compq, char compz, lapack_int n,
		lapack_int ilo, lapack_int ihi, double *h, double *t, double *ar,
		double *ai, double *beta, double *q, double *z) {
	lapack_int lwork = n > 0 ? n : 1;
	double *work = malloc((size_t)lwork*sizeof(double));
	lapack_int info = LAPACKE_dhgeqz_work(LAPACK_ROW_MAJOR, job, compq, compz,
		n, ilo, ihi, h, n, t, n, ar, ai, beta, q, n, z, n, work, lwork);
	free(work);
	return info;
}

static lapack_int run_dtgsen_singular(void) {
	lapack_logical select[2] = {1, 0};
	double a[4] = {1, 0, 0, 1};
	double b[4] = {1, 0, 0, 1};
	double ar[2] = {0, 0}, ai[2] = {0, 0}, beta[2] = {0, 0};
	double q[4] = {1, 0, 0, 1}, z[4] = {1, 0, 0, 1};
	double dif[2] = {0, 0}, pl = 0, pr = 0;
	lapack_int m = 0;
	return LAPACKE_dtgsen(LAPACK_ROW_MAJOR, 1, 0, 0, select, 2, a, 2, b, 2,
		ar, ai, beta, q, 2, z, 2, &m, &pl, &pr, dif);
}

static lapack_int run_dtgsen_case(lapack_int ijob, lapack_logical wantq,
		lapack_logical wantz, uint64_t mask, lapack_int n, double *a, double *b,
		double *ar, double *ai, double *beta, double *q, double *z, lapack_int *m,
		double *pl, double *pr, double *dif) {
	lapack_logical *select = malloc((size_t)n*sizeof(lapack_logical));
	for (lapack_int i = 0; i < n; i++) select[i] = (mask >> i) & 1;
	lapack_int info = LAPACKE_dtgsen(LAPACK_ROW_MAJOR, ijob, wantq, wantz,
		select, n, a, n, b, n, ar, ai, beta, q, n, z, n, m, pl, pr, dif);
	free(select);
	return info;
}

extern void dtgex2_(int*, int*, int*, double*, int*, double*, int*, double*, int*,
	double*, int*, int*, int*, int*, double*, int*, int*);
extern void dgetc2_(int*, double*, int*, int*, int*, int*);
extern void dlatdf_(int*, int*, double*, int*, double*, double*, double*, int*, int*);
extern void dlanv2_(double*, double*, double*, double*, double*, double*,
		double*, double*, double*, double*);
extern void dtgsy2_(char*, int*, int*, int*, double*, int*, double*, int*,
		double*, int*, double*, int*, double*, int*, double*, int*, double*,
		double*, double*, int*, int*, int*);

static void row_to_col(int n, const double *src, double *dst) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) dst[j*n+i] = src[i*n+j];
}

static void col_to_row(int n, const double *src, double *dst) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) dst[i*n+j] = src[j*n+i];
}

static void row_to_col_rect(int m, int n, const double *src, double *dst) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) dst[i+j*m] = src[i*n+j];
}

static void col_to_row_rect(int m, int n, const double *src, double *dst) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) dst[i*n+j] = src[i+j*m];
}

static int run_dlatdf(int job, int n, const double *z, double *rhs,
		double *rdsum, double *rdscal) {
	double *zc = malloc((size_t)n*n*sizeof(double));
	int *ipiv = malloc((size_t)n*sizeof(int));
	int *jpiv = malloc((size_t)n*sizeof(int));
	row_to_col(n, z, zc);
	int info = 0, ldz = n;
	dgetc2_(&n, zc, &ldz, ipiv, jpiv, &info);
	dlatdf_(&job, &n, zc, &ldz, rhs, rdsum, rdscal, ipiv, jpiv);
	free(zc); free(ipiv); free(jpiv);
	return info;
}

static int run_dtgsy2(char trans, int ijob, int m, int n,
		const double *a, const double *b, double *c, const double *d,
		const double *e, double *f, double *scale, double *rdsum,
		double *rdscal, int *pq) {
	double *ac = malloc((size_t)m*m*sizeof(double));
	double *bc = malloc((size_t)n*n*sizeof(double));
	double *cc = malloc((size_t)m*n*sizeof(double));
	double *dc = malloc((size_t)m*m*sizeof(double));
	double *ec = malloc((size_t)n*n*sizeof(double));
	double *fc = malloc((size_t)m*n*sizeof(double));
	int *iwork = malloc((size_t)(m+n+2)*sizeof(int));
	row_to_col(m, a, ac); row_to_col(n, b, bc);
	row_to_col_rect(m, n, c, cc);
	row_to_col(m, d, dc); row_to_col(n, e, ec);
	row_to_col_rect(m, n, f, fc);
	int info = 0, lda = m, ldb = n, ldc = m, ldd = m, lde = n, ldf = m;
	dtgsy2_(&trans, &ijob, &m, &n, ac, &lda, bc, &ldb, cc, &ldc,
		dc, &ldd, ec, &lde, fc, &ldf, scale, rdsum, rdscal, iwork, pq, &info);
	col_to_row_rect(m, n, cc, c); col_to_row_rect(m, n, fc, f);
	free(ac); free(bc); free(cc); free(dc); free(ec); free(fc); free(iwork);
	return info;
}

static int run_dtgex2(int n, double *a, double *b, int j1, int n1, int n2) {
	double *ac = malloc((size_t)n*n*sizeof(double));
	double *bc = malloc((size_t)n*n*sizeof(double));
	double *q = calloc((size_t)n*n, sizeof(double));
	double *z = calloc((size_t)n*n, sizeof(double));
	int lwork = n*(n1+n2);
	if (lwork < 2*(n1+n2)*(n1+n2)) lwork = 2*(n1+n2)*(n1+n2);
	double *work = malloc((size_t)lwork*sizeof(double));
	for (int i = 0; i < n; i++) q[i*n+i] = z[i*n+i] = 1;
	row_to_col(n, a, ac);
	row_to_col(n, b, bc);
	int want = 1, lda = n, info = 0, fj1 = j1 + 1;
	dtgex2_(&want, &want, &n, ac, &lda, bc, &lda, q, &lda, z, &lda,
		&fj1, &n1, &n2, work, &lwork, &info);
	col_to_row(n, ac, a);
	col_to_row(n, bc, b);
	free(ac); free(bc); free(q); free(z); free(work);
	return info;
}

static lapack_int run_dtgexc(lapack_int n, double *a, double *b, lapack_int *ifst, lapack_int *ilst) {
	double *ac = malloc((size_t)n*n*sizeof(double));
	double *bc = malloc((size_t)n*n*sizeof(double));
	double *q = calloc((size_t)n*n, sizeof(double));
	double *z = calloc((size_t)n*n, sizeof(double));
	lapack_int lwork = n <= 1 ? 1 : 4*n + 16;
	double *work = malloc((size_t)lwork*sizeof(double));
	for (int i = 0; i < n; i++) q[i*n+i] = z[i*n+i] = 1;
	row_to_col(n, a, ac);
	row_to_col(n, b, bc);
	lapack_logical want = 1;
	lapack_int lda = n, info = 0;
	LAPACK_dtgexc(&want, &want, &n, ac, &lda, bc, &lda, q, &lda, z, &lda,
		ifst, ilst, work, &lwork, &info);
	col_to_row(n, ac, a);
	col_to_row(n, bc, b);
	free(ac); free(bc); free(q); free(z); free(work);
	return info;
}
*/
import "C"

import (
	"unsafe"

	"plantcontrol.org/v1/gonum/lapack"
)

func Dgges(n int, a, b []float64, dosort bool, ar, ai, beta, vsl, vsr []float64) (sdim, info int) {
	return DggesOptions('V', 'V', n, a, b, dosort, ar, ai, beta, vsl, vsr)
}

func DggesOptions(jobvsl, jobvsr byte, n int, a, b []float64, dosort bool,
	ar, ai, beta, vsl, vsr []float64) (sdim, info int) {
	var csdim C.lapack_int
	cinfo := C.run_dgges(C.char(jobvsl), C.char(jobvsr), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		C.lapack_int(boolInt(dosort)), &csdim, (*C.double)(unsafe.Pointer(&ar[0])),
		(*C.double)(unsafe.Pointer(&ai[0])), (*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&vsl[0])), (*C.double)(unsafe.Pointer(&vsr[0])))
	return int(csdim), int(cinfo)
}

func DggesLargeAlpha(n int, a, b []float64, ar, ai, beta, vsl, vsr []float64) (sdim, info int) {
	var csdim C.lapack_int
	cinfo := C.run_dgges_large_alpha(C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		&csdim, (*C.double)(unsafe.Pointer(&ar[0])), (*C.double)(unsafe.Pointer(&ai[0])),
		(*C.double)(unsafe.Pointer(&beta[0])), (*C.double)(unsafe.Pointer(&vsl[0])),
		(*C.double)(unsafe.Pointer(&vsr[0])))
	return int(csdim), int(cinfo)
}

func Dggbal(job byte, n int, a, b, lscale, rscale []float64) (ilo, ihi, info int) {
	var cilo, cihi C.lapack_int
	cinfo := C.LAPACKE_dggbal(C.int(C.LAPACK_ROW_MAJOR), C.char(job), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&a[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&b[0])), C.lapack_int(n), &cilo, &cihi,
		(*C.double)(unsafe.Pointer(&lscale[0])), (*C.double)(unsafe.Pointer(&rscale[0])))
	ilo = int(cilo) - 1
	ihi = int(cihi) - 1
	for i := 0; i < n; i++ {
		if i < ilo || i > ihi {
			lscale[i]--
			rscale[i]--
		}
	}
	return ilo, ihi, int(cinfo)
}

func Dggbak(job, side byte, n, ilo, ihi int, lscale, rscale []float64, m int, v []float64) int {
	lcopy := append([]float64(nil), lscale...)
	rcopy := append([]float64(nil), rscale...)
	for i := 0; i < n; i++ {
		if i < ilo || i > ihi {
			lcopy[i]++
			rcopy[i]++
		}
	}
	return int(C.LAPACKE_dggbak(C.int(C.LAPACK_ROW_MAJOR), C.char(job), C.char(side),
		C.lapack_int(n), C.lapack_int(ilo+1), C.lapack_int(ihi+1),
		(*C.double)(unsafe.Pointer(&lcopy[0])), (*C.double)(unsafe.Pointer(&rcopy[0])),
		C.lapack_int(m), (*C.double)(unsafe.Pointer(&v[0])), C.lapack_int(m)))
}

func Dgghrd(compq, compz byte, n, ilo, ihi int, a, b, q, z []float64) int {
	return int(C.LAPACKE_dgghrd(C.int(C.LAPACK_ROW_MAJOR), C.char(compq), C.char(compz),
		C.lapack_int(n), C.lapack_int(ilo+1), C.lapack_int(ihi+1),
		(*C.double)(unsafe.Pointer(&a[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&b[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&q[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&z[0])), C.lapack_int(n)))
}

func DtgsenWorkspace(ijob, n, lwork, liwork int) (work float64, iwork, info int) {
	var cwork C.double
	var ciwork C.lapack_int
	cinfo := C.query_dtgsen(C.lapack_int(ijob), C.lapack_int(n), C.lapack_int(lwork), C.lapack_int(liwork),
		&cwork, &ciwork)
	return float64(cwork), int(ciwork), int(cinfo)
}

func Dhgeqz(job, compq, compz byte, n, ilo, ihi int, h, t, ar, ai, beta, q, z []float64) int {
	return int(C.run_dhgeqz(C.char(job), C.char(compq), C.char(compz), C.lapack_int(n),
		C.lapack_int(ilo+1), C.lapack_int(ihi+1), (*C.double)(unsafe.Pointer(&h[0])),
		(*C.double)(unsafe.Pointer(&t[0])), (*C.double)(unsafe.Pointer(&ar[0])),
		(*C.double)(unsafe.Pointer(&ai[0])), (*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&q[0])), (*C.double)(unsafe.Pointer(&z[0]))))
}

func DtgsenSingular() int {
	return int(C.run_dtgsen_singular())
}

func Dtgsen(ijob int, wantq, wantz bool, selected []bool, n int,
	a, b, ar, ai, beta, q, z []float64) (m int, pl, pr float64, dif [2]float64, info int) {
	var mask uint64
	for i, selected := range selected {
		if selected {
			mask |= 1 << i
		}
	}
	var cm C.lapack_int
	var cpl, cpr C.double
	var cdif [2]C.double
	cinfo := C.run_dtgsen_case(C.lapack_int(ijob), C.lapack_logical(boolInt(wantq)),
		C.lapack_logical(boolInt(wantz)), C.uint64_t(mask), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&ar[0])), (*C.double)(unsafe.Pointer(&ai[0])),
		(*C.double)(unsafe.Pointer(&beta[0])), (*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&z[0])), &cm, &cpl, &cpr, &cdif[0])
	return int(cm), float64(cpl), float64(cpr),
		[2]float64{float64(cdif[0]), float64(cdif[1])}, int(cinfo)
}

func Dtgsyl(trans byte, ijob, m, n int,
	a, b, c, d, e, f []float64) (scale, dif float64, info int) {
	var cscale, cdif C.double
	cinfo := C.LAPACKE_dtgsyl(C.int(C.LAPACK_ROW_MAJOR), C.char(trans), C.lapack_int(ijob),
		C.lapack_int(m), C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), C.lapack_int(m),
		(*C.double)(unsafe.Pointer(&b[0])), C.lapack_int(n), (*C.double)(unsafe.Pointer(&c[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&d[0])), C.lapack_int(m), (*C.double)(unsafe.Pointer(&e[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&f[0])), C.lapack_int(n), &cscale, &cdif)
	return float64(cscale), float64(cdif), int(cinfo)
}

func Dtgsy2(trans byte, ijob, m, n int,
	a, b, c, d, e, f []float64, rdsum, rdscal float64) (scale, sum, scal float64, pq, info int) {
	var cscale C.double
	csum, cscal := C.double(rdsum), C.double(rdscal)
	var cpq C.int
	cinfo := C.run_dtgsy2(C.char(trans), C.int(ijob), C.int(m), C.int(n),
		(*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])), (*C.double)(unsafe.Pointer(&d[0])),
		(*C.double)(unsafe.Pointer(&e[0])), (*C.double)(unsafe.Pointer(&f[0])),
		&cscale, &csum, &cscal, &cpq)
	return float64(cscale), float64(csum), float64(cscal), int(cpq), int(cinfo)
}

func Dlatdf(job lapack.MaximizeNormXJob, n int, z, rhs []float64,
	rdsum, rdscal float64) (sum, scale float64, info int) {
	csum, cscale := C.double(rdsum), C.double(rdscal)
	cinfo := C.run_dlatdf(C.int(job), C.int(n), (*C.double)(unsafe.Pointer(&z[0])),
		(*C.double)(unsafe.Pointer(&rhs[0])), &csum, &cscale)
	return float64(csum), float64(cscale), int(cinfo)
}

func Dlanv2(a, b, c, d float64) (aa, bb, cc, dd, rt1r, rt1i, rt2r, rt2i, cs, sn float64) {
	ca, cb, cc_, cd := C.double(a), C.double(b), C.double(c), C.double(d)
	var r1r, r1i, r2r, r2i, ccs, csn C.double
	C.dlanv2_(&ca, &cb, &cc_, &cd, &r1r, &r1i, &r2r, &r2i, &ccs, &csn)
	return float64(ca), float64(cb), float64(cc_), float64(cd),
		float64(r1r), float64(r1i), float64(r2r), float64(r2i), float64(ccs), float64(csn)
}

func Dtgex2(n int, a, b []float64, j1, n1, n2 int) int {
	return int(C.run_dtgex2(C.int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		C.int(j1), C.int(n1), C.int(n2)))
}

func Dtgexc(n int, a, b []float64, ifst, ilst int) (ifstOut, ilstOut, info int) {
	cifst, cilst := C.lapack_int(ifst+1), C.lapack_int(ilst+1)
	cinfo := C.run_dtgexc(C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		&cifst, &cilst)
	return int(cifst) - 1, int(cilst) - 1, int(cinfo)
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// Version returns the linked LAPACK runtime version, rather than its dylib name.
func Version() (major, minor, patch int) {
	var a, b, c C.lapack_int
	C.LAPACKE_ilaver(&a, &b, &c)
	return int(a), int(b), int(c)
}

// DggesWorkspace owns preallocated scratch for column-major DGGES calls.
// Selection is 0 (none), 1 (negative real part), or 2 (inside the unit circle).
// Each instance is used by a single caller at a time. n must be positive.
type DggesWorkspace struct {
	n           int
	left, right byte
	selection   int
	work        []float64
	bwork       []C.lapack_logical
	sdim        []C.lapack_int
}

func NewDggesWorkspace(n int, left, right byte, selection int) *DggesWorkspace {
	if n < 1 || selection < 0 || selection > 2 {
		panic("netlib: invalid DGGES workspace configuration")
	}
	w := &DggesWorkspace{n: n, left: left, right: right, selection: selection,
		work: make([]float64, 1), bwork: make([]C.lapack_logical, n), sdim: make([]C.lapack_int, 1)}
	// LAPACK does not reference matrix data during a workspace query.
	dummy := make([]float64, 1)
	_, info := w.run(dummy, dummy, dummy, dummy, dummy, dummy, dummy, -1)
	if info != 0 {
		panic("netlib: DGGES workspace query failed")
	}
	w.work = make([]float64, int(w.work[0]))
	return w
}

// Run overwrites column-major matrices A and B. All slices must contain at
// least n entries, or n*n for A, B and requested vector matrices. Unrequested
// vector matrices may contain a single dummy element.
func (w *DggesWorkspace) Run(a, b, ar, ai, beta, q, z []float64) (sdim, info int) {
	n := w.n
	if len(a) < n*n || len(b) < n*n || len(ar) < n || len(ai) < n || len(beta) < n || len(q) < 1 || len(z) < 1 || w.left == 'V' && len(q) < n*n || w.right == 'V' && len(z) < n*n {
		panic("netlib: short DGGES input")
	}

	return w.run(a, b, ar, ai, beta, q, z, len(w.work))
}

func (w *DggesWorkspace) run(a, b, ar, ai, beta, q, z []float64, lwork int) (sdim, info int) {
	result := C.run_dgges_work(C.char(w.left), C.char(w.right), C.lapack_int(w.selection), C.lapack_int(w.n),
		(*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])), &w.sdim[0],
		(*C.double)(unsafe.Pointer(&ar[0])), (*C.double)(unsafe.Pointer(&ai[0])), (*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&q[0])), (*C.double)(unsafe.Pointer(&z[0])),
		(*C.double)(unsafe.Pointer(&w.work[0])), C.lapack_int(lwork), &w.bwork[0])
	return int(w.sdim[0]), int(result)
}
