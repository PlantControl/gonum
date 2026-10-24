#!/bin/sh
# Copyright ©2026 The Gonum Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# check.sh runs the netlib-tagged tests and fails unless every listed
# top-level test reported PASS or SKIP.
#
# Reference LAPACK's XERBLA stops the process with exit status 0 when a
# routine is given an illegal argument, so a bad oracle call ends the test
# binary early and "go test" still prints ok. This script compares the
# number of tests "go test -list" reports with the number that finished,
# and fails on any XERBLA "illegal value" message.
#
# Run it from the module root, with any go test flags appended:
#
#	sh lapack/gonum/internal/netlib/check.sh
#	sh lapack/gonum/internal/netlib/check.sh -count=1
#
# The packages default to ./lapack/...; set PKGS to override.

set -u
pkgs=${PKGS:-./lapack/...}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/netlibcheck.XXXXXX") || exit 1
trap 'rm -rf "$tmp"' EXIT
status=0
for pkg in $(go list -tags netlib $pkgs); do
	want=$(go test -tags netlib -list '.*' "$@" "$pkg" | grep -c '^Test')
	if [ "$want" -eq 0 ]; then
		continue
	fi
	go test -tags netlib -v "$@" "$pkg" >"$tmp/out" 2>&1
	code=$?
	got=$(grep -cE '^--- (PASS|SKIP)' "$tmp/out")
	fail=$(grep -c '^--- FAIL' "$tmp/out")
	xerbla=$(grep -ci 'illegal value' "$tmp/out")
	echo "$pkg: listed=$want finished=$got failed=$fail xerbla=$xerbla exit=$code"
	if [ "$code" -ne 0 ] || [ "$got" -ne "$want" ] || [ "$fail" -ne 0 ] || [ "$xerbla" -ne 0 ]; then
		grep -iE '^--- FAIL|illegal value|^panic' "$tmp/out" | head -20
		status=1
	fi
done
if [ "$status" -ne 0 ]; then
	echo "netlib check FAILED"
else
	echo "netlib check ok"
fi
exit "$status"
