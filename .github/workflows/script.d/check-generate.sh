#!/bin/bash

# Reset the tree to the current commit to handle
# any writes during the build.
git reset --hard

go generate plantcontrol.org/v1/gonum/blas
go generate plantcontrol.org/v1/gonum/blas/gonum
go generate plantcontrol.org/v1/gonum/unit
go generate plantcontrol.org/v1/gonum/unit/constant
go generate plantcontrol.org/v1/gonum/graph/formats/dot
go generate plantcontrol.org/v1/gonum/graph/formats/rdf
go generate plantcontrol.org/v1/gonum/stat/card

if [ -n "$(git diff)" ]; then	
	git diff
	exit 1
fi

rm -rf *
git reset --hard
