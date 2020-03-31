#!/bin/sh

nsys profile --output=profile-all --trace=cuda,nvtx,mpi \
             --mpi-impl=openmpi --stats=true -e _="$(command -v nsys)" \
    mpirun -np 2 --output-filename output \
        julia --project=. try.jl
