#!/bin/bash
#PBS -l walltime=00:15:00 -l nodes=1:ppn=20

cd P1.3_seed/Opt_tech

module load intel/14.0
module load mkl/11.1

./a.out > results_mm_series.txt
./a.out >> results_mm_series.txt
./a.out >> results_mm_series.txt
