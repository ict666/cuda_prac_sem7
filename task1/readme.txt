compilation:
nvcc -o task1 task1.cu
run:
mpisubmit.pl --g task1 -- "filter num" "im size"

result in ./images/output