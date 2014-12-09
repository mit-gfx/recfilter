#! /bin/sh
# Demo script for gaussian_bench
# Pascal Getreuer 2013

# Echo shell commands
# set -v

# Set parameters
N=100
sigma=5
algo=vyv
K=3
tol=1e-6
runs=1e5

opts="-N $N -s $sigma -a $algo -K $K -t $tol -r $runs"

# Perform the convolution $runs times and get the average run time
./gaussian_bench speed $opts
./gaussian_bench speed $opts

# Compute the L^infty operator error
./gaussian_bench accuracy $opts

# Compute impulse response, saved to "impulse.txt"
./gaussian_bench impulse $opts
# Plot the response with Gnuplot
gnuplot plotimpulse.gp
