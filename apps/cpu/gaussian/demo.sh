#! /bin/sh
# Demo script for gaussian_demo
# Pascal Getreuer 2013

# Echo shell commands
set -v

# Set parameters
sigma=5
algo=box
K=3
tol=1e-2

# Perform Gaussian convolution on the image einstein.png
./gaussian_demo -s $sigma -a $algo -K $K -t $tol world.png blurred.png
