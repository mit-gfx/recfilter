/**
 * \file gaussian_vyv_cpu.c
 * \brief Speed benchmark for Vliet-Young-Verbeek approximation for Gaussian blur; inspired by gaussian_bench.c
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "basic.h"
#include "strategy_gaussian_conv.h"
#include "filter_util.h"

typedef struct
{
    const char *bench_type;     /**< Benchmark type                        */
    long N;                     /**< Length of the input signal            */
    long num_runs;              /**< Number of runs for speed measurement  */
    long n0;                    /**< Impulse location (default = N/2)      */
    const char *algo;           /**< Name of the convolution algorithm     */
    double sigma;               /**< sigma parameter of the Gaussian       */
    int K;                      /**< Parameter K                           */
    double tol;                 /**< Tolerance                             */
} program_params;

int speed_test_2D(program_params p, num *output, num *input)
{
    gconv *g = NULL;
    unsigned long time_start, time_stop;
    long run;

    if (!(g = gconv_plan(output, input, p.N, 1,
        p.algo, p.sigma, p.K, p.tol)))
        return 0;

    time_start = millisecond_timer();

    for (run = 0; run < p.num_runs; ++run)
        gconv_execute_2D(g);

    time_stop = millisecond_timer();
    fprintf(stderr, "%ld\t%e\t%ld\n",
        p.N, ((double)(time_stop - time_start)), p.num_runs);
    gconv_free(g);
    return 1;
}

int main(int argc, char **argv)
{
    program_params param;

    num *input = NULL;
    num *output = NULL;
    int success = 0;

    long N;

    param.bench_type = (const char *)"speed2D";
    param.N = 100;
    param.n0 = (param.N + 1) / 2;
    param.num_runs = 100;
    param.sigma = 5;
    param.algo = (const char *)"vyv";
    param.K = 3;
    param.tol = 1e-2;

    if (argc == 2)
    {
        param.num_runs = (long)atof(argv[1]);
        if (param.num_runs <= 0)
        {
            fprintf(stderr, "Number of runs must be positive.\n");
            return 0;
        }
    }
    else
    {
        fprintf(stderr, "Usage: gaussian_vyv_cpu [number of benchmarking runs]\n");
        goto fail;
    }

    fprintf(stderr, "\nwidth\ttime (ms)\truns\n");
    for (N = 64; N <= 8192; N += 64)
    {
        param.N = N;

        if (!(input = (num *)realloc(input, sizeof(num) * param.N * param.N))
                || !(output = (num *)realloc(output, sizeof(num) * param.N * param.N)))
        {
            fprintf(stderr, "Out of memory\n");
            goto fail;
        }
        if (!speed_test_2D(param, output, input))
            goto fail;
    }

    success = 1;
fail:
    if (!success)
        fprintf(stderr, "Benchmark failed\n");

    if (output)
        free(output);
    if (input)
        free(input);
    return !success;
}
