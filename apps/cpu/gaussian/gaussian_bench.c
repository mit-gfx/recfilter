/**
 * \file gaussian_bench.c
 * \brief Benchmark for testing Gaussian convolution algorithms
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2013, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * \mainpage
 *
 * Version 20131215 (Dec 15, 2013) \n
 * Pascal Getreuer, <getreuer@cmla.ens-cachan.fr>, CMLA, ENS Cachan
 *
 * Please cite IPOL article "A Survey of Gaussian Convolution Algorithms" if
 * you publish results obtained with this software. This software was written
 * by Pascal Getreuer and is distributed under the terms of the GPL or
 * simplified BSD license. Future releases and updates will be posted at
 * <http://dev.ipol.im/~getreuer/code/>.
 *
 * \section Overview
 *
 * This C source code accompanies with Image Processing On Line (IPOL) article
 * "A Survey of Gaussian Convolution Algorithms" by Pascal Getreuer. The
 * article and online demo can be found at <http://www.ipol.im>. This work
 * surveys algorithms for computing and efficiently approximating Gaussian
 * convolution,
 * \f[ u(x)=(G_\sigma*f)(x):=\int_{\mathbb{R}^d}G_\sigma(x-y)f(y)\,dy,\f]
 * where f is the input signal, u is the filtered signal, and
 * \f$ G_\sigma \f$ is the Gaussian with standard deviation \f$ \sigma \f$,
 * \f[ G_\sigma(x)=(2\pi\sigma^2)^{-d/2}\exp\left(-\frac{\lVert x
\rVert_2^2}{2\sigma^2}\right). \f]
 *
 * \section Algorithms
 *  - \ref fir_gaussian
 *  - \ref dct_gaussian
 *  - \ref box_gaussian
 *  - \ref ebox_gaussian
 *  - \ref sii_gaussian
 *  - \ref am_gaussian
 *  - \ref deriche_gaussian
 *  - \ref vyv_gaussian
 *
 * \section License
 * This code is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 *
 * \section Compilation
 * A makefile makefile.gcc is included for compilation with GCC on Linux or
 * MinGW+MSYS on Windows.
 *
 * Compiling requires the FFTW3 Fourier transform library (libfftw)
 * <http://www.fftw.org/>. For supporting additional image formats, the
 * programs can optionally be compiled with libjpeg, libpng, and/or libtiff.
 * Windows BMP images are always supported.
 *
 * Please see the readme for more detailed instructions.
 *
 * \section Use
 * \subsection gaussian_demo
 * The gaussian_demo.c program applies 2D convolution using any algorithm
 * on an image.
 *
 * \par Syntax
\verbatim
gaussian_demo [options] <input> <output>
\endverbatim
 *
 * where `<input>` and `<output>` are BMP files (JPEG, PNG, or TIFF files can
 * also be used if the program is compiled with libjpeg, libpng, and/or
 * libtiff).
 *
 * The algorithm and Gaussian standard deviation are configured through the
 * command line options:
 *
 * | Option        | Description                                         |
 * |---------------|-----------------------------------------------------|
 * | `-a <algo>`   | algorithm to use, choices listed below              |
 * | `-s <number>` | sigma, standard deviation of the Gaussian           |
 * | `-K <number>` | specifies number of steps (box, ebox, sii, am)      |
 * | `-t <number>` | accuracy tolerance (fir, am, deriche, vyv)          |
 *
 * Specified with the `-a <algo>` option, choices of algorithm are
*
 * | `<algo>`      | Description                                         |
 * |---------------|-----------------------------------------------------|
 * | `fir`         | FIR approximation, tol = kernel accuracy            |
 * | `dct`         | DCT-based convolution                               |
 * | `box`         | box filtering, K = # passes                         |
 * | `ebox`        | extended box filtering, K = # passes                |
 * | `sii`         | stacked integral images, K = # boxes                |
 * | `am`          | Alvarez-Mazorra using regression on q, K = # passes |
 * | `deriche`     | Deriche recursive filtering, K = order              |
 * | `vyv`         | Vliet-Young-Verbeek recursive filtering, K = order  |
 *
 * \subsection gaussian_bench
 * The gaussian_bench.c program tests the speed, accuracy, and impulse
 * response of any of the Gaussian convolution algorithms.
 *
 * \par Syntax
\verbatim
gaussian_bench [bench type] [options]
\endverbatim
 *
 * The program can run three different bench types:
 *
 * | Bench type    | Description                                         |
 * |---------------|-----------------------------------------------------|
 * | `speed`       | measure computation time                            |
 * | `accuracy`    | measure \f$ \ell^\infty \f$ operator norm error     |
 * | `impulse`     | compute impulse response, written to bench.out      |
 *
 * The algorithm and Gaussian standard deviation are specified using the same
 * options as with the gaussian_demo program. Additionally, the following
 * options configure the benchmark:
 *
 * | Option        | Description                                         |
 * |---------------|-----------------------------------------------------|
 * | `-N <number>` | signal length                                       |
 * | `-r <number>` | (for speed bench) number of runs                    |
 * | `-n <number>` | (for impulse bench) position of the impulse         |
 *
 * \subsection Examples
\verbatim
# Perform Gaussian convolution on the image einstein.png
./gaussian_demo -s 5 -a deriche -K 3 -t 1e-6 einstein.png blurred.png

# Compute the L^infty operator error
./gaussian_bench accuracy -N 1000 -s 5 -a deriche -K 3 -t 1e-6
\endverbatim
Further examples of using these programs are included in the shell scripts
demo.sh and bench.sh.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "basic.h"
#include "strategy_gaussian_conv.h"
#include "filter_util.h"

/** \brief Output file for impulse test */
#define OUTPUT_FILE     "impulse.txt"

#ifndef M_SQRT2PI
/** \brief The constant sqrt(2 pi) */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif


/** \brief Print program usage help message */
void print_usage()
{
    puts("Gaussian benchmark, P. Getreuer 2012-2013");
#ifdef NUM_SINGLE
    puts("Configuration: single-precision computation");
#else
    puts("Configuration: double-precision computation");
#endif
    puts("\nSyntax: gaussian_bench [bench type] [options] [output]\n");
    puts("Bench type:");
    puts("   speed         measure computation time");
    puts("   accuracy      measure L^infty operator norm error");
    puts("   impulse       impulse response, written to " OUTPUT_FILE "\n");
    puts("Options:");
    puts("   -a <algo>     algorithm to use, choices are");
    puts("                 fir     FIR approximation, tol = kernel accuracy");
    puts("                 dct     DCT-based convolution");
    puts("                 box     box filtering, K = # passes");
    puts("                 sii     stacked integral images, K = # boxes");
    puts("                 am_orig Alvarez-Mazorra original method,");
    puts("                         K = # passes, tol = boundary accuracy");
    puts("                 am      Alvarez-Mazorra using regression on q,");
    puts("                         K = # passes, tol = boundary accuracy");
    puts("                 deriche Deriche recursive filtering,");
    puts("                         K = order, tol = boundary accuracy");
    puts("                 vyv     Vliet-Young-Verbeek recursive filtering,");
    puts("                         K = order, tol = boundary accuracy");
    puts("   -s <number>   sigma, standard deviation of the Gaussian");
    puts("   -K <number>   specifies number of steps (box, sii, am)");
    puts("   -t <number>   accuracy tolerance (fir, am, deriche, vyv)");
    puts("   -N <number>   signal length\n");
    puts("   -r <number>   (speed bench) number of runs");
    puts("   -n <number>   (impulse bench) position of the impulse\n");
}

/** \brief struct of program parameters */
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

int parse_params(program_params *param, int argc, char **argv);

int speed_test(program_params p, num *output, num *input)
{
    gconv *g = NULL;
    unsigned long time_start, time_stop;
    long run;

    if (!(g = gconv_plan(output, input, p.N, 1,
        p.algo, p.sigma, p.K, p.tol)))
        return 0;

    time_start = millisecond_timer();

    for (run = 0; run < p.num_runs; ++run)
        gconv_execute(g);

    time_stop = millisecond_timer();
    printf("%.5e\n",
        ((double)(time_stop - time_start)) / p.num_runs);
    gconv_free(g);
    return 1;
}

void make_impulse_signal(num *signal, long N, long n0)
{
    long n;

    for (n = 0; n < N; ++n)
        signal[n] = 0;

    signal[n0] = 1;
}

int accuracy_test(program_params p, num *output, num *input)
{
    double *error_sums = NULL;
    num *output0 = NULL;
    gconv *g0 = NULL, *g = NULL;
    double linf_norm = 0.0;
    long m, n;
    int success = 0;

    if (!(error_sums = (double *)malloc(sizeof(double) * p.N))
        || !(output0 = (num *)malloc(sizeof(num) * p.N))
        || !(g0 = gconv_plan(output0, input, p.N, 1,
            "fir", p.sigma, p.K, 1e-15))
        || !(g = gconv_plan(output, input, p.N, 1,
            p.algo, p.sigma, p.K, p.tol)))
        goto fail;

    for (n = 0; n < p.N; ++n)
        error_sums[n] = 0.0;

    for (n = 0; n < p.N; ++n)
    {
        make_impulse_signal(input, p.N, n);
        gconv_execute(g0);
        gconv_execute(g);

        for (m = 0; m < p.N; ++m)
            error_sums[m] += fabs((double)output0[m] - (double)output[m]);
    }

    for (n = 0; n < p.N; ++n)
        if (error_sums[n] > linf_norm)
            linf_norm = error_sums[n];

    printf("%.8e\n", linf_norm);
    success = 1;
fail:
    gconv_free(g0);
    gconv_free(g);
    if (output0)
        free(output0);
    if (error_sums)
        free(error_sums);
    return success;
}

int write_output(const char *filename, num *output, num *exact, long N)
{
    FILE *f;
    long n;

    if (!(f = fopen(filename, "wt")))
        return 0;

    fprintf(f, "# n\toutput value\texact value\n");

    for (n = 0; n < N; ++n)
        fprintf(f, "%ld\t%.16e\t%.16e\n", n, output[n], exact[n]);

    fclose(f);
    return 1;
}

int impulse_test(program_params p, num *output, num *input)
{
    num *exact = NULL;
    gconv *g = NULL, *g_exact = NULL;
    int success = 0;

    if (!(exact = (num *)malloc(sizeof(num) * p.N))
        || !(g = gconv_plan(output, input, p.N, 1,
            p.algo, p.sigma, p.K, p.tol))
        || !(g_exact = gconv_plan(exact, input, p.N, 1,
            "fir", p.sigma, p.K, 1e-15)))
        goto fail;

    make_impulse_signal(input, p.N, p.n0);
    gconv_execute(g);
    gconv_execute(g_exact);

    if (!write_output(OUTPUT_FILE, output, exact, p.N))
        goto fail;

    success = 1;
fail:
    gconv_free(g_exact);
    gconv_free(g);
    if (exact)
        free(exact);
    return success;
}

int main(int argc, char **argv)
{
    program_params param;
    num *input = NULL;
    num *output = NULL;
    int success = 0;

    if (!parse_params(&param, argc, argv))
        return 1;

    /* Create input signal */
    if (!(input = (num *)malloc(sizeof(num) * param.N))
        || !(output = (num *)malloc(sizeof(num) * param.N)))
    {
        fprintf(stderr, "Out of memory\n");
        goto fail;
    }

    if (!strcmp(param.bench_type, "speed"))
    {
        if (!speed_test(param, output, input))
            goto fail;
    }
    else if (!strcmp(param.bench_type, "accuracy"))
    {
        if (!accuracy_test(param, output, input))
            goto fail;
    }
    else if (!strcmp(param.bench_type, "impulse"))
    {
        if (!impulse_test(param, output, input))
            goto fail;
    }
    else
    {
        fprintf(stderr, "Invalid bench type \"%s\"\n", param.bench_type);
        goto fail;
    }

    success = 1;
fail:
    if (!success)
        fprintf(stderr, "Bench %s failed\n", param.bench_type);

    if (output)
        free(output);
    if (input)
        free(input);
    return !success;
}

int parse_params(program_params *param, int argc, char **argv)
{
    static const char *default_algo = (const char *)"exact";
    char *option_string;
    char option_char;
    int i;

    if (argc < 2)
    {
        print_usage();
        return 0;
    }

    param->bench_type = argv[1];

    /* Set parameter defaults. */
    param->N = 100;
    param->n0 = -1;
    param->num_runs = 10;
    param->sigma = 5;
    param->algo = default_algo;
    param->K = 3;
    param->tol = 1e-2;

    for (i = 2; i < argc;)
    {
        if (argv[i] && argv[i][0] == '-')
        {
            if ((option_char = argv[i][1]) == 0)
            {
                fprintf(stderr, "Invalid parameter format.\n");
                return 0;
            }

            if (argv[i][2])
                option_string = &argv[i][2];
            else if (++i < argc)
                option_string = argv[i];
            else
            {
                fprintf(stderr, "Invalid parameter format.\n");
                return 0;
            }

            switch (option_char)
            {
            case 'a':   /* Read algorithm. */
                param->algo = option_string;
                break;
            case 's':   /* Read sigma parameter. */
                param->sigma = atof(option_string);

                if (param->sigma < 0)
                {
                    fprintf(stderr, "sigma must be positive.\n");
                    return 0;
                }
                break;
            case 'K':   /* Read number of steps. */
                param->K = atoi(option_string);

                if (param->K <= 0)
                {
                    fprintf(stderr, "K must be positive.\n");
                    return 0;
                }
                break;
            case 't':   /* Read tolerance. */
                param->tol = atof(option_string);

                if (param->tol < 0)
                {
                    fprintf(stderr, "Tolerance must be positive.\n");
                    return 0;
                }
                break;
            case 'N':   /* Input length. */
                param->N = atoi(option_string);

                if (param->N < 0)
                {
                    fprintf(stderr, "Signal length must be positive.\n");
                    return 0;
                }
                break;
            case 'r':   /* Read number of runs. */
                param->num_runs = (long)atof(option_string);

                if (param->num_runs <= 0)
                {
                    fprintf(stderr, "Number of runs must be positive.\n");
                    return 0;
                }
                break;
            case 'n':   /* Impulse position. */
                param->n0 = atoi(option_string);
                break;
            case '-':
                print_usage();
                return 0;
            default:
                if (isprint(option_char))
                    fprintf(stderr, "Unknown option \"-%c\".\n", option_char);
                else
                    fprintf(stderr, "Unknown option.\n");

                return 0;
            }

            i++;
        }
        else
            i++;
    }

    if (param->n0 < 0 || param->n0 >= param->N)
        param->n0 = (param->N + 1) / 2;

    return 1;
}
