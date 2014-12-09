/**
 * \file strategy_gaussian_conv.c
 * \brief Suite of different 1D Gaussian convolution methods
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
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "strategy_gaussian_conv.h"
#include <stdlib.h>
#include <string.h>
#include "gaussian_conv_fir.h"
#include "gaussian_conv_dct.h"
#include "gaussian_conv_box.h"
#include "gaussian_conv_ebox.h"
#include "gaussian_conv_sii.h"
#include "gaussian_conv_am.h"
#include "gaussian_conv_deriche.h"
#include "gaussian_conv_vyv.h"

struct gconv_
{
    num *dest;                  /**< destination array                 */
    const num *src;             /**< sourcec array                     */
    num *buffer;                /**< workspace memory (if needed)      */
    long N;                     /**< number of samples                 */
    long stride;                /**< stride between successive samples */
    const char *algo;           /**< algorithm name                    */
    double sigma;               /**< Gaussian standard deviation       */
    int K;                      /**< steps/filter order parameter      */
    num tol;                    /**< accuracy parameter                */
    void *coeffs;               /**< algorithm-specific data           */
    void (*execute)(gconv*);    /**< algorithm execution function      */
    void (*execute_2D)(gconv*); /**< algorithm 2D execution function   */
    void (*free)(gconv*);       /**< algorithm clean up function       */
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
static int gconv_fir_plan(gconv *g);
static int gconv_dct_plan(gconv *g);
static int gconv_box_plan(gconv *g);
static int gconv_ebox_plan(gconv *g);
static int gconv_sii_plan(gconv *g);
static int gconv_am_orig_plan(gconv *g);
static int gconv_am_plan(gconv *g);
static int gconv_deriche_plan(gconv *g);
static int gconv_vyv_plan(gconv *g);
#endif

/**
 * \brief Plan a 1D Gaussian convolution
 * \param dest      destination array
 * \param src       source array
 * \param N         number of samples
 * \param stride    stride between successive samples
 * \param algo      Gaussian convolution algorithm
 * \param sigma     Gaussian standard deviation
 * \param K         algorithm steps or filter order parameter
 * \param tol       algorithm accuracy parameter
 * \return gconv pointer or NULL on failure
 */
gconv* gconv_plan(num *dest, const num *src, long N, long stride,
    const char *algo, double sigma, int K, num tol)
{
    struct gconv_algo_entry
    {
        const char *name;
        int (*plan)(gconv*);
    } algos[] = {
        {"fir",         gconv_fir_plan},
        {"dct",         gconv_dct_plan},
        {"box",         gconv_box_plan},
        {"ebox",        gconv_ebox_plan},
        {"sii",         gconv_sii_plan},
        {"am_orig",     gconv_am_orig_plan},
        {"am",          gconv_am_plan},
        {"deriche",     gconv_deriche_plan},
        {"vyv",         gconv_vyv_plan}
        };
    gconv *g;
    size_t i;

    if (!dest || !src || N <= 0 || sigma <= 0 || tol < 0 || tol > 1
        || !(g = (gconv *)malloc(sizeof(gconv))))
        return NULL;

    g->dest = dest;
    g->src = src;
    g->buffer = NULL;
    g->N = N;
    g->stride = stride;
    g->algo = algo;
    g->K = K;
    g->sigma = sigma;
    g->tol = tol;
    g->coeffs = NULL;

    for (i = 0; i < sizeof(algos)/sizeof(*algos); ++i)
        if (!strcmp(algo, algos[i].name))
        {
            if (algos[i].plan(g))
                return g;
            else
                break;
        }

    gconv_free(g);
    return NULL;
}

/**
 * \brief Execute a 1D Gaussian convolution
 * \param g     gconv* created by gconv_plan()
 */
void gconv_execute(gconv *g)
{
    g->execute(g);
}

/**
 * \brief Execute a 2D Gaussian convolution
 * \param g     gconv* created by gconv_plan_2D()
 */
void gconv_execute_2D(gconv *g)
{
    g->execute_2D(g);
}

/**
 * \brief Free memory associated with a gconv
 * \param g     gconv* created by gconv_plan()
 */
void gconv_free(gconv *g)
{
    if (g)
    {
        if (g->free)
            g->free(g);

        free(g);
    }
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/* FIR filtering. */
static void gconv_fir_execute(gconv *g);
static void gconv_fir_free(gconv *g);

static int gconv_fir_plan(gconv *g)
{
    g->execute = gconv_fir_execute;
    g->free = gconv_fir_free;
    return (g->coeffs = malloc(sizeof(fir_coeffs)))
        && fir_precomp((fir_coeffs *)g->coeffs, g->sigma, g->tol);
}

static void gconv_fir_execute(gconv *g)
{
    fir_gaussian_conv(*((fir_coeffs *)g->coeffs),
        g->dest, g->src, g->N, g->stride);
}

static void gconv_fir_free(gconv *g)
{
    if (g->coeffs)
    {
        fir_free((fir_coeffs *)g->coeffs);
        free(g->coeffs);
    }
}

/* DCT (discrete cosine transform) based convolution. */
static void gconv_dct_execute(gconv *g);
static void gconv_dct_free(gconv *g);

static int gconv_dct_plan(gconv *g)
{
    g->execute = gconv_dct_execute;
    g->free = gconv_dct_free;
    return (g->coeffs = malloc(sizeof(dct_coeffs)))
        && dct_precomp((dct_coeffs *)g->coeffs, g->dest, g->src,
            g->N, g->stride, g->sigma);
}

static void gconv_dct_execute(gconv *g)
{
    dct_gaussian_conv(*((dct_coeffs *)g->coeffs));
}

static void gconv_dct_free(gconv *g)
{
    if (g->coeffs)
    {
        dct_free((dct_coeffs *)g->coeffs);
        free(g->coeffs);
    }
}

/* Box filtering. */
static void gconv_box_execute(gconv *g);
static void gconv_box_execute_2D(gconv *g);
static void gconv_box_free(gconv *g);

static int gconv_box_plan(gconv *g)
{
    g->execute = gconv_box_execute;
    g->execute_2D = gconv_box_execute_2D;
    g->free = gconv_box_free;
    return ((g->K > 0) && (g->buffer = malloc(sizeof(num)*g->N))) ? 1 : 0;
}

static void gconv_box_execute_2D(gconv *g)
{
    int x;
    int y;
    int width = g->N;
    int height= g->N;

    num *dest_y = g->dest;
    const num *src_y = g->src;

    /* Filter each column of the channel. */
#pragma omp parallel for
    for (y = 0; y < height; ++y)
    {
        box_gaussian_conv(dest_y, g->buffer, src_y,
                width, 1, g->sigma, g->K);
        dest_y += width;
        src_y += width;
    }

    /* Filter each row of the channel. */
#pragma omp parallel for
    for (x = 0; x < width; ++x)
        box_gaussian_conv(g->dest + x, g->buffer, g->dest + x,
                height, width, g->sigma, g->K);
}

static void gconv_box_execute(gconv *g)
{
    box_gaussian_conv(g->dest, g->buffer, g->src, g->N, g->stride,
        g->sigma, g->K);
}

static void gconv_box_free(gconv *g)
{
    if (g->buffer)
        free(g->buffer);
}

/* Extended box filter. */
static void gconv_ebox_execute(gconv *g);
static void gconv_ebox_free(gconv *g);

static int gconv_ebox_plan(gconv *g)
{
    g->execute = gconv_ebox_execute;
    g->free = gconv_ebox_free;

    if ((g->K <= 0) || !(g->buffer = malloc(sizeof(num) * g->N))
        || !(g->coeffs = malloc(sizeof(ebox_coeffs))))
        return 0;

    ebox_precomp((ebox_coeffs *)g->coeffs, g->sigma, g->K);
    return 1;
}

static void gconv_ebox_execute(gconv *g)
{
    ebox_gaussian_conv(*((ebox_coeffs *)g->coeffs),
        g->dest, g->buffer, g->src, g->N, g->stride);
}

static void gconv_ebox_free(gconv *g)
{
    if (g->buffer)
        free(g->buffer);
    if (g->coeffs)
        free(g->coeffs);
}

/* SII Stacked integral images. */
static void gconv_sii_execute(gconv *g);
static void gconv_sii_free(gconv *g);

static int gconv_sii_plan(gconv *g)
{
    g->execute = gconv_sii_execute;
    g->free = gconv_sii_free;

    if (!SII_VALID_K(g->K) || !(g->coeffs = malloc(sizeof(sii_coeffs))))
        return 0;

    sii_precomp((sii_coeffs *)g->coeffs, g->sigma, g->K);

    return (g->buffer = malloc(sizeof(num) * sii_buffer_size(
        *((sii_coeffs *)g->coeffs), g->N))) ? 1 : 0;
}

static void gconv_sii_execute(gconv *g)
{
    sii_gaussian_conv(*((sii_coeffs *)g->coeffs),
        g->dest, g->buffer, g->src, g->N, g->stride);
}

static void gconv_sii_free(gconv *g)
{
    if (g->buffer)
        free(g->buffer);
    if (g->coeffs)
        free(g->coeffs);
}

/* Alvarez-Mazorra, original algorithm. */
static void gconv_am_orig_execute(gconv *g);

static int gconv_am_orig_plan(gconv *g)
{
    g->execute = gconv_am_orig_execute;
    g->free = NULL;
    return 1;
}

static void gconv_am_orig_execute(gconv *g)
{
    am_gaussian_conv(g->dest, g->src, g->N, 1,
        g->sigma, g->K, g->tol, 0);
}

/* Alvarez-Mazorra with proposed regression on parameter q. */
static void gconv_am_execute(gconv *g);

static int gconv_am_plan(gconv *g)
{
    g->execute = gconv_am_execute;
    g->free = NULL;
    return (g->K > 0);
}

static void gconv_am_execute(gconv *g)
{
    am_gaussian_conv(g->dest, g->src, g->N, 1,
        g->sigma, g->K, g->tol, 1);
}

/* Deriche recursive filtering. */
static void gconv_deriche_execute(gconv *g);
static void gconv_deriche_free(gconv *g);

static int gconv_deriche_plan(gconv *g)
{
    g->execute = gconv_deriche_execute;
    g->free = gconv_deriche_free;

    if (!DERICHE_VALID_K(g->K)
        || !(g->buffer = malloc(sizeof(num) * 2 * g->N))
        || !(g->coeffs = malloc(sizeof(deriche_coeffs))))
        return 0;

    deriche_precomp((deriche_coeffs *)g->coeffs, g->sigma, g->K, g->tol);
    return 1;
}

static void gconv_deriche_execute(gconv *g)
{
    deriche_gaussian_conv(*((deriche_coeffs *)g->coeffs),
        g->dest, g->buffer, g->src, g->N, g->stride);
}

static void gconv_deriche_free(gconv *g)
{
    if (g->buffer)
        free(g->buffer);
    if (g->coeffs)
        free(g->coeffs);
}

/* Vliet-Young-Verbeek recursive filtering. */
static void gconv_vyv_execute(gconv *g);
static void gconv_vyv_execute_2D(gconv *g);
static void gconv_vyv_free(gconv *g);

static int gconv_vyv_plan(gconv *g)
{
    g->execute = gconv_vyv_execute;
    g->execute_2D = gconv_vyv_execute_2D;
    g->free = gconv_vyv_free;

    if (!VYV_VALID_K(g->K) || !(g->coeffs = malloc(sizeof(vyv_coeffs))))
        return 0;

    vyv_precomp((vyv_coeffs *)g->coeffs, g->sigma, g->K, g->tol);
    return 1;
}

static void gconv_vyv_execute_2D(gconv *g)
{
    int x;
    int y;
    int width = g->N;
    int height= g->N;

    num *dest_y = g->dest;
    const num *src_y = g->src;

    /* Filter each row of the channel. */
#pragma omp parallel for
    for (y = 0; y < height; ++y)
    {
        vyv_gaussian_conv(*((vyv_coeffs *)g->coeffs), dest_y, src_y, width, 1);
        dest_y += width;
        src_y += width;
    }

    /* Filter each column of the channel. */
#pragma omp parallel for
    for (x = 0; x < width; ++x)
        vyv_gaussian_conv(*((vyv_coeffs *)g->coeffs), g->dest + x, g->dest + x, height, width);
}

static void gconv_vyv_execute(gconv *g)
{
    vyv_gaussian_conv(*((vyv_coeffs *)g->coeffs),
        g->dest, g->src, g->N, g->stride);
}

static void gconv_vyv_free(gconv *g)
{
    if (g->coeffs)
        free(g->coeffs);
}
#endif
