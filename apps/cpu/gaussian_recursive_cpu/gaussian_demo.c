/**
 * \file gaussian_demo.c
 * \brief Gaussian convolution demo
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2012-2013, Pascal Getreuer
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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "imageio.h"
#include "gaussian_conv_fir.h"
#include "gaussian_conv_dct.h"
#include "gaussian_conv_box.h"
#include "gaussian_conv_ebox.h"
#include "gaussian_conv_sii.h"
#include "gaussian_conv_am.h"
#include "gaussian_conv_deriche.h"
#include "gaussian_conv_vyv.h"

/** \brief Print program usage help message */
void print_usage()
{
    puts("Gaussian convolution demo, P. Getreuer 2013");
#ifdef NUM_SINGLE
    puts("Single-precision computation");
#else
    puts("Double-precision computation");
#endif
    puts("\nSyntax: gaussian_conv_demo [options] <input> <output>\n");
    puts("Only " READIMAGE_FORMATS_SUPPORTED " images are supported.\n");
    puts("Options:");
    puts("   -a <algo>     algorithm to use, choices are");
    puts("                 fir     FIR approximation, tol = kernel accuracy");
    puts("                 dct     DCT-based convolution");
    puts("                 box     box filtering, K = # passes");
    puts("                 sii     stacked integral images, K = # boxes");
    puts("                 am      Alvarez-Mazorra using regression on q,");
    puts("                         K = # passes, tol = boundary accuracy");
    puts("                 deriche Deriche recursive filtering,");
    puts("                         K = order, tol = boundary accuracy");
    puts("                 vyv     Vliet-Young-Verbeek recursive filtering,");
    puts("                         K = order, tol = boundary accuracy");
    puts("   -s <number>   sigma, standard deviation of the Gaussian");
    puts("   -K <number>   specifies number of steps (box, sii, am)");
    puts("   -t <number>   accuracy tolerance (fir, am, deriche, yv)\n");
}

/** \brief struct of program parameters */
typedef struct
{
    /** \brief Input file */
    const char *input_file;
    /** \brief Output file (blurred) */
    const char *output_file;

    /** \brief Name of the convolution algorithm */
    const char *algo;
    /** \brief sigma parameter of the Gaussian */
    double sigma;
    /** \brief parameter K */
    int K;
    /** \brief Tolerance */
    double tol;
} program_params;

int parse_params(program_params *param, int argc, char **argv);
int is_grayscale(const num *rgb_image, long num_pixels);

void normalize(num *output_image, long num_pixels) {
  num min_value = output_image[0];
  num max_value = output_image[0];
  num scale;
  long i;
  for (i = 1; i < num_pixels; ++i) {
    if (output_image[i] < min_value) {
      min_value = output_image[i];
    } else if (output_image[i] > max_value) {
      max_value = output_image[i];
    }
  }
  scale = 1.0 / (max_value - min_value);
  for (i = 0; i < num_pixels; ++i) {
    output_image[i] = (output_image[i] - min_value) * scale;
  }
}

int main(int argc, char **argv)
{
    program_params param;
    num *input_image = NULL;
    num *output_image = NULL;
    unsigned long time_start;
    long num_pixels;
    int width, height, num_channels, success = 0;

    if (!parse_params(&param, argc, argv))
        return 1;

    /* Read the input image. */
    if (!(input_image = (num *)read_image(&width, &height, param.input_file,
        IMAGEIO_NUM | IMAGEIO_PLANAR | IMAGEIO_RGB)))
        goto fail;

    num_pixels = ((long)width) * ((long)height);
    num_channels = is_grayscale(input_image, num_pixels) ? 1 : 3;

    /* Allocate the output image. */
    if (!(output_image = (num *)malloc(sizeof(num)
        * num_channels * num_pixels)))
        goto fail;

    printf("Convolving %dx%d %s image with Gaussian, sigma=%g\n",
        width, height, (num_channels == 3 ? "RGB" : "gray"), param.sigma);
    time_start = millisecond_timer();

    if (!strcmp(param.algo, "fir"))
    {   /* FIR convolution. */
        num *buffer = NULL;
        fir_coeffs c;

        printf("FIR convolution, tol=%g\n", param.tol);

        if (!(buffer = (num *)malloc(sizeof(num) * width)))
            goto fail;
        if (!(fir_precomp(&c, param.sigma, param.tol)))
        {
            free(buffer);
            goto fail;
        }

        fir_gaussian_conv_image(c, output_image, buffer, input_image,
            width, height, num_channels);
        fir_free(&c);
        free(buffer);
    }
    else if (!strcmp(param.algo, "dct"))
    {   /* DCT-based convolution. */
        dct_coeffs c;

        printf("DCT-based convolution\n");

        if (!(dct_precomp_image(&c, output_image, input_image,
            width, height, num_channels, param.sigma)))
            goto fail;

        dct_gaussian_conv(c);
        dct_free(&c);
    }
    else if (!strcmp(param.algo, "box"))
    {   /* Basic box filtering. */
        num *buffer = NULL;

        printf("Box filtering, K=%d passes\n",
            param.K);

        if (!(buffer = (num *)malloc(sizeof(num) *
            ((width >= height) ? width : height))))
        {
            fprintf(stderr, "Error: Out of memory\n");
            goto fail;
        }

        box_gaussian_conv_image(output_image, buffer, input_image,
            width, height, num_channels, param.sigma, param.K);
        free(buffer);
    }
    else if (!strcmp(param.algo, "ebox"))
    {   /* Extended box filtering. */
        num *buffer = NULL;
        ebox_coeffs c;

        printf("Extended box filtering, K=%d passes\n",
            param.K);

        if (!(buffer = (num *)malloc(sizeof(num) *
            ((width >= height) ? width : height))))
        {
            fprintf(stderr, "Error: Out of memory\n");
            goto fail;
        }

        ebox_precomp(&c, param.sigma, param.K);
        ebox_gaussian_conv_image(c, output_image, buffer, input_image,
            width, height, num_channels);
        free(buffer);
    }
    else if (!strcmp(param.algo, "sii"))
    {   /* Stacked integral images. */
        num *buffer = NULL;
        sii_coeffs c;

        if (!SII_VALID_K(param.K))
        {
            fprintf(stderr, "Error: K=%d is invalid for SII\n", param.K);
            goto fail;
        }

        printf("Stacked integral images, K=%d boxes\n", param.K);
        sii_precomp(&c, param.sigma, param.K);

        if (!(buffer = (num *)malloc(sizeof(num) * sii_buffer_size(c,
            ((width >= height) ? width : height)))))
            goto fail;

        sii_gaussian_conv_image(c, output_image, buffer, input_image,
            width, height, num_channels);
        free(buffer);
    }
    else if (!strcmp(param.algo, "am"))
    {   /* Alvarez-Mazorra recursive filtering (using regression on q). */
        printf("Alvarez-Mazorra recursive filtering, K=%d passes,"
            " tol=%g left boundary accuracy\n", param.K, param.tol);
        am_gaussian_conv_image(output_image, input_image,
            width, height, num_channels,
            param.sigma, param.K, param.tol, 1);
    }
    else if (!strcmp(param.algo, "deriche"))
    {   /* Deriche recursive filtering. */
        num *buffer = NULL;
        deriche_coeffs c;

        if (!DERICHE_VALID_K(param.K))
        {
            fprintf(stderr, "Error: K=%d is invalid for Deriche\n", param.K);
            goto fail;
        }

        printf("Deriche recursive filtering,"
            " K=%d, tol=%g boundary accuracy\n", param.K, param.tol);

        if (!DERICHE_VALID_K(param.K)
            || !(buffer = (num *)malloc(sizeof(num) * 2 *
            ((width >= height) ? width : height))))
        {
            fprintf(stderr, "Error: Out of memory\n");
            goto fail;
        }

        deriche_precomp(&c, param.sigma, param.K, param.tol);
        deriche_gaussian_conv_image(c, output_image, buffer, input_image,
            width, height, num_channels);
        free(buffer);
    }
    else if (!strcmp(param.algo, "vyv"))
    {   /* Vliet-Young-Verbeek recursive filtering. */
        vyv_coeffs c;

        if (!VYV_VALID_K(param.K))
        {
            fprintf(stderr, "Error: K=%d is invalid for VYV\n", param.K);
            goto fail;
        }

        printf("Vliet-Young-Verbeek recursive filtering,"
            " K=%d, tol=%g left boundary accuracy\n", param.K, param.tol);
        vyv_precomp(&c, param.sigma, param.K, param.tol);
        vyv_gaussian_conv_image(c, output_image, input_image,
            width, height, num_channels);
    }
    else
    {
        fprintf(stderr, "Unknown method \"%s\".\n", param.algo);
        goto fail;
    }

    printf("CPU Time: %.3f s\n", 0.001f*(millisecond_timer() - time_start));

    normalize(output_image, width * height);

    /* Write output image */
    if (!write_image(output_image, width, height, param.output_file,
        IMAGEIO_NUM | IMAGEIO_PLANAR
        | ((num_channels == 1) ? IMAGEIO_GRAYSCALE : IMAGEIO_RGB), 95))
        goto fail;

    success = 1;
fail:
    if (output_image)
        free(output_image);
    if (input_image)
        free(input_image);
    return !success;
}

/**
 * \brief Test whether image is grayscale (R = G = B)
 * \param rgb_image     image in planar order
 * \param num_pixels    number of pixels
 * \return 1 if the image is grayscale, 0 otherwise
 */
int is_grayscale(const num *rgb_image, long num_pixels)
{
    const num *red   = rgb_image;
    const num *green = rgb_image + num_pixels;
    const num *blue  = rgb_image + 2 * num_pixels;
    long i;

    for (i = 0; i < num_pixels; ++i)
        if (red[i] != green[i] || red[i] != blue[i])
            return 0;    /* Not a grayscale image. */

    return 1;    /* This is a grayscale image. */
}

/** \brief Parse command line arguments */
int parse_params(program_params *param, int argc, char **argv)
{
    static const char *default_output_file = (const char *)"out.bmp";
    static const char *default_algo = (const char *)"exact";
    char *option_string;
    char option_char;
    int i;

    if (argc < 2)
    {
        print_usage();
        return 0;
    }

    /* Set parameter defaults. */
    param->input_file = NULL;
    param->output_file = default_output_file;
    param->sigma = 5;
    param->algo = default_algo;
    param->K = 3;
    param->tol = 1e-2;

    for (i = 1; i < argc;)
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

                if (param->K < 0)
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
        {
            if (!param->input_file)
                param->input_file = argv[i];
            else
                param->output_file = argv[i];

            i++;
        }
    }

    if (!param->input_file)
    {
        print_usage();
        return 0;
    }

    return 1;
}
