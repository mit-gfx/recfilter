#!/bin/bash

gpu/nvRecursiveGaussian 2> gaussian_filter.nvidia.perflog
gpu/nvSummedTable 2> summed_table.nvidia.perflog
gpu/nvboxFilter 1 2> box_filter_1.nvidia.perflog
gpu/nvboxFilter 3 2> box_filter_3.nvidia.perflog
gpu/nvboxFilter 6 2> box_filter_6.nvidia.perflog
gpu/nvBicubic     2> bicubic_filter.nvidia.perflog
gpu/nvBiquintic   2> biquintic_filter.nvidia.perflog
