### Benchmarking tests

These tests run the application multiple times on random data to test running times. The output is checked for correctness.

| Application                   | Description                                           |
|:------------------------------|-------------------------------------------------------|
| ``summed_table/``             | summed area table                                     |
| ``box/``                      | iterated box filters                                  |
| ``gaussian/``                 | Vliet-Young-Verbeek approxmiation of Gaussian blur    |
| ``bspline/``                  | bicubic and biquintic b-spline filters                |
| ``usm/``                      | unsharp mask using Vliet-Young-Verbeek Gaussian blur  |
| ``audio_filter/``             | high order 1D IIR filters used for audio processing   |
