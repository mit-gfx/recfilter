#ifndef _GAUSSIAN_WEIGHTS_H_
#define _GAUSSIAN_WEIGHTS_H_

#include <cmath>
#include <complex>
#include <Halide.h>


/**
 * @brief Compute the Gaussian, derivative of Gaussian and integral of Gaussian
 * @param[in] x Input (float or Halide::Expr)
 * @param[in] mu Mean of the Gaussian function
 * @param[in] sigma Standard deviation of the Gaussian
 */
// @ {
//Expr gaussian(Expr x, float mu, float sigma) {
//    Expr xx = Internal::Cast::make(type_of<float>(),x);
//    Expr y = (xx - mu) / sigma;
//    return exp(-.5*y*y) / (sigma * 2.50662827463);
//}
//Expr gaussDerivative(Expr x, float mu, float sigma) {
//    Expr xx = Internal::Cast::make(type_of<float>(),x);
//    Expr y = (xx - mu) / sigma;
//    return (mu - xx) * exp(-.5*y*y) / (sigma*sigma*sigma * 2.50662827463);
//}
//Expr gaussIntegral(float x, float mu, float sigma) {
//    Expr xx = Internal::Cast::make(type_of<float>(),x);
//    return .5 * ( 1 + erf((xx-mu) / (sigma * 1.41421356237)) );
//}
float gaussian(float x, float mu, float sigma) {
    float y = (x - mu) / sigma;
    return (std::exp(-0.5f*y*y) / (sigma * 2.50662827463f));
}
float gaussDerivative(float x, float mu, float sigma) {
    float y = (x - mu) / sigma;
    return ((mu - x) * std::exp(-0.5f*y*y) / (sigma*sigma*sigma * 2.50662827463f));
}
float gaussIntegral(float x, float mu, float sigma) {
    return (0.5f * ( 1.0f + erf((x-mu) / (sigma * 1.41421356237f)) ));
}
// @}



/**
 *  @brief Compute recursive filtering scaling factor
 *
 *  Compute the scaling factor of the recursive filter representing a
 *  true Gaussian filter convolution with arbitrary support sigma.
 *
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Scaling factor q of the recursive filter approximation
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
float qs(const float& s) {
    return 0.00399341f + 0.4715161f * s;
}

/**
 *  @brief Rescale poles of the recursive filtering z-transform
 *
 *  Given a complex-valued pole on |z|=1 ROC of the recursive filter
 *  z-transform compute a rescaled pole representing a true Gaussian
 *  filter convolution with arbitrary support sigma.
 *
 *  @param[in] d Complex-valued pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled complex-valued pole of the recursive filter approximation
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
std::complex<float> ds(const std::complex<float>& d, const float& s) {
    float q = qs(s);
    return std::polar(std::pow(std::abs(d), 1.0f/q), std::arg(d)/q);
}

/**
 *  @brief Rescale poles in the real-axis of the recursive filtering z-transform
 *
 *  Given a real pole on |z|=1 ROC of the recursive filter z-transform
 *  compute a rescaled pole representing a true Gaussian filter
 *  convolution with arbitrary support sigma.
 *
 *  @param[in] d Real pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled real pole of the recursive filter approximation
 *  @tparam float Sigma value type
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
float ds(const float& d, const float& s ) {
    return std::pow(d, 1.0f/qs(s));
}

/**
 *  @brief Compute first-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first-order coefficients.
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
void weights1(const float& s, float& b0, float& a1) {
    const float d3 = 1.86543f;
    float d = ds(d3, s);
    b0 = -(1.0f-d)/d;
    a1 = -1.0f/d;
}

/**
 *  @brief Compute first and second-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first- and second-order coefficients.
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @param[out] a2 Feedback second-order coefficient
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
void weights2(const float& s, float& b0, float& a1, float& a2) {
    const std::complex<float> d1(1.41650f, 1.00829f);
    std::complex<float> d = ds(d1, s);
    float n2 = std::abs(d);
    n2 *= n2;
    float re = std::real(d);
    b0 = (1.0f-2.0f*re+n2)/n2;
    a1 = -2.0f*re/n2;
    a2 = 1.0f/n2;
}

#endif // _GAUSSIAN_WEIGHTS_H_
