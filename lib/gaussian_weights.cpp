#include "gaussian_weights.h"

using namespace Halide;

using std::vector;

/// Compute the factorial of an integer
static int factorial(int k) {
    assert(k>=0);
    int r = 1;
    for (int i=1; i<=k; i++) {
        r *= i;
    }
    return r;
}

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
static double qs(const double& s) {
    return 0.00399341 + 0.4715161 * s;
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
static std::complex<double> ds(const std::complex<double>& d, const double& s) {
    double q = qs(s);
    return std::polar(std::pow(std::abs(d), 1.0/q), std::arg(d)/q);
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
 *  @tparam double Sigma value type
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static double ds(const double& d, const double& s ) {
    return std::pow(d, 1.0/qs(s));
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
static void weights1(const double& s, double& b0, double& a1) {
    const double d3 = 1.86543;
    double d = ds(d3, s);
    b0 = -(1.0-d)/d;
    a1 = -1.0/d;
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
static void weights2(const double& s, double& b0, double& a1, double& a2) {
    const std::complex<double> d1(1.41650, 1.00829);
    std::complex<double> d = ds(d1, s);
    double n2 = std::abs(d);
    n2 *= n2;
    double re = std::real(d);
    b0 = (1.0-2.0*re+n2)/n2;
    a1 = -2.0*re/n2;
    a2 = 1.0/n2;
}

/**
 *  @brief Compute third order recursive filter weights for approximating Gaussian
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @param[out] a2 Feedback second-order coefficient
 *  @param[out] a3 Feedback third-order coefficient
 *
 *  Coefficients equivalent to applying a first order recursive filter followed
 *  by second order filter
 */
static void weights3(const double& s, double& b0, double& a1, double& a2, double& a3) {
    double b10, b20;
    double a11, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    a1 = a11+a21;
    a2 = a11*a21 + a22;
    a3 = a11*a22;
    b0 = b10*b20;
}


vector<double> gaussian_weights(double sigma, int order) {
    double b0 = 0.0;
    vector<double> a(order+1, 0.0);

    switch(order) {
        case 1: weights1(sigma, a[0], a[1]); break;
        case 2: weights2(sigma, a[0], a[1], a[2]); break;
        default:weights3(sigma, a[0], a[1], a[2], a[3]); break;
    }

    for (int i=1; i<a.size(); i++) {
        a[i] = -a[i];
    }

    return a;
}

Expr gaussian(Expr x, double mu, double sigma) {
    Expr xx = Internal::Cast::make(type_of<double>(),x);
    Expr y = (xx - mu) / sigma;
    return Halide::fast_exp(-0.5*y*y) / (sigma * 2.50662827463);
}
Expr gaussDerivative(Expr x, double mu, double sigma) {
    Expr xx = Internal::Cast::make(type_of<double>(),x);
    Expr y = (xx - mu) / sigma;
    return (mu - xx) * Halide::fast_exp(-0.5*y*y) / (sigma*sigma*sigma * 2.50662827463);
}
Expr gaussIntegral(Expr x, double mu, double sigma) {
    Expr xx = Internal::Cast::make(type_of<double>(),x);
    return 0.5 * ( 1.0 + Halide::erf((xx-mu) / (sigma * 1.41421356237)) );
}
double gaussian(double x, double mu, double sigma) {
    double y = (x - mu) / sigma;
    return (std::exp(-0.5*y*y) / (sigma * 2.50662827463));
}
double gaussDerivative(double x, double mu, double sigma) {
    double y = (x - mu) / sigma;
    return ((mu - x) * std::exp(-0.5*y*y) / (sigma*sigma*sigma * 2.50662827463));
}
double gaussIntegral(double x, double mu, double sigma) {
    return (0.5 * ( 1.0 + erf((x-mu) / (sigma * 1.41421356237)) ));
}

int gaussian_box_filter(int k, double sigma) {
    double sum = 0.0;
    double alpha = 0.005;
    int sum_limit = int(std::floor((double(k)-1.0)/2.0));
    for (int i=0; i<=sum_limit; i++) {
        int f_k   = factorial(k);
        int f_i   = factorial(i);
        int f_k_i = factorial(k-i);
        int f_k_1 = factorial(k-1);
        double f   = double(f_k / (f_i*f_k_i));
        double p   = std::pow(-1.0,i)/double(f_k_1);
        sum      += p * f * std::pow((double(k)/2.0-i), k-1);
    }
    sum = std::sqrt(2.0*M_PI) * (sum+alpha) * sigma;
    return int(std::ceil(sum));
}

Image<double> reference_gaussian(Image<double> in, double sigma) {
    int width = in.width();
    int height= in.height();
    Image<double> ref(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            double a = 0.0;
            double w = 0.0;
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    double d = (x-i)*(x-i) + (y-j)*(y-j);
                    double g = gaussian(std::sqrt(d), 0.0, sigma);
                    a += g * in(i,j);
                    w += g;
                }
            }
            ref(x,y) = a/w;
        }
    }
    return ref;
}
