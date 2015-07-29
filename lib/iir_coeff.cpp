#include "iir_coeff.h"

using namespace Halide;

using std::vector;

/** Compute the factorial of an integer */
static inline int factorial(int k) {
    assert(k>=0);
    int r = 1;
    for (int i=1; i<=k; i++) {
        r *= i;
    }
    return r;
}

/** Compute the i-th binomial coeff of the expansion of (1-r*x)^n */
static inline float binomial_coeff(int n, int i, float r) {
    int n_choose_i = factorial(n)/(factorial(i)*factorial(n-i));
    return (pow(-r,i)*float(n_choose_i));
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
static float qs(const float& s) {
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
static std::complex<double> ds(const std::complex<double>& d, const float& s) {
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
 *  @tparam float Sigma value type
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static float ds(const float& d, const float& s ) {
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
static void weights1(const float& s, float& b0, float& a1) {
    const float d3 = 1.86543;
    float d = ds(d3, s);
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
static void weights2(const float& s, float& b0, float& a1, float& a2) {
    const std::complex<double> d1(1.41650, 1.00829);
    std::complex<double> d = ds(d1, s);
    float n2 = std::abs(d);
    n2 *= n2;
    float re = std::real(d);
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
static void weights3(const float& s, float& b0, float& a1, float& a2, float& a3) {
    float b10, b20;
    float a11, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    a1 = a11+a21;
    a2 = a11*a21 + a22;
    a3 = a11*a22;
    b0 = b10*b20;
}


vector<float> gaussian_weights(float sigma, int order) {
    float b0 = 0.0;
    vector<float> a(order+1, 0.0);

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

Expr gaussian(Expr x, float mu, float sigma) {
    Expr xx = Internal::Cast::make(type_of<float>(),x);
    Expr y = (xx - mu) / sigma;
    return Halide::fast_exp(-0.5f*y*y) / (sigma * 2.50662827463f);
}
Expr gaussDerivative(Expr x, float mu, float sigma) {
    Expr xx = Internal::Cast::make(type_of<float>(),x);
    Expr y = (xx - mu) / sigma;
    return (mu - xx) * Halide::fast_exp(-0.5f*y*y) / (sigma*sigma*sigma * 2.50662827463f);
}
Expr gaussIntegral(Expr x, float mu, float sigma) {
    Expr xx = Internal::Cast::make(type_of<float>(),x);
    return 0.5f * ( 1.0f + Halide::erf((xx-mu) / (sigma * 1.41421356237f)) );
}
float gaussian(float x, float mu, float sigma) {
    float y = (x - mu) / sigma;
    return (std::exp(-0.5*y*y) / (sigma * 2.50662827463));
}
float gaussDerivative(float x, float mu, float sigma) {
    float y = (x - mu) / sigma;
    return ((mu - x) * std::exp(-0.5*y*y) / (sigma*sigma*sigma * 2.50662827463));
}
float gaussIntegral(float x, float mu, float sigma) {
    return (0.5 * ( 1.0 + erf((x-mu) / (sigma * 1.41421356237)) ));
}

int gaussian_box_filter(int k, float sigma) {
    float sum = 0.0;
    float alpha = 0.005;
    int sum_limit = int(std::floor((float(k)-1.0)/2.0));
    for (int i=0; i<=sum_limit; i++) {
        int f_k   = factorial(k);
        int f_i   = factorial(i);
        int f_k_i = factorial(k-i);
        int f_k_1 = factorial(k-1);
        float f   = float(f_k / (f_i*f_k_i));
        float p   = std::pow(-1.0,i)/float(f_k_1);
        sum      += p * f * std::pow((float(k)/2.0-i), k-1);
    }
    sum = std::sqrt(2.0*M_PI) * (sum+alpha) * sigma;
    return int(std::ceil(sum));
}

vector<float> integral_image_coeff(int n) {
    vector<float> coeff(n+1, 0.0f);

    // set feedforward coeff = 1.0f
    coeff[0] = 1.0f;

    // feedback coeff are binomial expansion of (1-x)^n multiplied by -1
    for (int i=1; i<=n; i++) {
        coeff[i] = -1.0f * binomial_coeff(n,i,1.0f);
    }

    return coeff;
}

vector<float> overlap_feedback_coeff(vector<float> a, vector<float> b) {
    for (int i=0; i<a.size(); i++) {
        a[i] = -a[i];
    }
    for (int i=0; i<b.size(); i++) {
        b[i] = -b[i];
    }

    a.insert(a.begin(), 1.0f);
    b.insert(b.begin(), 1.0f);

    vector<float> c(a.size()+b.size()-1, 0.0f);

    for (int i=0; i<c.size(); i++) {
        for (int j=0; j<=i; j++) {
            if (j<a.size() && i-j<b.size()) {
                c[i] += a[j]*b[i-j];
            }
        }
    }

    c.erase(c.begin());
    for (int i=0; i<c.size(); i++) {
        c[i] = -c[i];
    }

    return c;
}
