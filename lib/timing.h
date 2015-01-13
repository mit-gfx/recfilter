#ifndef _TIMING_H_
#define _TIMING_H_

#include <fstream>
#include <iostream>
#include <string>

/** Logging utility */
class Log {
private:
    std::fstream out;
public:
    Log(std::string filename) {
        if (!filename.empty()) {
            out.open(filename, std::ios_base::out);
            if (!out.is_open()) {
                std::cerr << "Could not open " << filename << " for logging" << std::endl;
                assert(false);
            }
        }
    }

    template<typename T>
    std::fstream& operator<<(T x) {
        if (out.is_open()) {
            out << x;
        }
        return out;
    }
};

/** Compute the throughput in Gibipixels = 2^30 pixels
 * \param runtime running time in milliseconds
 * \param pixels number of pixels
 * \returns throughput in GiP/s
 */
float throughput(float runtime, int pixels) {
    int gibipixels = 2^30;
    return float(pixels) / float(gibipixels*runtime*1000.0f);
}

/**
 * Millisecond-precision timer function
 * \return Clock value in milliseconds
 *
 * This routine implements a timer with millisecond precision.  In order to
 * obtain timing at high resolution, platform-specific functions are needed:
 *
 *    - On Windows systems, the GetSystemTime function is used.
 *    - On Mac and POSIX systems, the gettimeofday function is used.
 *
 * Preprocessor symbols are checked in attempt to detect whether the platform
 * is POSIX or Windows or Mac and defines millisecond_timer() accordingly.
 */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
unsigned long millisecond_timer(void) {
    static SYSTEMTIME t;
    GetSystemTime(&t);
    return (unsigned long)((unsigned long)t.wMilliseconds
            + 1000*((unsigned long)t.wSecond
                + 60*((unsigned long)t.wMinute
                    + 60*((unsigned long)t.wHour
                        + 24*(unsigned long)t.wDay))));
}
#elif defined(_APPLE_) || defined(__APPLE__) || \
    defined(APPLE)   || defined(_APPLE)    || defined(__APPLE) || \
defined(unix)    || defined(__unix__)  || defined(__unix)
#include <unistd.h>
#include <sys/time.h>
unsigned long millisecond_timer(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}
#else
unsigned long millisecond_timer(void) {
    std::cerr << "Warning: no timer implementation available" << std::endl;
    return 0;
}
#endif


#endif // _TIMING_H_
