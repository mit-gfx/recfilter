#ifndef _TIMING_H_
#define _TIMING_H_

#include <fstream>
#include <iostream>
#include <cstdlib>
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
                exit(EXIT_FAILURE);
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
float throughput(float runtime, int pixels);

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
unsigned long millisecond_timer(void);

#endif // _TIMING_H_
