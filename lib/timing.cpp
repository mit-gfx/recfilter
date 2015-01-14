#include "timing.h"

float throughput(float runtime, int pixels) {
    return (pixels*1000.0f)/(runtime*1024*1024);
}

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
