#ifndef _TIMER_H_
#define _TIMER_H_

/**
 * \brief Millisecond-precision timer function
 * \return Clock value in units of milliseconds
 *
 * This routine implements a timer with millisecond precision.  In order to
 * obtain timing at high resolution, platform-specific functions are needed:
 *
 *    - On Windows systems, the GetSystemTime function is used.
 *    - On Mac and POSIX systems, the gettimeofday function is used.
 *
 * Otherwise as a fallback, time.h time is used, and in this case
 * millisecond_timer() has only second accuracy.  Preprocessor symbols are
 * checked in attempt to detect whether the platform is POSIX or Windows and
 * defines millisecond_timer() accordingly.  A particular implementation can
 * be forced by defining USE_GETSYSTEMTIME, USE_GETTIMEOFDAY, or USE_TIME.
 */
unsigned long millisecond_timer(void);

/* Autodetect whether to use Windows, Mac, or POSIX */
#if !defined(USE_GETSYSTEMTIME) && !defined(USE_GETTIMEOFDAY) && !defined(USE_TIME)
#   if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#       define USE_GETSYSTEMTIME
#   elif defined(_APPLE_) || defined(__APPLE__) || defined(APPLE) || defined(_APPLE) || defined(__APPLE)
#       include <unistd.h>
#       if (_POSIX_TIMERS) || (_POSIX_VERSION >= 200112L)
#           define USE_GETTIMEOFDAY
#       endif
#   elif defined(unix) || defined(__unix__) || defined(__unix)
#       include <unistd.h>
#       if (_POSIX_TIMERS) || (_POSIX_VERSION >= 200112L)
#           define USE_GETTIMEOFDAY
#       endif
#   endif
#endif

#if defined(USE_GETSYSTEMTIME)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
unsigned long millisecond_timer(void) { /* Windows implementation */
    static SYSTEMTIME t;
    GetSystemTime(&t);
    return (unsigned long)((unsigned long)t.wMilliseconds
        + 1000*((unsigned long)t.wSecond
        + 60*((unsigned long)t.wMinute
        + 60*((unsigned long)t.wHour
        + 24*(unsigned long)t.wDay))));
}
#elif defined(USE_GETTIMEOFDAY)
#include <unistd.h>
#include <sys/time.h>
unsigned long millisecond_timer(void) { /* POSIX implementation */
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}
#else
unsigned long millisecond_timer(void) { /* No millisecond timer available */
    return 0;
//    time_t raw_time;
//    struct tm *t;
//    time(&raw_time);
//    t = localtime(&raw_time);
//    return (unsigned long)(1000*((unsigned long)t->tm_sec
//        + 60*((unsigned long)t->tm_min
//        + 60*((unsigned long)t->tm_hour
//        + 24*(unsigned long)t->tm_mday))));
}
#endif

#endif // _TIMER_H_
