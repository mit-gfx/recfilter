#ifndef _TIMER_H_
#define _TIMER_H_

/* Autodetect whether to use Windows, POSIX,
   or fallback implementation for Clock.  */
#if !defined(USE_GETSYSTEMTIME) && !defined(USE_GETTIMEOFDAY) && !defined(USE_TIME)
#   if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#       define USE_GETSYSTEMTIME
#   elif defined(unix) || defined(__unix__) || defined(__unix)
#       include <unistd.h>
#       if (_POSIX_TIMERS) || (_POSIX_VERSION >= 200112L)
#           define USE_GETTIMEOFDAY
#       endif
#   endif
#endif

/* Define millisecond_timer(), get the system clock in milliseconds */
#if defined(USE_GETSYSTEMTIME)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

unsigned long millisecond_timer()   /* Windows implementation */
{
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

unsigned long millisecond_timer()   /* POSIX implementation */
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}
#else
#include <time.h>

unsigned long millisecond_timer()   /* Fallback implementation */
{
    time_t raw_time;
    struct tm *t;
    time(&raw_time);
    t = localtime(&raw_time);
    return (unsigned long)(1000*((unsigned long)t->tm_sec
        + 60*((unsigned long)t->tm_min
        + 60*((unsigned long)t->tm_hour
        + 24*(unsigned long)t->tm_mday))));
}
#endif

#endif // _TIMER_H_
