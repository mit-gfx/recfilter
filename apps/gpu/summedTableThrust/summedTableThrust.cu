#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>
#include <iomanip>

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


// This example computes a summed area table using segmented scan
// http://en.wikipedia.org/wiki/Summed_area_table



// convert a linear index to a linear index in the transpose
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
  size_t n;

  __host__ __device__
  row_index(size_t _n) : n(_n) {}

  __host__ __device__
  size_t operator()(size_t i)
  {
      return i / n;
  }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
  thrust::counting_iterator<size_t> indices(0);

  thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
     thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
     src.begin(),
     dst.begin());
}


// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);

  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(n)),
     thrust::make_transform_iterator(indices, row_index(n)) + d_data.size(),
     d_data.begin(),
     d_data.begin());
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::host_vector<T> h_data = d_data;

  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    std::cout << "\n";
  }
}

int main(int argc, char** argv)
{
  size_t num_runs = 100;

  std::cerr << "Width\tSummed_table_Thrust" << std::endl;

  for (int width=64; width<=4096; width+=32)
  {
    size_t m = width;
    size_t n = width;

    unsigned long time_start = millisecond_timer();

    for (int j=0; j<num_runs; j++)
    {
      // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
      thrust::device_vector<float> data(m * n, 1.0f);

      //std::cout << "[step 0] initial array" << std::endl;
      //print(m, n, data);

      //std::cout << "[step 1] scan horizontally" << std::endl;
      scan_horizontally(m, n, data);
      //print(m, n, data);

      //std::cout << "[step 2] transpose array" << std::endl;
      thrust::device_vector<float> temp(m * n);
      transpose(m, n, data, temp);
      //print(n, m, temp);

      //std::cout << "[step 3] scan transpose horizontally" << std::endl;
      scan_horizontally(n, m, temp);
      //print(n, m, temp);

      //std::cout << "[step 4] transpose the transpose" << std::endl;
      transpose(n, m, temp, data);
      //print(m, n, data);
    }

    unsigned long time_end = millisecond_timer();

    double millisec = double(time_end-time_start)/double(num_runs);

    float throughput = (width*width*1000.0f)/(millisec*1024*1024);
    std::cerr << width << throughput << std::endl;

    // std::cerr << width << "\t" << time << std::endl;
  }
  return 0;
}
