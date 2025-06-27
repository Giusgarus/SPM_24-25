#include <random>       
#include <cstdint>      
#include <iostream>     

#include <hpc_helpers.hpp>
#include <immintrin.h>


void init(float * data, uint64_t length) {

    std::mt19937 engine(42);
    std::uniform_real_distribution<float> density(-1L<<28, 1L<<28);

    for (uint64_t i = 0; i < length; i++)
        data[i] = density(engine);
}

float plain_max(float * data, uint64_t length) {

    float max = -INFINITY;

    //#pragma omp simd reduction(max:max)
    for (uint64_t i = 0; i < length; i++)
      max = std::max(max, data[i]);

    return max;
}

float plain_max_unroll_2(float * data, uint64_t length) {

    float max_0 = -INFINITY;
    float max_1 = -INFINITY;

    for (uint64_t i = 0; i < length; i += 2) {
        max_0 = std::max(max_0, data[i+0]);
        max_1 = std::max(max_1, data[i+1]);
    }

    return std::max(max_0, max_1);
}

float plain_max_unroll_4(float * __restrict__ data, const uint64_t length) {

    float max_0 = -INFINITY;
    float max_1 = -INFINITY;
    float max_2 = -INFINITY;
    float max_3 = -INFINITY;

    for (uint64_t i = 0; i < length; i += 4) {
        max_0 = std::max(max_0, data[i+0]);
        max_1 = std::max(max_1, data[i+1]);
        max_2 = std::max(max_2, data[i+2]);
        max_3 = std::max(max_3, data[i+3]);
    }

    return std::max(max_0,
           std::max(max_1,
           std::max(max_2, max_3)));
}

#if 1
float avx_max(float * data, uint64_t length) {

    __m256 maxVec = _mm256_set1_ps(-INFINITY);
    for (uint64_t i=0; i < length; i += 8) {
        __m256 vec = _mm256_load_ps(&data[i]);
        maxVec = _mm256_max_ps(maxVec, vec);
    }
    alignas(32)   float tmp[8];

    _mm256_store_ps(tmp, maxVec);

#if 0
    float max=tmp[0];
    for(int i=1;i<8;++i)
      max=std::max(max, tmp[i]);
    return max;
#else

    float max_0 = tmp[0];
    float max_1 = tmp[1];
    float max_2 = tmp[2];
    float max_3 = tmp[3];
    max_0=std::max(max_0, tmp[4]);
    max_1=std::max(max_1, tmp[5]);
    max_2=std::max(max_2, tmp[6]);
    max_3=std::max(max_3, tmp[7]);
    max_0=std::max(max_0, max_1);
    max_2=std::max(max_2, max_3);

    return std::max(max_0, max_2);
#endif

    
}
#endif

int main () {

    const uint64_t num_entries = 1UL << 28;
    const uint64_t num_bytes = num_entries*sizeof(float);

    auto data = new float[num_bytes];

    TIMERSTART(init);
    init(data, num_entries);
    TIMERSTOP(init);

    TIMERSTART(plain_max);
    std::cout << plain_max(data, num_entries) << std::endl;
    TIMERSTOP(plain_max);

    TIMERSTART(avx_max);
    std::cout << avx_max(data, num_entries) << std::endl;
    TIMERSTOP(avx_max);
    
    TIMERSTART(plain_max_unroll_2);
    std::cout << plain_max_unroll_2(data, num_entries) << std::endl;
    TIMERSTOP(plain_max_unroll_2);

    TIMERSTART(plain_max_unroll_4);
    std::cout << plain_max_unroll_4(data, num_entries) << std::endl;
    TIMERSTOP(plain_max_unroll_4);

		
    delete [] data;
}
