#include <stdint.h>

template<typename T, int N, int S = 1>
static inline __attribute__((always_inline))
void unrolled_load(T* __restrict dst,
                   const T* __restrict src)
{
    T buf[N];

    // load
    bsg_unroll(24)
    for (int i = 0; i < N; i++) {
        register T r = src[i * S];
        buf[i] = r;
    }

    asm volatile("" ::: "memory");

    // store
    bsg_unroll(24)
    for (int i = 0; i < N; i++) {
        dst[i * S] = buf[i];
    }
}
