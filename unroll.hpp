#include <stdint.h>

template<int N>
static inline __attribute__((always_inline))
void unrolled_load(uint8_t* __restrict dst,
                   const uint8_t* __restrict src)
{
  // N >= 16 (hide latency)
  if (N >= 16) {
    for (int i = 0; i < (N / 16); i++) {
      register uint8_t r0  = src[0];
      register uint8_t r1  = src[1];
      register uint8_t r2  = src[2];
      register uint8_t r3  = src[3];
      register uint8_t r4  = src[4];
      register uint8_t r5  = src[5];
      register uint8_t r6  = src[6];
      register uint8_t r7  = src[7];
      register uint8_t r8  = src[8];
      register uint8_t r9  = src[9];
      register uint8_t r10 = src[10];
      register uint8_t r11 = src[11];
      register uint8_t r12 = src[12];
      register uint8_t r13 = src[13];
      register uint8_t r14 = src[14];
      register uint8_t r15 = src[15];

      asm volatile("" ::: "memory");

      dst[0]  = r0;  dst[1]  = r1;  dst[2]  = r2;  dst[3]  = r3;
      dst[4]  = r4;  dst[5]  = r5;  dst[6]  = r6;  dst[7]  = r7;
      dst[8]  = r8;  dst[9]  = r9;  dst[10] = r10; dst[11] = r11;
      dst[12] = r12; dst[13] = r13; dst[14] = r14; dst[15] = r15;

      src += 16;
      dst += 16;
    }
  } // N == 8 
  else if (N == 8) {
    register uint8_t r0 = src[0];
    register uint8_t r1 = src[1];
    register uint8_t r2 = src[2];
    register uint8_t r3 = src[3];
    register uint8_t r4 = src[4];
    register uint8_t r5 = src[5];
    register uint8_t r6 = src[6];
    register uint8_t r7 = src[7];

    asm volatile("" ::: "memory");

    dst[0] = r0; dst[1] = r1; dst[2] = r2; dst[3] = r3;
    dst[4] = r4; dst[5] = r5; dst[6] = r6; dst[7] = r7;
  } // N == 4 
  else if (N == 4) {
    register uint8_t r0 = src[0];
    register uint8_t r1 = src[1];
    register uint8_t r2 = src[2];
    register uint8_t r3 = src[3];

    asm volatile("" ::: "memory");

    dst[0] = r0; dst[1] = r1; dst[2] = r2; dst[3] = r3;
  } // N == 2 
  else if (N == 2) {
    register uint8_t r0 = src[0];
    register uint8_t r1 = src[1];

    asm volatile("" ::: "memory");

    dst[0] = r0; dst[1] = r1;
  } // N == 1
  else { 
    register uint8_t r0 = src[0];

    asm volatile("" ::: "memory");

    dst[0] = r0;
  }
}