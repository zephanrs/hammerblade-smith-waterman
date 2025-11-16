#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_barrier_multipod.h"
#include <cstdint>

// flags
#define SEQUENCES  1
#define SEQ_LENGTH 32

#define QRY_CORE (SEQ_LENGTH / bsg_tiles_Y)
#define REF_CORE (SEQ_LENGTH / bsg_tiles_X)

// parameters
#define NUM_TILES       (bsg_tiles_X*bsg_tiles_Y)
#define MATCH     1
#define MISMATCH -1
#define GAP       1

inline int max(int a, int b) {
  if (a > b) {
    return a;
  }
  return b;
}

inline int max(int a, int b, int c) {
  return max(a, max(b,c));
}

inline int max(int a, int b, int c, int d) {
  return max(max(a,b), max(c,d));
}

int dp[QRY_CORE+1][REF_CORE+1];
uint8_t qrybuf[QRY_CORE];
uint8_t refbuf[REF_CORE];
int max_left = -1;
int max_top  = -1;
int maxv     = -1;

// Kernel main;
extern "C" int kernel(uint8_t* qry, uint8_t* ref, int* output, int pod_id)
{
  bsg_barrier_tile_group_init();
  bsg_barrier_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  for (int i = 0; i <= REF_CORE; i++)
    dp[0][i] = 0;

  for (int j = 0; j <= QRY_CORE; j++)
    dp[j][0] = 0;

  int tmp;

  if (!__bsg_y) {
    // load reference
    for (int i = 0; i < REF_CORE; i++) 
      refbuf[i] = ref[(__bsg_x * REF_CORE) + i ];
  }

  if (!__bsg_x) {
    // load query
    for (int i = 0; i < QRY_CORE; i++) 
      qrybuf[i] = qry[(__bsg_y * QRY_CORE) + i];
  }
  
  if (__bsg_y) {
    // wait for core above to write
    tmp = bsg_lr(&max_top);
    if (__bsg_y && (tmp == -1)) bsg_lr_aq(&max_top);
    asm volatile("" ::: "memory");
  }

  if (__bsg_x) {
    // wait for core to the left to write
    tmp = bsg_lr(&max_left);
    if (__bsg_x && (tmp == -1)) bsg_lr_aq(&max_left);
    asm volatile("" ::: "memory");
  }
  
  // derive maximum value
  maxv = max(0, max_top, max_left);
  
  // do dp calculation
  for (int i = 1; i <= QRY_CORE; i++) {
    for (int j = 1; j <= REF_CORE; j++) {
      int match = (qrybuf[i-1] == refbuf[j-1]) ? MATCH : MISMATCH;

      int score_diag = dp[i-1][j-1] + match;
      int score_up   = dp[i-1][j]   - GAP;
      int score_left = dp[i]  [j-1] - GAP;

      dp[i][j] = max(0, score_diag, score_up, score_left);

      if (dp[i][j] > maxv) {
        maxv = dp[i][j];
      }
    }
  }

  int *ndp, *nmax;
  uint8_t *nseq;

  
  if (__bsg_y < (bsg_tiles_Y - 1)) {
    // copy dp below
    ndp =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &dp[0][0]);
    for (int j = 0; j <= REF_CORE; j++)
      ndp[j] = dp[QRY_CORE][j];

    // copy reference below
    nseq = (uint8_t*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &refbuf);
    for (int j = 0; j < REF_CORE; j++)
      nseq[j] = refbuf[j];

    // activate bottom core (and transfer max)
    nmax =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &max_top);
    *nmax = maxv;
  }

  if (__bsg_x < (bsg_tiles_X - 1)) {
    // copy dp right
    ndp = (int*) bsg_remote_ptr(__bsg_x+1, __bsg_y, &dp[0][0]);
    for (int j = 0; j <= QRY_CORE; j++)
      ndp[(REF_CORE+1)*j] = dp[j][REF_CORE];

    // copy query below
    nseq = (uint8_t*) bsg_remote_ptr(__bsg_x+1, __bsg_y, &qrybuf);
    for (int j = 0; j < QRY_CORE; j++)
      nseq[j] = qrybuf[j];

    // activate right core (and transfer max)
    nmax =  (int*) bsg_remote_ptr(__bsg_x+1, __bsg_y, &max_left);
    *nmax = maxv;
  }

  if ((__bsg_x == (bsg_tiles_X - 1)) && (__bsg_y == (bsg_tiles_Y - 1)))
    *output = maxv;
  
  // kernel end;
  bsg_fence();
  bsg_barrier_tile_group_sync();
  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  return 0;
}
