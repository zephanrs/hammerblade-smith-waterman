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

int max_left = -1;
int max_top  = -1;
int maxv     = -1;

int dp[QRY_CORE+1][REF_CORE+1];

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

  // wait for core above to write
  tmp = bsg_lr(&max_top);
  if (__bsg_y && (tmp == -1)) bsg_lr_aq(&max_top);
  asm volatile("" ::: "memory");

  // wait for core to the left to write
  tmp = bsg_lr(&max_left);
  if (__bsg_x && (tmp == -1)) bsg_lr_aq(&max_left);
  asm volatile("" ::: "memory");

  // derive maximum value
  maxv = max(0, max_top, max_left);
  
  // do dp calculation
  for (int i = 1; i <= QRY_CORE; i++) {
    for (int j = 1; j <= REF_CORE; j++) {
      int qi = (__bsg_y * QRY_CORE) + (i - 1);
      int rj = (__bsg_x * REF_CORE) + (j - 1);

      int match = (qry[qi] == ref[rj]) ? MATCH : MISMATCH;

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

  // copy data below
  if (__bsg_y < (bsg_tiles_Y - 1)) {
    ndp =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &dp[0][0]);
    for (int j = 0; j <= REF_CORE; j++) {
      ndp[j] = dp[QRY_CORE][j];
    }

    // activate bottom core (and transfer max)
    nmax =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &max_top);
    *nmax = maxv;
  }

  // copy data right
  if (__bsg_x < (bsg_tiles_X - 1)) {
    ndp = (int*) bsg_remote_ptr(__bsg_x+1, __bsg_y, &dp[0][0]);
    for (int j = 0; j <= QRY_CORE; j++) {
      ndp[(REF_CORE+1)*j] = dp[j][REF_CORE];
    }

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
