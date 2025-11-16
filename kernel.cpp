#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_barrier_multipod.h"
#include <cstdint>

// flags
#define SEQUENCES  1
#define SEQ_LENGTH 32

// parameters
#define NUM_TILES       (bsg_tiles_X*bsg_tiles_Y)
#define MATCH_SCORE     1
#define MISMATCH_SCORE -1
#define GAP             1

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

int maxv;
int dp[3][33];

// Kernel main;
extern "C" int kernel(uint8_t* qry, uint8_t* ref, int* output, int pod_id)
{
  bsg_barrier_tile_group_init();
  bsg_barrier_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  for (int i = 0; i <= 32; i++) 
    dp[0][i] = 0;
  
  dp[0][1] = 0;
  dp[0][2] = 0;
  maxv = 0;

  if (__bsg_x == 0) {
    // wait for core above to write
    bsg_lr(&maxv);
    if (__bsg_y) bsg_lr_aq(&maxv);
    
    // do dp calculation
    for (int i = 1; i <= 2; i++) {
      for (int j = 1; j <= 32; j++) {
        int match = (qry[__bsg_y * 2 + i] == ref[j]) ? 1 : -1;

        int score_diag = dp[i-1][j-1] + match;
        int score_up = dp[i-1][j] - 1;
        int score_left = dp[i][j-1] - 1;

        dp[i][j] = max(0, score_diag, score_up, score_left);

        if (dp[i][j] > maxv) {
          maxv = dp[i][j];
        }
      }
    }

    // copy over data to next core
    if (__bsg_y < 15) {
      int *ndp =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &dp[0][0]);
      for (int j = 1; j <= 32; j++) {
        ndp[j] = dp[2][j];
      }

      // activate next core (and transfer max)
      int *nmax =  (int*) bsg_remote_ptr(__bsg_x, __bsg_y+1, &maxv);
      *nmax     = maxv;
    } else *output = maxv;
  }
  
  // kernel end;
  bsg_fence();
  bsg_barrier_tile_group_sync();
  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  return 0;
}
