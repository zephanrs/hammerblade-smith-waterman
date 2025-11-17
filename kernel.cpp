#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_barrier_multipod.h"
#include "unroll.hpp"
#include <cstdint>

// flags
#define SEQUENCES  8
#define SEQ_LENGTH 32

#define QRY_CORE (SEQ_LENGTH / bsg_tiles_Y)
#define REF_CORE (SEQ_LENGTH / bsg_tiles_X)

// parameters
#define NUM_TILES (bsg_tiles_X*bsg_tiles_Y)
#define MATCH     1
#define MISMATCH -1
#define GAP       1

// default buffer values
#define BUFFER { .right_rdy = 1, .bottom_rdy = 1, .max_left = -1, .max_top = -1 }


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


typedef struct {
  // dp matrix and reference/query
  int      dp[QRY_CORE+1][REF_CORE+1];
  uint8_t  qrybuf[QRY_CORE];
  uint8_t  refbuf[REF_CORE];

  // valid/ready signals and max value
  int right_rdy;
  int bottom_rdy;
  int max_left;
  int max_top;

  // remote pointers
  int      *left_rdy;
  int      *top_rdy;
  int      *right_dp;
  int      *bottom_dp;
  int      *right_max;
  int      *bottom_max;
  uint8_t  *next_qry;
  uint8_t  *next_ref;
} buffer_t;

static void init_buffer(buffer_t *b, int x, int y) {
  b->left_rdy   = (int *)     bsg_remote_ptr(x - 1, y, &b->right_rdy );
  b->top_rdy    = (int *)     bsg_remote_ptr(x, y - 1, &b->bottom_rdy);
  b->right_dp   = (int *)     bsg_remote_ptr(x + 1, y, &b->dp[0][0]  );
  b->bottom_dp  = (int *)     bsg_remote_ptr(x, y + 1, &b->dp[0][0]  );
  b->right_max  = (int *)     bsg_remote_ptr(x + 1, y, &b->max_left  );
  b->bottom_max = (int *)     bsg_remote_ptr(x, y + 1, &b->max_top   );
  b->next_qry   = (uint8_t *) bsg_remote_ptr(x + 1, y, &b->qrybuf[0] );
  b->next_ref   = (uint8_t *) bsg_remote_ptr(x, y + 1, &b->refbuf[0] );
}

static buffer_t buffers[2] = { BUFFER, BUFFER };

// current buffer
buffer_t *curr;

// Kernel main;
extern "C" int kernel(uint8_t* qry, uint8_t* ref, int* output, int pod_id)
{
  bsg_barrier_tile_group_init();
  bsg_barrier_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  init_buffer(&buffers[0], __bsg_x, __bsg_y);
  init_buffer(&buffers[1], __bsg_x, __bsg_y);

  for (int i = 0; i < SEQUENCES; i++) {
    curr = &buffers[i & 0x1];

    if (!__bsg_y) {
      // load reference
      unrolled_load<REF_CORE>(
        curr->refbuf,
        &ref[SEQ_LENGTH * i + (__bsg_x * REF_CORE)]
      );
    }

    if (!__bsg_x) {
      // load query
      unrolled_load<QRY_CORE>(
        curr->qrybuf,
        &qry[SEQ_LENGTH * i + (__bsg_y * QRY_CORE)]
      );
    }
    
    if (__bsg_y) {
      // wait for core above to write
      int rdy = bsg_lr(&(curr->max_top));
      if (rdy == -1) bsg_lr_aq(&(curr->max_top));
      asm volatile("" ::: "memory");
    }

    if (__bsg_x) {
      // wait for core to the left to write
      int rdy = bsg_lr(&(curr->max_left));
      if (rdy == -1) bsg_lr_aq(&(curr->max_left));
      asm volatile("" ::: "memory");
    }
    
    // derive maximum value
    int maxv = max(0, curr->max_top, curr->max_left);
    
    // do dp calculation
    for (int j = 1; j <= QRY_CORE; j++) {
      for (int k = 1; k <= REF_CORE; k++) {
        int match      = (curr->qrybuf[j-1] == curr->refbuf[k-1]) ? MATCH : MISMATCH;

        int score_diag = curr->dp[j-1][k-1] + match;
        int score_up   = curr->dp[j-1][k]   - GAP;
        int score_left = curr->dp[j][k-1]   - GAP;

        int val = max(0, score_diag, score_up, score_left);

        if (val > maxv) {
          maxv = val;
        }

        curr->dp[j][k] = val;
      }
    }
    
    if (__bsg_y < (bsg_tiles_Y - 1)) {
      // ensure bottom core is ready to receive
      int rdy = bsg_lr(&(curr->bottom_rdy));
      if (!rdy) bsg_lr_aq(&(curr->bottom_rdy));
      asm volatile("" ::: "memory");

      // copy dp below
      for (int j = 0; j <= REF_CORE; j++)
        curr->bottom_dp[j] = curr->dp[QRY_CORE][j];

      // copy reference below
      for (int j = 0; j < REF_CORE; j++)
        curr->next_ref[j] = curr->refbuf[j];

      // activate bottom core (and transfer max)
      *(curr->bottom_max) = maxv;

      // indicate buffer is in-use
      curr->bottom_rdy = 0;
    }

    if (__bsg_x < (bsg_tiles_X - 1)) {
      // ensure right core is ready to receive
      int rdy = bsg_lr(&(curr->right_rdy));
      if (!rdy) bsg_lr_aq(&(curr->right_rdy));
      asm volatile("" ::: "memory");

      // copy dp right
      for (int j = 0; j <= QRY_CORE; j++)
        curr->right_dp[(REF_CORE+1)*j] = curr->dp[j][REF_CORE];

      // copy query below
      for (int j = 0; j < QRY_CORE; j++)
        curr->next_qry[j] = curr->qrybuf[j];

      // activate right core (and transfer max)
      *(curr->right_max) = maxv;

      // indicate buffer is in-use
      curr->right_rdy = 0;
    }

    // invalidate buffer
    curr->max_left = -1;
    curr->max_top  = -1;

    // indicate buffer is free
    if (__bsg_x) *(curr->left_rdy) = 1;
    if (__bsg_y) *(curr->top_rdy)  = 1;

    // write result
    if ((__bsg_x == (bsg_tiles_X - 1)) && (__bsg_y == (bsg_tiles_Y - 1)))
      output[i] = maxv;
  }
  
  // kernel end;
  bsg_fence();
  bsg_barrier_tile_group_sync();
  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  return 0;
}
