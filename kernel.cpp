#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_barrier_multipod.h"
#include "unroll.hpp"
#include <cstdint>

#define GROUP_ID __bsg_x
#define NUM_GROUPS bsg_tiles_X
#define CORE_ID __bsg_y
#define CORES_PER_GROUP bsg_tiles_Y

// parameters
#define MAX_REF_CORE (MAX_SEQ_LEN / CORES_PER_GROUP)
#define NUM_TILES (bsg_tiles_X*bsg_tiles_Y)
#define MATCH     1
#define MISMATCH -1
#define GAP       1

inline int max(int a, int b) {
  return (a > b) ? a : b;
}

inline int max(int a, int b, int c) {
  return max(a, max(b,c));
}

inline int max(int a, int b, int c, int d) {
  return max(max(a,b), max(c,d));
}

// sequence info (double-buffered)
// qry_len: 0 = empty, -1 = stop, >0 = valid
struct seq_info_t {
  int qry_len;
  int ref_len;
  int seq_id;
};

seq_info_t info_curr = {0, 0, 0};
seq_info_t info_next = {0, 0, 0};

// inter-core mailbox
struct mailbox_t {
  int      dp_val;
  volatile int full;
  int      max_val;
  uint8_t  qry_char;
};

// local mailbox to receive data from left core
mailbox_t mailbox = {0, 0, 0, 0};
volatile int next_is_ready = 1;

// global buffers
uint8_t refbuf[MAX_REF_CORE + 1];
int H1[MAX_REF_CORE + 1];
int H2[MAX_REF_CORE + 1];

// Kernel main;
extern "C" int kernel(
  uint8_t* qry, uint8_t* ref,
  int* qry_lens, int* ref_lens,
  int* seq_counter, int num_seq,
  int* output, int pod_id)
{
  bsg_barrier_tile_group_init();
  bsg_barrier_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  // remote pointers
  mailbox_t *next_mailbox = (mailbox_t *)bsg_remote_ptr(__bsg_x, __bsg_y + 1, &mailbox);
  volatile int *prev_next_is_ready = (volatile int *)bsg_remote_ptr(__bsg_x, __bsg_y - 1, (void*)&next_is_ready);

  // remote pointer for forwarding seq info to next core
  seq_info_t *next_info = (seq_info_t *)bsg_remote_ptr(__bsg_x, __bsg_y + 1, &info_next);

  // main loop
  while (1) {

    if (CORE_ID == 0) {
      // atomically grab next sequence
      int s;
      asm volatile("amoadd.w %0, %1, (%2)"
        : "=r"(s) : "r"(1), "r"(seq_counter) : "memory");
      if (s >= num_seq) {
        info_curr.qry_len = -1;
      } else {
        info_curr.seq_id  = s;
        info_curr.ref_len = ref_lens[s];
        info_curr.qry_len = qry_lens[s];
      }
    } else {
      // wait for previous core to forward seq info
      int rdy = bsg_lr((int*)&(info_next.qry_len));
      if (rdy == 0) bsg_lr_aq((int*)&(info_next.qry_len));
      asm volatile("" ::: "memory");

      info_curr = info_next;
      info_next.qry_len = 0;
    }

    // forward to next core
    if (CORE_ID < CORES_PER_GROUP - 1) {
      next_info->seq_id  = info_curr.seq_id;
      next_info->ref_len = info_curr.ref_len;
      asm volatile("" ::: "memory");
      next_info->qry_len = info_curr.qry_len; // write last
    }

    // check for stop
    if (info_curr.qry_len < 0) break;

    // compute ref_core for this sequence
    int s        = info_curr.seq_id;
    int qry_len  = info_curr.qry_len;
    int ref_len  = info_curr.ref_len;
    int ref_core = ref_len / CORES_PER_GROUP;

    // DP row buffers
    int *H_curr = H1;
    int *H_prev = H2;

    for (int k = 0; k <= ref_core; k++) {
      H_prev[k] = 0;
    }

    int maxv = 0;

    // load reference chunk (batches of 8, then rest)
    uint8_t *ref_src = &ref[ref_len * s + (CORE_ID * ref_core)];
    int k = 0;
    for (; k + 8 <= ref_core; k += 8) {
      unrolled_load<uint8_t, 8>(&refbuf[k + 1], &ref_src[k]);
    }
    for (; k < ref_core; k++) {
      refbuf[k + 1] = ref_src[k];
    }

    // do dp calculation row by row
    for (int i = 0; i < qry_len; i++) {
      uint8_t qry_char;

      if (CORE_ID == 0) {
        qry_char = qry[qry_len * s + i];
        H_curr[0] = 0;
      } else {
        // wait for core to the left to write
        int rdy = bsg_lr((int*)&(mailbox.full));
        if (rdy == 0) bsg_lr_aq((int*)&(mailbox.full));
        asm volatile("" ::: "memory");

        H_curr[0] = mailbox.dp_val;
        qry_char = mailbox.qry_char;
        int left_maxv = mailbox.max_val;
        if (left_maxv > maxv) {
          maxv = left_maxv;
        }

        // indicate we are done with the buffer
        asm volatile("" ::: "memory");
        mailbox.full = 0;
        *prev_next_is_ready = 1;
      }

      for (int k = 1; k <= ref_core; k++) {
        int match      = (qry_char == refbuf[k]) ? MATCH : MISMATCH;

        int score_diag = H_prev[k-1] + match;
        int score_up   = H_prev[k]   - GAP;
        int score_left = H_curr[k-1] - GAP;

        int val = max(0, score_diag, score_up, score_left);
        H_curr[k] = val;

        if (val > maxv) {
          maxv = val;
        }
      }

      if (CORE_ID < CORES_PER_GROUP - 1) {
        // activate right core, wait for it to be empty
        int rdy = bsg_lr((int*)&next_is_ready);
        if (rdy == 0) bsg_lr_aq((int*)&next_is_ready);
        asm volatile("" ::: "memory");

        next_is_ready = 0;

        next_mailbox->dp_val = H_curr[ref_core];
        next_mailbox->max_val = maxv;
        next_mailbox->qry_char = qry_char;
        asm volatile("" ::: "memory");
        next_mailbox->full = 1;
      }

      // swap DP rows
      int *tmp = H_curr;
      H_curr = H_prev;
      H_prev = tmp;
    }

    if (CORE_ID == CORES_PER_GROUP - 1) {
      // write result
      output[s] = maxv;
    }
  }

  // kernel end;
  bsg_fence();
  bsg_barrier_tile_group_sync();
  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  return 0;
}
