#include <bsg_manycore_errno.h>
#include <bsg_manycore_cuda.h>
#include <cstdlib>
#include <time.h>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstdio>
#include <bsg_manycore_regression.h>
#include <bsg_manycore.h>
#include <cstdint>
#include <vector>
#include <map>

#define ALLOC_NAME "default_allocator"

void read_seq(const char* filename, uint8_t* seq, int num_seq) {
  FILE* file = fopen(filename, "r");
  for (int i = 0; i < num_seq; i++) {
    char temp_seq[64];
    fscanf(file, "%s", temp_seq); // skip line number;
    fscanf(file, "%s", temp_seq);
    for (int j = 0; j < 32; j++) {
      seq[(32*i)+j] = temp_seq[j]; 
    }
  } 
  fclose(file);
}


void read_output(const char* filename, int* output, int num_seq)
{
  FILE* file = fopen(filename, "r");
  for (int i = 0; i < num_seq; i++) {
    int score;
    fscanf(file, "%d", &score);
    output[i] = score;
  } 
  fclose(file);
}



// Host main;
int sw_multipod(int argc, char ** argv) {
  int r = 0;
  
  // command line;
  const char *bin_path = argv[1];
  const char *query_path = argv[2];
  const char *ref_path = argv[3];
  const char *output_path = argv[4];

  // parameters;
  int num_seq = NUM_SEQ;//NUM_SEQ; // per pod;
  int seq_len = 32;
  printf("num_seq=%d\n", num_seq);
  printf("seq_len=%d\n", seq_len);
  
  // prepare inputs;
  uint8_t* query = (uint8_t*) malloc(num_seq*(seq_len+1)*sizeof(uint8_t));
  uint8_t* ref = (uint8_t*) malloc(num_seq*(seq_len+1)*sizeof(uint8_t));
  int* output = (int*) malloc(num_seq*sizeof(int));
  read_seq(query_path, query, num_seq);
  read_seq(ref_path, ref, num_seq);
  read_output(output_path, output, num_seq);

 
  // initialize device; 
  hb_mc_device_t device;
  BSG_CUDA_CALL(hb_mc_device_init(&device, "sw_multipod", HB_MC_DEVICE_ID));

  eva_t d_query;
  eva_t d_ref;
  eva_t d_output;

  hb_mc_pod_id_t pod;
  hb_mc_device_foreach_pod_id(&device, pod)
  {
    printf("Loading program for pod %d\n", pod);
    BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
    BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

    // Allocate memory on device;
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, num_seq*(seq_len+1)*sizeof(uint8_t), &d_query));
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, num_seq*(seq_len+1)*sizeof(uint8_t), &d_ref));
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, num_seq*sizeof(int), &d_output));
   
    // DMA transfer;
    printf("Transferring data: pod %d\n", pod);
    std::vector<hb_mc_dma_htod_t> htod_job;
    htod_job.push_back({d_query, query, num_seq*seq_len*sizeof(uint8_t)});
    htod_job.push_back({d_ref, ref, num_seq*seq_len*sizeof(uint8_t)});
    BSG_CUDA_CALL(hb_mc_device_transfer_data_to_device(&device, htod_job.data(), htod_job.size()));

    // Cuda args;
    hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
    hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};
    #define CUDA_ARGC 4
    uint32_t cuda_argv[CUDA_ARGC] = {d_query, d_ref, d_output, pod};


    // Enqueue kernel;
    printf("Enqueue Kernel: pod %d\n", pod);
    BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel", CUDA_ARGC, cuda_argv));
  }


  // Launch pod;
  printf("Launching all pods\n");
  hb_mc_manycore_trace_enable((&device)->mc);
  BSG_CUDA_CALL(hb_mc_device_pods_kernels_execute(&device));
  hb_mc_manycore_trace_disable((&device)->mc);


  // Read from device;
  int* actual_output = (int*) malloc(num_seq*sizeof(int));

  bool fail = false;
  hb_mc_device_foreach_pod_id(&device, pod) {
    printf("Reading results: pods %d\n", pod);
    BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));

    // clear buf;
    for (int i = 0; i < num_seq; i++) {
      actual_output[i] = 0;
    }

    // DMA transfer; device -> host;
    std::vector<hb_mc_dma_dtoh_t> dtoh_job;
    dtoh_job.push_back({d_output, actual_output, num_seq*sizeof(int)});
    BSG_CUDA_CALL(hb_mc_device_transfer_data_to_host(&device, dtoh_job.data(), dtoh_job.size()));

    int H[33][33];
    int m[num_seq];
    int mj = 0, mk=0;
    for (int i = 0; i < num_seq; i++) {
      memset(H, 0, sizeof(H));
      m[i] = 0;
      for (int j = 0; j < 32; j++) {
        for (int k = 0; k < 32; k++) {
          int match = (query[32*i+j] == ref[32*i+k]) ? 1 : -1;
          int score_diag = H[j][k] + match;
          int score_up = H[j][k+1] - 1;
          int score_left = H[j+1][k] - 1;
          H[j+1][k+1] = std::max(0, std::max(score_diag, std::max(score_up, score_left)));
          if (H[j+1][k+1] > m[i]) {
            mj = j;
            mk = k;
            m[i] = H[j+1][k+1];
          }
        }
      }
    }

    // validate;
    printf("Max is at row %d, column %d\n", mj, mk);
    for (int i = 0; i < num_seq; i++) {
      int actual = actual_output[i];
      int expected = m[i];
      if (actual != expected) {
        fail = true;
        printf("Mismatch: i=%d, actual=%d, expected=%d\n", i, actual, expected);
      }
    }
  }


  // Finish;
  BSG_CUDA_CALL(hb_mc_device_finish(&device));
  if (fail) {
    return HB_MC_FAIL;
  } else {
    return HB_MC_SUCCESS;
  }
}


declare_program_main("sw_multipod", sw_multipod);
