//  AMOADD barrier

//  a0 = amo lock addr (in DRAM). Initialized to 0
//  a1 = sense word addr (in DMEM).  Initialized to 1

//  void bsg_barrier_amoadd(int*, int*);

.extern __bsg_x
.extern __bsg_y

.text
.globl bsg_barrier_amoadd
bsg_barrier_amoadd:
  // t0 - sense val
  // t1 - amo result
  // t2 - check val
  // t3 - wakeup val
  
  // t4 - base address
  // t5 - offset
  // t6 - end value

  // send amoadd
  lw t0, 0(a1)
  andi t0, t0, 3
  amoadd.w t1, t0, 0(a0)

  // is sense 1 or 2?
  // set wakeup val
  xori t3, t0, 3
 
  // calculate count
  li t2, (bsg_tiles_X*bsg_tiles_Y)-1
  sll t2, t2, t0
  srli t2, t2, 1

  // check count
  bne t2, t1, bsg_barrier_amoadd_sleep

  // set own sense
  sw t3, 0(a1)

  // reset counter and wake-up
  sw x0, 0(a0)
  li t5, bsg_tiles_Y
  slli t5, t5, 24

bsg_barrier_amoadd_wakeup:
  // and out potential sense bits
  andi t5, t5, -4

  // construct address mask
  li a4, (0x2003FFFF | ((bsg_tiles_Y - 1) << 24) | ((bsg_tiles_X - 1) << 18))

  // load in x and y values of tile
  la a2, __bsg_y
  la a3, __bsg_x
  lw a2, 0(a2)
  lw a3, 0(a3)

  // construct address
  li t4, 0x20000000
  slli a2, a2, 24
  add t4, t4, a2
  slli a3, a3, 18
  add t4, t4, a3
  add t4, t4, a1

  // initial end value
  addi t6, x0, 1
  slli t6, t6, 24

  // if waking up in x
  blt t5, t6, bsg_barrier_amoadd_continue_x
  // else wake up y
  j bsg_barrier_amoadd_wakeup_loop

bsg_barrier_amoadd_done:
  srli t5, t5, 18
  // if non-zero, just finished y
  bne t5, x0, bsg_barrier_amoadd_prepare_x
  // if done with x, return
  ret

bsg_barrier_amoadd_prepare_x:
  // prepare for x direction, change x value
  li t5, bsg_tiles_X
  slli t5, t5, 18

bsg_barrier_amoadd_continue_x:
  // set end value
  addi t6, x0, 1
  slli t6, t6, 18

bsg_barrier_amoadd_wakeup_loop:
  // halve distance
  srli t5, t5, 1

  // if done with this direction, branch
  blt t5, t6, bsg_barrier_amoadd_done

  // otherwise compute target address
  add a2, t4, t5
  and a2, a2, a4 // mask address
  add a3, t5, t3 // combine sense with distance
  sw a3, 0(a2)
  
  j bsg_barrier_amoadd_wakeup_loop

bsg_barrier_amoadd_sleep:
  lr.w t5, 0(a1)
  andi t0, t5, 3
  beq t3, t0, bsg_barrier_amoadd_wakeup
  // we need to check this, in order to support the reservation
  // being cleared by a context switch
  lr.w.aq t5, 0(a1)
  andi t0, t5, 3
  beq t3, t0, bsg_barrier_amoadd_sleep

bsg_barrier_amoadd_end:
  ret