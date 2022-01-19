// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
// (C) Copyright [2020-2021] Hewlett Packard Enterprise Development LP
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//  
// author:  Nathan Wichmann  (wichmann@hpe.com)
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
//  This file implements the Highly Asychronous Buffered Update (HABU) Library.
//  HABU is a library that does message aggregation, data transfer, 
//  and updates scattered across the machine.
//  HABU utilizes shmem but is meant to be relatively communication library agnostic.
//  The goal is to accelerate global random updates via message aggregation and 
//  to dramatically improve programmer productivity by adding new capabilities 
//  such as user defined remote functions, including recursive HABU calls.

//  HABU assumes what is call a Symmetrically One-sided communication epoch.
//  Symmetrically One-sided epoch is a situation where the user programs update 
//  or communication patterns in a way that (mostly) appears to be one-sided but
//  During these update patterns all participating PEs and threads are actively calling HABU.
//  HABU assumes all participating PEs are continuously making calls to HABU.
//  The library buffers up smaller references, transfers those buffers between PEs, 
//  and perform the “updates”.  The user is NOT actively engaged in coding the actions 
//  on both the source and target sides.  A call to habu_barrier executes the "endgame",
//  I.E., drains all of the buffers and sychronizes all of the PEs and contexts.

//  HABU does not assume any OS or thread library support and can live entirely in user space.
//  HABU is thread safe as long as the user calls HABU with unique context ids.

#include <stdio.h>
#include <string.h>
// #include <time.h>
#include <sys/time.h> 
#include <sys/times.h>
#include <stdlib.h>

#include <unistd.h>
#include <getopt.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>
#include <mpp/shmem.h>
#include <mpp/shmemx.h>
#include <math.h>
#include "habu.h"


//  Defines the size of the buffers used in habu aggregation and communication.
#define l2_habu_base_buffsize 14
#define habu_base_buffsize (1L<<l2_habu_base_buffsize)
#define unchar unsigned char
#define habu_ipe_t int32_t
#define size_habu_ipe_t 4
// #define size_habu_op_t 1


/* Macros for timing */
struct tms t;
#define WSEC() (times(&t) / (double)sysconf(_SC_CLK_TCK))
#define CPUSEC() (clock() / (double)CLOCKS_PER_SEC)
#define min(a,b)( a < b ? a : b )
#define max(a,b)( a > b ? a : b )

double get_time() {
  struct timeval tp;
  int retVal = gettimeofday(&tp,NULL);
  if (retVal == -1) { perror("gettimeofday:"); fflush(stderr); }
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
// double get_time()
// {
//   struct timespec tp;
//   clock_gettime(CLOCK_MONOTONIC, &tp);
//   return (((double)tp.tv_sec) + 1e-9 * ((double)tp.tv_nsec));
// }


char *bartime_name;
double *total_bartime;
double *total_quiettime;
double *total_opertime;
double *memcpytime;


/* Package Information */
#define HABU_PACKAGE_NAME "Highly Asychronous Buffered Updates"
#define HABU_PACKAGE_VERSION "2.0"

#define oldbuffid(ipe,ctxt) ((ipe+ctxt*(npes+maxslices)))

//  buffid takes in a groupid, a memberid, a sliceid, and a context number to compute
//  the buffid that is unique to that combination.
//  If we are in a intergroup phase, the groupid should be 0<=groupid<ngroups and the 
//  memberid and sliceid should be zeros.
//  If we are in the intragroup phase, the groupid should be ngroups, shifting the bid
//  to the region of bid used for intragroup sorting.  0<=memberid<maxmembers and 0<=sliceid<maxslices.
//  The last context number shifts the bid to a set of bids that are unique to that context.
#define buffid(groupid,memberid,sliceid,ctxt) (groupid+memberid*maxctxt+sliceid+ctxt*(ngroups+npes_per_group[0]*maxctxt))

#define maxprocess 9
  


int64_t buffsize;
int64_t *processcalls;
int64_t *treceived;
int *done;
int64_t *active_buffs;
uint64_t *recvheader;
unchar *recvbuff;
int64_t *recvready;
unchar *sendbuff;
int64_t *buffcount;
int *npes_per_group;
int ngroups, groupleader, npes_my_group,mygid, mymid;

int* ticktock;


int *recvedgroup;
int *recvedtid;
int nendpoints;

double inv_numelems[habu_max_memory_register];
int handletaken[habu_max_memory_register];
int item_size[habu_max_memory_register];
int offset_bytesize[habu_max_memory_register];
int op_handletaken[habu_max_opcodes];

habu_op_t HABU_PUT;
habu_op_t HABU_ADD;
habu_op_t HABU_INC;
habu_op_t HABU_GET;
habu_op_t HABU_FADD;
habu_op_t HABU_PUTV;
habu_op_t HABU_GETV;

void *habu_target_pointer[habu_max_memory_register];

int mype;
int npes;
int maxctxt,maxslices;
int num_active_ctxt;

int habu_payload_size[habu_max_opcodes];
void *habu_local_args[habu_max_opcodes];

void (*habu_op_ptr_arr[habu_max_opcodes])(habu_mem_t target_handle, int64_t lindex, void *payload,void *local_args, const int ctxt);
void (*habu_fop_ptr_arr[habu_max_opcodes])(habu_mem_t return_handle,int64_t return_lindex ,int return_pe,habu_mem_t target_handle, int64_t lindex, void *payload, void *local_args, const int ctxt);


void show_mem_rep(char *start, int n){ 
    int i; 
    for (i = 0; i < n; i++) 
         printf(" %.2x", start[i]); 
    printf("\n"); 
}

int64_t habu_local_reference( habu_mem_t thandle, int64_t ioffset){
//  Handy for debugging.  It just allows one to check the value of the local table using a handle
//  instead of a pointer.  
  int64_t *array = habu_handle_to_pointer(thandle);
  return array[ioffset];
}
void *habu_handle_to_pointer( habu_mem_t thandle){
//  Handy for debugging.  It just allows one to check the value of the local table using a handle
//  instead of a pointer.  
  return habu_target_pointer[thandle];
}
int habu_sizeof_item( habu_mem_t thandle){
//  Handy for debugging.  It just allows one to check the value of the local table using a handle
//  instead of a pointer.  
  return item_size[thandle];
}

void habu_set_num_contexts(int nc){
  shmem_barrier_all();
  num_active_ctxt=nc;
}

int habu_get_max_contexts(){ return maxctxt;}
int habu_get_num_contexts(){ return num_active_ctxt;}
void habu_inc_operation(habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *payload = p;
  int64_t *array = habu_handle_to_pointer(thandle);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu inc offset %ld\n",ioffset); fflush(stdout);
#endif  
  array[ioffset] += 1;
  treceived[ctxt]++;
}
void habu_add_operation(habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *payload = p;
  int64_t *array = habu_handle_to_pointer(thandle);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu adding offset %ld\n", ioffset); fflush(stdout);
#endif  
  array[ioffset] += payload[0];
}

void habu_put_operation(habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *payload = p;
  int64_t *array = habu_handle_to_pointer(thandle);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu putting offset %ld\n", ioffset); fflush(stdout);
#endif  
  array[ioffset] = payload[0];
}

void habu_putv_operation(habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int *num_items = p;
  unchar *payload = p+sizeof(int);
  unchar *array = habu_handle_to_pointer(thandle);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  int64_t *val = p+sizeof(int);
  printf("habu putving offset %ld num_items=%d val=%ld\n", ioffset,num_items[0],val[0]); fflush(stdout);
#endif  
  memcpy(&array[ioffset*item_size[thandle]], &payload[0],num_items[0]*item_size[thandle]);
}


void habu_getv_operation(habu_mem_t rhandle, int64_t return_offset, int return_pe, habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *num_items = p;
  int64_t *payload = p;
  unchar *array = habu_handle_to_pointer(thandle);
  
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu get offset %ld num_items=%ld\n", ioffset, num_items[0]); fflush(stdout);
#endif  
  habu_putv(rhandle,return_offset,return_pe,&array[ioffset*item_size[thandle]],num_items[0],ctxt);
}

void habu_fadd_operation(habu_mem_t rhandle, int64_t return_offset, int return_pe, habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *payload = p;
  int64_t *array = habu_handle_to_pointer(thandle);

#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu fadd offset %ld\n", ioffset); fflush(stdout);
#endif  
  habu_op( rhandle,return_offset,return_pe, HABU_PUT,&array[ioffset],ctxt); 
  array[ioffset] += payload[0];
}
void habu_get_operation(habu_mem_t rhandle, int64_t return_offset, int return_pe, habu_mem_t thandle, int64_t ioffset, void *p,void *largs, int ctxt){
  int64_t *payload = p;
  int64_t *array = habu_handle_to_pointer(thandle);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("habu get offset %ld\n", ioffset); fflush(stdout);
#endif  
  habu_op( rhandle,return_offset,return_pe, HABU_PUT,&array[ioffset],ctxt); 
}



habu_op_t habu_register_op(void (*op_ptr_arr)(habu_mem_t, int64_t, void *,void *largs, int ctxt ), int payload_size,void * largs){
  shmem_barrier_all();  //  Force this to be a collective.
  for(int i=0;i<habu_max_opcodes;i+=2){
    if(op_handletaken[i]==0){
      habu_op_ptr_arr[i] = op_ptr_arr; //  Keep track of the pointer
      op_handletaken[i]=1;  //  Mark this handle as taken.
      habu_payload_size[i]=payload_size;
      habu_local_args[i]=largs;
//       printf("%d setting op handle %d\n",mype,i);
      return i;  //  Return this handle id
    }
  }
  //  If you got to here we have run out of handles.  No need to collapse, but 
  //  issue a warning and return an error code.
  printf("WARNING in habu_register_op.  You have run out of habu op handles.\n");
  return -1;
}
habu_op_t habu_register_fop(void (*fop_ptr_arr)(habu_mem_t,int64_t,int,habu_mem_t, int64_t, void*,void *largs,int), int payload_size,void * largs){
  shmem_barrier_all();  //  Force this to be a collective.
  for(int i=1;i<habu_max_opcodes;i+=2){
    if(op_handletaken[i]==0){
      habu_fop_ptr_arr[i] = fop_ptr_arr; //  Keep track of the pointer
      op_handletaken[i]=1;  //  Mark this handle as taken.
      habu_payload_size[i]=payload_size;
      habu_local_args[i]=largs;
//       printf("%d setting handle %d\n",mype,i);
      return i;  //  Return this handle id
    }
  }
  //  If you got to here we have run out of handles.  No need to collapse, but 
  //  issue a warning and return an error code.
  printf("WARNING in habu_register_fop.  You have run out of habu op handles.\n");
  return -1;
}


habu_mem_t habu_register_memory(void *target, int this_item_size,int64_t num_local_elements){
//  This routine registers a local pointer with habu, saving the pointer for later
//  use and returning a handle.  We use this because we want to have habu be
//  implementation dependent and not rely on symmetric addresses.
//  This also allows use to have a small field for the registration handle and more
//  easily pack the handle in the header.  
  shmem_barrier_all();  //  Force this to be a collective.
  
  for(int i=0;i<habu_max_memory_register;i+=1){
    if(handletaken[i]==0){
      habu_target_pointer[i] = target;  //  Keep track of the pointer
      double ne=num_local_elements;
//       num_active_ctxt = num_ctxt;
//       double nc=num_ctxt;
      if(ne<=0 || num_active_ctxt==1){
        //  The user is does NOT want atomicity, perhaps because they are running without threads.
        inv_numelems[i] = -1.;  
      }else{
//         inv_numelems[i] = 1.0/(ne/nc+1.0); 
//         inv_numelems[i] = 1.0/((ne+maxctxt)/nc); 
//         inv_numelems[i] = (1.0*nc)/(ne+maxctxt); 
        inv_numelems[i] = (1.0)/(ne+maxctxt); 
      }
//       printf("inv_numelems= %e\n",inv_numelems[i]);
      handletaken[i]=1;  //  Mark this handle as taken.
      item_size[i] = this_item_size;
//       printf("%d setting handle %d\n",mype,i);
      return i;  //  Return this handle id
    }
  }
  //  If you got to here we have run out of handles.  No need to collapse, but 
  //  issue a warning and return an error code.
  printf("WARNING in habu_register.  You have run out of habu memory handles.\n");
  return -1;
}



void habu_unregister_memory(habu_mem_t handle){shmem_barrier_all();handletaken[handle]=0;}  //  Mark this handle as free.
void habu_unregister_op(habu_mem_t handle){shmem_barrier_all(); op_handletaken[handle]=0; } //Mark handle as free. 
void habu_unregister_fop(habu_mem_t handle){shmem_barrier_all(); op_handletaken[handle]=0; } //Mark handle as free. 


int habu_init(int mctxt){
  npes = shmem_n_pes();
  mype = shmem_my_pe();
  maxctxt = mctxt ;
  
  buffsize = habu_base_buffsize;
  
  
  npes_per_group = (int*)shmem_malloc(4);
  npes_per_group[0] = 1;
  
  if(mype == 0 ){
    int ppn = shmemx_local_npes()/2;
//     ppn = 1;
    for(int i=2;i<=ppn;i++){
      if(npes%i==0 && ppn%i==0){
        //  Find the largest factor of both npes and the number of local pes.
        npes_per_group[0] = i;
      }
    }
    for(int i=1;i<npes;i++){
      shmem_put32(npes_per_group,npes_per_group,1,i);
    }

  }

  ngroups = npes/npes_per_group[0];
  shmem_barrier_all();  
  ngroups = npes/npes_per_group[0];
  groupleader = mype/npes_per_group[0]*npes_per_group[0];
  npes_my_group = npes_per_group[0];
  mygid = mype/npes_per_group[0];
  mymid = mype%npes_per_group[0];
  int maxslices = maxctxt * npes_per_group[0];
  
  nendpoints = (ngroups+npes_my_group*maxctxt) * maxctxt;

  if(mype==0){
    printf("\nHABU: %s Version %s \n", HABU_PACKAGE_NAME, HABU_PACKAGE_VERSION);
    printf("HABU: habu_base_buffsize = %ld bytes\n",habu_base_buffsize);
    printf("HABU: Total buffsize for each PE  = %ld Kbytes\n",2*buffsize*nendpoints/1024);
    printf("HABU: NPES = %d MAX CONTEXTS=%d\n",npes,maxctxt);
    printf("HABU:  npes_per_group= %d ngroups= %d \n",npes_per_group[0],ngroups);
  }
  
  if((mype <= 2*npes_per_group[0]) || (mype >= npes-2*npes_per_group[0]) ){
//     printf("HABU:  %d npes_per_group= %d ngroups= %d groupleader= %d maxslices= %d mygid=%d mymid=%d \n",mype,npes_my_group,ngroups,groupleader,maxslices,mygid, mymid);
  }
  
  done = (int*)shmem_malloc(64*maxslices);
  if(done==NULL){printf("Error. Malloc of done failed.\n"); return 1;  }
  active_buffs = (int64_t*)shmem_malloc(64*maxslices);
  if(active_buffs==NULL){printf("Error. Malloc of active_buffs failed.\n"); return 1;  }
  ticktock = (int*)shmem_malloc(64*maxslices);
  if(ticktock==NULL){printf("Error. Malloc of ticktock failed.\n"); return 1;  }

  recvheader = (uint64_t*)shmem_malloc(8*nendpoints);
  if(recvheader==NULL){printf("Error. Malloc of recvheader failed.\n"); return 1;  }
  recvbuff = (unchar*)shmem_malloc(buffsize*nendpoints);
  if(recvbuff==NULL){printf("Error. Malloc of recvbuff failed.\n"); return 1;  }
  recvready = (int64_t*)shmem_malloc(8*nendpoints*8);
  if(recvready==NULL){printf("Error. Malloc of recvready failed.\n"); return 1;  }

  sendbuff = (unchar*)shmem_malloc(buffsize*nendpoints);
  if(sendbuff==NULL){printf("Error. Malloc of sendbuff failed. buffsize=%ld nendpoints=%d\n", buffsize,nendpoints); return 1;  }
  buffcount = (int64_t*)shmem_malloc(8*nendpoints);
  if(buffcount==NULL){printf("Error. Malloc of buffcount failed.\n"); return 1;  }

  processcalls = (int64_t*)shmem_malloc(8*maxslices);
  if(processcalls==NULL){printf("Error. Malloc of processcalls failed.\n"); return 1;  }
  treceived = (int64_t*)shmem_malloc(8*maxslices);
  if(treceived==NULL){printf("Error. Malloc of processcalls failed.\n"); return 1;  }
  recvedgroup = (int*)shmem_malloc(4*maxslices*16);
  if(recvedgroup==NULL){printf("Error. Malloc of recvedgroup failed.\n"); return 1;  }
  recvedtid = (int*)shmem_malloc(4*maxslices*16);
  if(recvedtid==NULL){printf("Error. Malloc of recvedtid failed.\n"); return 1;  }
  
  bartime_name = (char*)malloc(16*8*sizeof(char));
  if(bartime_name==NULL){printf("Error. Malloc of bartime_name failed.\n"); return 1;  }
  total_bartime = (double*)shmem_malloc(8*maxslices*sizeof(double));
  if(total_bartime==NULL){printf("Error. Malloc of total_bartime failed.\n"); return 1;  }
  total_quiettime = (double*)shmem_malloc(8*maxslices*sizeof(double));
  if(total_quiettime==NULL){printf("Error. Malloc of total_quiettime failed.\n"); return 1;  }
  total_opertime = (double*)shmem_malloc(8*maxslices*sizeof(double));
  if(total_opertime==NULL){printf("Error. Malloc of total_opertime failed.\n"); return 1;  }
  memcpytime = (double*)shmem_malloc(8*maxslices*sizeof(double));
  if(memcpytime==NULL){printf("Error. Malloc of memcpytime failed.\n"); return 1;  }
    
  for( int i=0;i<nendpoints;i+=1) buffcount[i]=0;  
  for( int i=0;i<nendpoints;i+=1) recvheader[i]=0;
  for( int i=0;i<nendpoints;i+=1) recvready[i]=1;

  for( int i=0;i<maxslices*16;i+=1) recvedgroup[i]=0;
  for( int i=0;i<maxslices*16;i+=1) recvedtid[i]=0;
  for( int i=0;i<maxslices*8;i+=1) processcalls[i]=0;
  for( int i=0;i<maxslices*8;i+=1) treceived[i]=0;
  for( int i=0;i<maxslices*16;i+=1) done[i]=0;
  for( int i=0;i<maxslices*16;i+=1) ticktock[i]=0;
  for( int i=0;i<maxslices*16;i+=1) active_buffs[i]=0;
  for( int i=0;i<habu_max_memory_register;i+=1) handletaken[i]=0;
  for( int i=0;i<habu_max_opcodes;i+=1) op_handletaken[i]=0;

  strncpy(&bartime_name[0*16], "Tbarrier",16);
  strncpy(&bartime_name[1*16], "barflush",16);
  strncpy(&bartime_name[2*16], "barsig",16);
  strncpy(&bartime_name[3*16], "barwait",16);
  strncpy(&bartime_name[4*16], "barclear",16);
  for( int i=0;i<8*maxslices;i+=1) total_bartime[i]=0.;
  for( int i=0;i<8*maxslices;i+=1) total_quiettime[i]=0.;
  for( int i=0;i<8*maxslices;i+=1) total_opertime[i]=0.;
  for( int i=0;i<8*maxslices;i+=1) memcpytime[i]=0.;

  
  HABU_INC  = habu_register_op (habu_inc_operation ,0,NULL);
  HABU_PUT  = habu_register_op (habu_put_operation ,8,NULL);
  HABU_PUTV = habu_register_op (habu_putv_operation,4,NULL);
  HABU_ADD  = habu_register_op (habu_add_operation ,8,NULL);
  HABU_GET  = habu_register_fop(habu_get_operation ,0,NULL);
  HABU_GETV = habu_register_fop(habu_getv_operation,8,NULL);
  HABU_FADD = habu_register_fop(habu_fadd_operation,8,NULL);


  
  
  habu_set_num_contexts(maxctxt);
  
  shmem_barrier_all();  
  if(mype==0) printf("HABU: INIT COMPLETE\n\n");
  return 0;
  
  }

void habu_stats(int ctxt_id){
//   printf("%d %d HABU called process_requests %ld times\n",mype,ctxt_id,processcalls[ctxt_id]);
//    printf("%d %d HABU treceived %ld \n",mype,ctxt_id,treceived[ctxt_id]);
  mype = shmem_my_pe();
  npes = shmem_n_pes();
  double mint, maxt, avgt;
  int64_t imin, imax, iavg;
  
  for( int ibt=0;ibt<5;ibt++){
  if(mype==0){
    mint = maxt = avgt = total_bartime[ctxt_id*8+ibt];
    int minpe=0;
    for(int i=0;i<npes;i++){
      double t;
      shmem_get64(&t,&total_bartime[ctxt_id*8+ibt],1,i);
      if(t<mint){
        mint = t;
	minpe = i;
      }
      //mint = min(mint,t);
      maxt = max(maxt,t);
      avgt += t;
    }
//     printf("HABU %16s time min, avg, max  = %lf %d %lf %lf \n",&bartime_name[ibt*16],mint,minpe, avgt/npes,maxt);
  }
  }
  if(mype==0){
    mint = maxt = avgt = total_quiettime[ctxt_id*8];
    for(int i=1;i<npes;i++){
      double t;
      shmem_get64(&t,&total_quiettime[ctxt_id*8],1,i);
      mint = min(mint,t);
      maxt = max(maxt,t);
      avgt += t;
    }
//     printf("HABU shmem_quiet time min, avg, max  = %lf %lf %lf \n",mint, avgt/npes,maxt);
//     printf("HABU operations time min, avg, max  = %lf %lf %lf \n",mint, avgt/npes,maxt);
    mint = maxt = avgt = memcpytime[ctxt_id*8];
    for(int i=1;i<npes;i++){
      double t;
      shmem_get64(&t,&memcpytime[ctxt_id*8],1,i);
      mint = min(mint,t);
      maxt = max(maxt,t);
      avgt += t;
    }
//     printf("HABU memcpy time min, avg, max  = %lf %lf %lf \n",mint, avgt/npes,maxt);
  }
}



void habu_operations(void *rb,int64_t ireceived,int channel,int ctxt){
//  This routine examines the buffer and either forwards operation to another context based on the sliceid,
//  or, if atomicity is off or this slice already owns that index, directly call the function
//  for the execution of the operation.
//  The forwarding path is a recursive call but allows us to reuse all of the habu machinery.
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
  printf("%d %d in forwops irecv=%ld\n", mype,ctxt,ireceived);fflush(stdout);
#endif
  int pi=0;
  int i=0;
  unchar *urb = rb;
//   double ot = get_time();
  while(i<ireceived){
      habu_op_t opcode;
      habu_mem_t thandle;
      int64_t offset=0;
      habu_ipe_t ipe = 0;
      
  //     memcpy(&opcode, &rb[i],size_habu_op_t); i+=size_habu_op_t;
      opcode=urb[i]; i+=size_habu_op_t;
      thandle=urb[i];i+=size_habu_mem_t;
      memcpy(&offset, &rb[i],size_int64_t);i+=size_int64_t;
      if(channel==0) { memcpy(&ipe, &rb[i],size_habu_ipe_t);i+=size_habu_ipe_t;}
      if(opcode%2==0){
        int num_items=0;
        if(channel==1){
          //  Execute the operation.
          #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
          printf("%d %d executing th=%d opc=%d i=%d ioff=%ld \n", mype,ctxt,thandle,opcode,i,offset);fflush(stdout);
          #endif
          //  We still need to know the num_items so we can advance i later.
          //  We don't need to advance i for the num_items space because that is considered the payload.
          if(opcode==HABU_PUTV) memcpy(&num_items,&rb[i],sizeof(num_items));

          habu_op_ptr_arr[opcode](thandle,offset,&rb[i],habu_local_args[opcode],ctxt);
        }else{
          #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
          printf("%d %d forward i=%d ipe=%d channel=%d ioff=%ld ni=%d ibc=%d opc=%d\n", mype,ctxt,i,mype,channel,offset,num_items,i,opcode);fflush(stdout);
          #endif
          //  Forward the operation to the ipe that owns this slice
          if(opcode==HABU_PUTV){
            memcpy(&num_items,&rb[i],sizeof(num_items));
            habu_putv( thandle, offset,ipe, &rb[i+sizeof(num_items)], num_items,ctxt); 
          }else{
            habu_op( thandle,offset,ipe, opcode,&rb[i],ctxt); 
          }
        }
        i+=num_items*item_size[thandle];  //  If PUTV
        i+=habu_payload_size[opcode];
      }else{
        int rpe;
        habu_mem_t rhandle;
        int64_t loff=0;
        int64_t num_items=0;
        memcpy(&rpe, &rb[i],sizeof(int));i+=sizeof(int);
        memcpy(&rhandle, &rb[i],size_habu_mem_t); i+=size_habu_mem_t;
        memcpy(&loff, &rb[i],size_int64_t);i+=size_int64_t;
        if(opcode==HABU_GETV) memcpy(&num_items,&rb[i],sizeof(num_items));
        if(channel==1){
          habu_fop_ptr_arr[opcode](rhandle,loff,rpe, thandle, offset, &rb[i], habu_local_args[opcode], ctxt);
        }else{
          #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
          printf("%d %d forward fop %ld %d i=%d ipe=%d ioff=%ld\n", mype,ctxt, loff, rpe, i,mype,offset);fflush(stdout);
          #endif
          if(opcode==HABU_GETV){
            habu_getv(rhandle,loff,rpe,thandle,offset,ipe,num_items,ctxt);
          }else{
            habu_fop( rhandle,loff,rpe,thandle,offset,ipe, opcode,&rb[i],ctxt); 
          }
        }
        i+=habu_payload_size[opcode];
      }
      
    }
//   total_opertime[ctxt*8] += get_time()-ot;
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
  printf("%d %d exiting forwops irecv=%ld\n", mype,ctxt,ireceived);fflush(stdout);
#endif
  
}


int64_t process_requests(int ctxt){
//  This routine processes requests that are arriving from other PEs.
//  The idea is to check buffs to see if they contain anything, and if they do 
//  either forward or execute the operations.  
//  After the requests have been processed signal the sending PE that it is
//  safe to send more.  
//  We have a loop over PEs and we have a loop over contexts since they are very different scales
//  and they do different things.
  int64_t one=1;
  int64_t tr = 0;
//   printf("%d %d habu in process_requests\n",mype,ctxt);
  for(int icount=0;icount<maxprocess;icount+=1){  // Loop with a limit.
    // Check a few buffs to see if any are full.
    int rgid = recvedgroup[ctxt*16];  // This is our recvedgroup that we are checking.
    int bid  = rgid+ctxt*(ngroups + npes_my_group*maxctxt);
//  int bid = buffid(rgid,0,0,ctxt);

    int64_t ireceived = recvheader[bid];
    tr += ireceived;
//       printf("%d %d habu checking process_requests %d %d ireceived=%ld\n",mype,ctxt,rpe,ctxt,ireceived);
    recvedgroup[ctxt*16] = (recvedgroup[ctxt*16]+1)%ngroups; // Continuous loop on pes.
    if(ireceived>0){
      int rpe = rgid*npes_my_group+mymid;
      int rbid = mygid+ctxt*(ngroups + npes_my_group*maxctxt);
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu process_requests after bid=%d rpe=%d ireceived=%ld\n",mype,ctxt,bid,rpe,ireceived);fflush(stdout);
#endif        
      // We got something!  Process the updates.
      recvheader[bid]=0L;  //  Mark the buff as empty
      //  We have to make a stack copy of recvbuff since the habu_operations 
      // recursively calls habu by its very nature.
      unchar rb[ireceived];  
      memcpy(rb,&recvbuff[buffsize*bid],ireceived);
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu sending intergroup recvready to rbid=%d %d after ireceived=%ld!\n", mype,ctxt,rbid,rpe,ireceived);fflush(stdout);
#endif        
      //  Signal the sending PE we are ready to receive more.
      // We can send the signal already because we have made a copy of the recvbuff.
       shmem_put64(&recvready[rbid],&one,1,rpe);  
//      shmem_int64_atomic_add(&recvready[rbid],one,rpe);  
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu forward ops after bid=%d rpe=%d ireceived=%ld\n",mype,ctxt,bid,rpe,ireceived);fflush(stdout);
#endif        
      int channel=1;
      habu_operations(&rb[0],ireceived,channel, ctxt);
      break;
    }
  //recvedgroup[ctxt*16] = (recvedgroup[ctxt*16]+1)%ngroups; // Continuous loop on pes.
  }

//   for(int icount=0;icount<(maxctxt);icount+=1){
  for(int rms=0;rms<maxctxt*npes_my_group;rms+=1){
//     int rpe = npes+recvedtid[ctxt*16];
    // Check a few buffs to see if any are full.
    
    int bid = ngroups + rms + ctxt*(ngroups+npes_my_group*maxctxt);
    
    
    int64_t ireceived = recvheader[bid];
    tr += ireceived;
//       printf("%d %d habu checking thread process_requests %d %d ireceived=%ld\n",mype,ctxt,bid,rpe,ireceived);
    if(ireceived>0){
      int rctxt = rms%maxctxt;
      int rbid = ngroups + mymid*maxctxt+ctxt + rctxt*(ngroups+npes_my_group*maxctxt);
      int rpe = groupleader + rms/maxctxt;
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu process_requests after bid=%d mype=%d ireceived=%ld\n",mype,ctxt,bid,mype,ireceived);fflush(stdout);
#endif        
      // We got something!  Process the updates.
      recvheader[bid]=0L;  //  Mark the buff as empty
      unchar rb[ireceived];
      memcpy(rb,&recvbuff[buffsize*bid],ireceived);
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu sending intragroup recvready to rbid=%d %d after ireceived=%ld!\n", mype,ctxt,rbid,rpe,ireceived);fflush(stdout);
#endif        
//       recvready[rbid]=one;  //  Signal the sending PE we are ready to receive more
       shmem_put64(&recvready[rbid],&one,1,rpe);  
//      shmem_int64_atomic_add(&recvready[rbid],one,rpe); 
//       shmemx_thread_quiet();
      

#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu process ops after bid=%d mype=%d ireceived=%ld\n",mype,ctxt,bid,mype,ireceived);fflush(stdout);
#endif        
      // Exectute the operations inside the buffs!
      int channel=0;
      habu_operations(&rb[0],ireceived,channel,ctxt);
    }

//   recvedtid[ctxt*16] = (recvedtid[ctxt*16]+1)%(maxctxt*npes_my_group); // Continuous loop on threads.
  }
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
//   printf("%d %d habu exiting process_requests tr=%ld\n",mype,ctxt,tr);
#endif        
  
  return tr;
}


void send_buff(int bid,int ctxt){
  //  This routine is responsible for sending a buff.  The buff might be completly or
  //  partially full but if it has any items in it then it will be sent.
  
  int targetpe;  
  int64_t tr;
  
  int rbid;
  int64_t ibc = buffcount[bid];  //  Make a copy of the buff count, necessary for recursion.
  if(ibc==0)  return;  //  nothing to do so return.
  buffcount[bid] = 0;  //  Reset the buff count immediately.  
  int lbid = bid - ctxt*(ngroups + npes_my_group*maxctxt);
  if(lbid>=ngroups){
    //  If tgroup is >= ngroups, then that means we are in the intragroup phase and 
    //  we should be targeting members and slices.
    int msid = lbid - ngroups;
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d habu intragroup bid %d ngroups %d msid %d \n",mype,ctxt,bid,ngroups,msid);fflush(stdout);
#endif  
    int tgroup = mygid;
    int tmid = msid/maxctxt;
    int tctxt = msid%maxctxt;
    
    targetpe = mygid*npes_my_group+tmid;
    
    rbid = buffid(ngroups,mymid,ctxt,tctxt);
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d habu wants to send intragroup bid %d buff to rbid %d tpe %d ibc=%ld tctxt=%d recvready=%ld\n",mype,ctxt,bid,rbid,targetpe,ibc,tctxt,recvready[bid]);fflush(stdout);
#endif  

  }else{
    //  We are in the inter group phase.
    //  The bid contains the target group and we should be targeting the same memberslice in that group.
    int tgroup = lbid;
    targetpe = tgroup*npes_my_group+mymid;
    rbid = buffid(mygid,0,0,ctxt);
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d habu wants to send intergroup bid %d buff to rbid %d tpe %d ibc=%ld recvready=%ld\n",mype,ctxt,bid,rbid,targetpe,ibc,recvready[bid]);fflush(stdout);
#endif  
    
  }
  

#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
//   printf("%d %d habu wants to send bid %d buff to rbid %d %d ibc=%ld recvready=%ld\n",mype,ctxt,bid,rbid,targetpe,ibc,recvready[bid]);fflush(stdout);
#endif  
  //  Habu wants to send this buff, but we might need to go and process other
  //  buffs, which might wrap around to this same buff.
  //  We need to make a copy of what we are sending onto the stack because of recursion.
  unchar tb[ibc];
  if(recvready[bid] == 0 ){
    //  We are about to call process_requests which might cause recursion!!!!!
    //  We need to save a copy of the buff to the stack for recursion!!!!!
    memcpy(&tb, &sendbuff[buffsize*bid],ibc);

    // check to see if the target pe is ready to receive this buff.
    while( (*(volatile int64_t *)&recvready[bid]) == 0 ){
      #ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu in sendbuff waiting for recvready bid=%d \n",mype,ctxt,bid);fflush(stdout);
      #endif  
//         processcalls[ctxt]++;
      // The targetpe is not ready to receive.  Not ready to send. Process requests instead.
      tr = process_requests(ctxt);
    }
    uint64_t sig = ibc;
    //      Set the recvready flag to ZERO, I.E. it is not ready to recv another buff.
    #ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu zeroing recvready bid=%d!\n", mype,ctxtS,bid);fflush(stdout);
      printf("%d %d habu sending buff to rbid=%d %d ibc=%ld recvready=%ld\n",mype,ctxt,rbid,targetpe,ibc,recvready[bid]);fflush(stdout);
    #endif        
    recvready[bid]=0;
    //  Send data and the recvheader signal in one call.
    shmemx_putmem_signal(&recvbuff[0+buffsize*rbid],&tb[0],ibc,&recvheader[rbid],sig,targetpe);
  }else{
    // No recursion possible, don't do the copy,now send the buff
    uint64_t sig = ibc;
    //      Set the recvready flag to ZERO, I.E. it is not ready to recv another buff.
    #ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
      printf("%d %d habu zeroing recvready bid=%d!\n", mype,ctxtS,bid);fflush(stdout);
      printf("%d %d habu sending buff to rbid=%d %d ibc=%ld recvready=%ld\n",mype,ctxt,rbid,targetpe,ibc,recvready[bid]);fflush(stdout);
    #endif        
    recvready[bid]=0;
    //  Send data and the recvheader signal in one call.
    shmemx_putmem_signal(&recvbuff[0+buffsize*rbid], &sendbuff[0+buffsize*bid],ibc,&recvheader[rbid],sig,targetpe);
  }

  tr = process_requests(ctxt);  //  Seems like a good idea to call process_requests
  return;
}




int64_t flush_buffs(int ctxt){
//  We need to flush all of the buffs, probably because we are in the endgame.
//  We just try and send everything and see if there are any outstanding sends.  
  int64_t nfull=0;
//   printf("%d %d habu in flush!\n", mype,ctxt);fflush(stdout);
  
  for(int i=0;i<(ngroups+npes_my_group*maxctxt);i+=1){
    int bid = (i+mygid)%(ngroups+npes_my_group*maxctxt)+ctxt*(ngroups + npes_my_group*maxctxt);
    int pf=0;
    
//     int bid = buffid(ipe,ctxt);
    if(buffcount[bid]>0){
      send_buff(bid,ctxt);
      nfull +=1;
    }
    //  Calling process_requests npes number of times guarantees that all buffs will have been checked.
//     printf("%d %d habu in flush calling process_requests !\n", mype,ctxt);fflush(stdout);
    nfull+= process_requests(ctxt); 
    if(recvready[bid]==0){ 
//       printf("%d %d habu in flush recvready bid=%d eq 0!\n", mype,ctxt,bid);fflush(stdout);
      nfull+=1; }
  }
  shmemx_thread_quiet(); //  This is to make sure all of the signals are out before sending out the done signals.
  //  Return a count of anything outstanding.  If nfull==0 the calling routine knows all buffs have been flushed
//   printf("%d %d habu exiting flush nfull=%ld!\n", mype,ctxt,nfull);fflush(stdout);
  return nfull;
}

int64_t habu_propel(int ctxt){ return (flush_buffs(ctxt));}


void habu_barrier(int ctxt){
//  This barrier is to make sure all habu references have completed at the end of a epoch.  
  int num_active_zeros = 0;
  int64_t last_active_buffs = 1;
  int64_t group_active_buffs;
  int64_t val;
  int64_t tfb = 1;
  double barwait, clearsig;
  double bt = get_time();
// #define HABU_DEBUG_BARRIER  
#ifdef HABU_DEBUG_BARRIER
  printf("%d %d %d %d in habu_barrier!\n",mype, ctxt, npes, num_active_ctxt);fflush(stdout);
#endif  
  while(num_active_zeros<1){
    int64_t fb = 1;
    double fbt=get_time();
    while(fb>0){
      fb = flush_buffs(ctxt);
      tfb += fb;
    }
    double barsig=get_time();
    total_bartime[ctxt*8+1] += barsig-fbt;
//     printf("%d %d flushed %ld habu buffs!\n",mype, ctxt,tfb);fflush(stdout);
#ifdef HABU_DEBUG_BARRIER
    printf("%d %d counting active_buffs %ld habu buffs!\n",mype, ctxt,tfb);fflush(stdout);
#endif    
    //  This tells everyone else how many active_buffs this pe has.  
    //  If everyone has no active_buffs we are done.
    //  If anyone has >0 active_buffs everyone needs another iteration.   
    
    
    
    //  Within group phase
    for(int j=0;j<num_active_ctxt;j+=1){
      for(int i=0;i<npes_my_group;i+=1){
//         int ipe = ranpe[i];
//         int ipe = (i+mype)%npes;
        int ipe = groupleader + i;
        shmem_long_add(&active_buffs[j*16+ticktock[ctxt*16]],tfb,ipe);
    }} 
    tfb = 0;  //  reset tfb and start counting again
    shmemx_thread_quiet();  //  This is to make sure all of the active_buffs are out before sending out the done signals.
    //  Tell everyone else we have flushed our buffs
    for(int j=0;j<num_active_ctxt;j+=1){
      for(int i=0;i<npes_my_group;i+=1){
//         int ipe = ranpe[i];
//         int ipe = (i+mype)%npes;
        int ipe = groupleader + i;
        shmem_int_add(&done[j*4+ticktock[ctxt*16]],1,ipe);
    }}

    
    barwait=get_time();
    total_bartime[ctxt*8+2] += barwait - barsig;

    //  When this counts up to npes*num_active_ctxt everyone has flushed their buffs.
    //  Use habu_propel here to help test habu_propel, but we should be careful about the definition of habu_propel.
    while(done[ctxt*4+ticktock[ctxt*16]]<npes_my_group*num_active_ctxt){ tfb += habu_propel(ctxt); }  //  All contexts wait here.
    clearsig=get_time();
    total_bartime[ctxt*8+3] += clearsig - barwait;
    
    shmem_int_add(&done[ctxt*4+ticktock[ctxt*16]],-(npes_my_group*num_active_ctxt),mype);  // Reset done.

    group_active_buffs = shmem_long_swap(&active_buffs[16*ctxt+ticktock[ctxt*16]],0,mype);  // Copy and reset active_buffs.
    shmemx_thread_quiet();

    
#ifdef HABU_DEBUG_BARRIER
    printf("%d %d completed group phase barrier %ld !\n",mype, ctxt,group_active_buffs);fflush(stdout);
#endif    


    //  Cross group phase
    int ngsigs = (npes+npes_my_group-1)/npes_my_group;
    
    for(int ipe=(mype%npes_my_group);ipe<npes;ipe+=npes_my_group){
//     for(int i=0;i<ngsigs;i+=1){
//       int ig = (i+mype+ctxt)%ngsigs;
//       int ipe = mype%npes_my_group + ig*npes_my_group;
//       if(ipe>=npes) continue;
      shmem_long_add(&active_buffs[ctxt*16+2+ticktock[ctxt*16]],group_active_buffs,ipe);
    }
    shmemx_thread_quiet();  //  This is to make sure all of the active_buffs are out before sending out the done signals.
    //  Tell everyone else we have flushed our buffs
    for(int ipe=(mype%npes_my_group);ipe<npes;ipe+=npes_my_group){
//     for(int i=0;i<ngsigs;i+=1){
//       int ig = (i+mype+ctxt)%ngsigs;
//       int ipe = mype%npes_my_group + ig*npes_my_group;
//       if(ipe>=npes) continue;
      shmem_int_add(&done[ctxt*4+2+ticktock[ctxt*16]],npes_my_group*num_active_ctxt,ipe);
    }

    
    barwait=get_time();
    total_bartime[ctxt*8+2] += barwait - clearsig;

    //  When this counts up to npes*num_active_ctxt everyone has flushed their buffs.
    //  Use habu_propel here to help test habu_propel, but we should be careful about the definition of habu_propel.
    while(done[ctxt*4+2+ticktock[ctxt*16]]<npes*num_active_ctxt){ tfb += habu_propel(ctxt); }  //  All contexts wait here.
    clearsig=get_time();
    total_bartime[ctxt*8+3] += clearsig - barwait;
    
    shmem_int_add(&done[ctxt*4+2+ticktock[ctxt*16]],-(npes*num_active_ctxt),mype);  // Reset done.

    last_active_buffs = shmem_long_swap(&active_buffs[16*ctxt+2+ticktock[ctxt*16]],0,mype);  // Copy and reset active_buffs.
    shmemx_thread_quiet();
    
    
    
    
    
#ifdef HABU_DEBUG_BARRIER
    printf("%d %d in barrier last_active_buffs= %ld \n",mype, ctxt,last_active_buffs);fflush(stdout);
#endif    
    ticktock[ctxt*16] = (ticktock[ctxt*16]+1)%2;
    if(last_active_buffs==0){
      num_active_zeros++;
    }else{
      num_active_zeros=0;
    }
    total_bartime[ctxt*8+4] += get_time()-clearsig ;
      
  }
  total_bartime[ctxt*8+0] += get_time() - bt;
//     if(mype==0 && ctxt==0) habu_stats(ctxt);

//   }
}

  
void habu_op(habu_mem_t thandle,int64_t a_offset,int a_ipe,habu_op_t opcode, void *item,int ctxt){
//  This is the main generic habu routine that takes in a memory handle, 
//  target pe and index, opcode, item payload , and target location,
//  forms the header, buffs up the values and the hearders, checks to see if the 
//  buff is full and sends it off if it is full.  
  
  int ipe = a_ipe;
  int64_t offset = a_offset;
  int bid,target;
  int tgroup=-1;
  
  
  if(ipe==HABU_CYCLIC_DISTRIBUTION_PE){
    //  We have a word cyclic distribution and the offset passed in was a global offset.
    ipe  = a_offset % npes;
    offset = a_offset / npes;
//     printf("%d distribution is WC %d %ld %d %d %ld\n",mype,idistribution[thandle],a_offset,npes,ipe,offset);
  }
  int sliceid = offset*(num_active_ctxt*inv_numelems[thandle]);;
//  If inv_numelems is negative, atomicity is off.  Just set sliceid to this ctxt
  sliceid = sliceid<0.?ctxt:sliceid;   

  int tmid = ipe%npes_my_group;
  if((tmid != mymid) || (sliceid!=ctxt)){
    //  If the target member and slice don't match, then we are in the intragroup phase.
    //  Shift groupid to ngroups and use targte memberid and sliceid to determine target buffid.
    bid = buffid(ngroups,tmid,sliceid,ctxt);
  }else{
    //  We are in the intergroup phase. Determine the target group for the groupid.
    //  memberid and sliceid should be zero.
    tgroup = ipe/npes_my_group;
    bid = buffid(tgroup,0,0,ctxt);
  }
    int64_t ioff = offset ;
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS  
    printf("%d %d checking send_buff to prevent overflow ipe=%d tgroup=%d bid=%d ioff=%ld ibc=%ld, pls=%d  inv=%lf\n", mype,ctxt,ipe,tgroup,bid,ioff,buffcount[bid], habu_payload_size[opcode],inv_numelems[thandle]);fflush(stdout);
#endif
    
    
  // We check the buff to see if it is full right away.
  // Habu can be called recursively so the very act of calling send_buff could result
  // in this routine being called again.  We therefore need to do a while loop on the 
  // buff count because send_buff might both empty the buff and refill it!
  while(buffcount[bid] +size_habu_op_t+size_habu_mem_t+size_int64_t+size_habu_ipe_t+ habu_payload_size[opcode] >= buffsize){
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d calling send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,ioff,buffcount[bid]);fflush(stdout);
#endif
    //  This buff is about to overflow, send the buff instead.
    send_buff(bid,ctxt);
  }

  //double mt = get_time();
  //  We are now sure that there is space in sendbuff to place the item.
  //  First set the header information.
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &opcode, size_habu_op_t);buffcount[bid]+=size_habu_op_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &thandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ioff, size_int64_t); buffcount[bid]+=size_int64_t;
  if((bid-buffid(0,0,0,ctxt))>=ngroups){memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ipe, size_habu_ipe_t); buffcount[bid]+=size_habu_ipe_t;}
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("%d %d buffed ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,ioff,buffcount[bid]);fflush(stdout);
#endif  
  
  //  The payload sizes depends on the opcode.
  int ps = habu_payload_size[opcode];
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &item[0],ps);
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    int64_t *it = item;
    printf("%d %d buffed bid=%d ioff=%ld val=%ld ibc=%ld\n", mype,ctxt,bid,ioff,it[0],buffcount[bid]);fflush(stdout);
#endif  
  buffcount[bid]+=ps;
  //memcpytime[ctxt*8] += get_time()-mt;
  return;
}



void habu_fop(habu_mem_t rethandle,int64_t localoff,int rpe,habu_mem_t thandle,int64_t a_offset,int a_ipe,habu_op_t opcode, void *value,int ctxt){
//  This is the the second generic habu routine that takes in a memory handle, 
//  target pe and index, opcode, item payload , and target location,
//  forms the header, buffs up the values and the hearders, checks to see if the 
//  buff is full and sends it off if it is full.  
//  This routine implements fetching versions of operations.  
//  Fetching operations return some kind of data, although the return PE does not have to
//  be the same as the source PE.
  
  int ipe = a_ipe;
  int64_t offset = a_offset;
  int64_t *ivalue = value;
  int bid,target;
  int tgroup=-1;
  
  if(ipe==HABU_CDPE){
    //  We have a word cyclic distribution and the offset passed in was a global offset.
    ipe  = a_offset % npes;
    offset = a_offset / npes;
//     printf("%d distribution is WC %d %ld %d %d %ld\n",mype,idistribution[thandle],a_offset,npes,ipe,offset);
  }
  int sliceid = offset*(num_active_ctxt*inv_numelems[thandle]);;
//  If inv_numelems is negative, atomicity is off.  Just set sliceid to this ctxt
  sliceid = sliceid<0.?ctxt:sliceid;   

  int tmid = ipe%npes_my_group;
  if((tmid != mymid) || (sliceid!=ctxt)){
    //  If the target member and slice don't match, then we are in the intragroup phase.
    //  Shift groupid to ngroups and use targte memberid and sliceid to determine target buffid.
    bid = buffid(ngroups,tmid,sliceid,ctxt);
  }else{
    //  We are in the intergroup phase. Determine the target group for the groupid.
    //  memberid and sliceid should be zero.
    tgroup = ipe/npes_my_group;
    bid = buffid(tgroup,0,0,ctxt);
  }
    int64_t ioff = offset ;
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS  
    printf("%d %d checking send_buff to prevent overflow ipe=%d tgroup=%d bid=%d ioff=%ld ibc=%ld, pls=%d  inv=%lf\n", mype,ctxt,ipe,tgroup,bid,ioff,buffcount[bid], habu_payload_size[opcode],inv_numelems[thandle]);fflush(stdout);
#endif

  int64_t ibc =buffcount[bid];
  
  
  //  We first check to see if the buff will overflow if we add this item.
  //  If we are going to overflow we instead pause to send the buff instead.
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d checking send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld, pls=%d \n", mype,ctxt,ipe,bid,offset,ibc, habu_payload_size[opcode]);fflush(stdout);
#endif
  while(buffcount[bid] +size_habu_op_t+size_habu_mem_t+sizeof(offset)+size_habu_ipe_t+sizeof(rpe)+size_habu_mem_t+sizeof(localoff)  + habu_payload_size[opcode] >= buffsize){
    //  With recursive HABU, it is possible to not only send the buff but also refill it before 
    //  returning from send_buff, we therefore cannot assume the buff is empty when it returns
    //  from send_buff or that it even has enought space for this item.  A while loop is 
    //  necessary instead.
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d calling send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,offset,ibc);fflush(stdout);
#endif
    //  This buff is about to overflow, send the buff instead.
    send_buff(bid,ctxt);
  }
  
  //  Now that we know the buff has enough space to place this item, we need to place the
  //  item into the buff.
  //  First thing is to form the item header and place it at the beginning of the packet.
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &opcode, size_habu_op_t);buffcount[bid]+=size_habu_op_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &thandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &offset, sizeof(offset));buffcount[bid]+=sizeof(offset);
  if((bid-buffid(0,0,0,ctxt))>=ngroups){memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ipe, size_habu_ipe_t); buffcount[bid]+=size_habu_ipe_t;}

#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("%d %d buffed ipe=%d bid=%d ioff=%ld ibc=%ld opcode=%u\n", mype,ctxt,ipe,bid,offset,ibc,opcode);fflush(stdout);
#endif  
  if(rethandle>=0){
    //  This is a fetching operation which means that we need to form the reture header now
    //  so the target PE will know where to return the fetched data.
    memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &rpe,sizeof(rpe));buffcount[bid]+=sizeof(rpe);
    memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &rethandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
    memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &localoff,sizeof(localoff)); buffcount[bid]+=sizeof(localoff);
  #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d buffed returnheader rpe=%d reth=%u lo=%ld bid=%d ioff=%ld \n", mype,ctxt,rpe,rethandle,localoff,bid,offset);fflush(stdout);
  #endif  
  }

  int ps = habu_payload_size[opcode];
  if(ps>0 && ((ivalue==NULL)||(ivalue==0))){
    printf("ERROR:  Payload size >0 but pointer is NULL!  Returning to caller\n");
    return;
  }
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d buffed bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,bid,offset,ibc);fflush(stdout);
#endif  
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ivalue[0],ps);buffcount[bid]+=ps;
  return;
}

void habu_putv(habu_mem_t thandle,int64_t index,int a_ipe, void *item, int64_t num_items, int ctxt){
//  This is the putv habu routine that takes in a memory handle, 
//  target pe and index, opcode, item payload , and target location,
//  forms the header, buffs up an contiguous set of values values, checks to see if the 
//  buff is full and sends it off if it is full.  
  
  int ipe = a_ipe;
  int64_t offset = index;
  habu_op_t opcode = HABU_PUTV;
  int ps = item_size[thandle];
  int bid,target;
  int tgroup=-1;
  
  
  if(ipe==HABU_CDPE){
    //  We have a word cyclic distribution and the offset passed in was a global offset.
    ipe  = index % npes;
    offset = index / npes;
//     printf("%d distribution is WC %d %ld %d %d %ld\n",mype,idistribution[thandle],index,npes,ipe,offset);
  }
  int sliceid = offset*(num_active_ctxt*inv_numelems[thandle]);;
//  If inv_numelems is negative, atomicity is off.  Just set sliceid to this ctxt
  sliceid = sliceid<0.?ctxt:sliceid;  

  int tmid = ipe%npes_my_group;
  if((tmid != mymid) || (sliceid!=ctxt)){
    //  If the target member and slice don't match, then we are in the intragroup phase.
    //  Shift groupid to ngroups and use targte memberid and sliceid to determine target buffid.
    bid = buffid(ngroups,tmid,sliceid,ctxt);
  }else{
    //  We are in the intergroup phase. Determine the target group for the groupid.
    //  memberid and sliceid should be zero.
    tgroup = ipe/npes_my_group;
    bid = buffid(tgroup,0,0,ctxt);
  }
    int64_t ioff = offset ;
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS  
  printf("%d %d checking send_buff to prevent overflow ipe=%d tgroup=%d bid=%d ioff=%ld ibc=%ld, pls=%d  inv=%lf\n", mype,ctxt,ipe,tgroup,bid,ioff,buffcount[bid], habu_payload_size[opcode],inv_numelems[thandle]);fflush(stdout);
#endif

  int item_num = 0;
//  int max_items = ((buffsize-(size_habu_op_t+size_habu_mem_t+size_int64_t+size_habu_ipe_t   + sizeof(int)))/item_size[thandle])/4;
  int max_items = 64;
  for(item_num=0;item_num< num_items;item_num+=max_items){
//   while(item_num<num_items){
    int64_t ibc =buffcount[bid];
    int64_t ioff = offset + item_num;
    int nif =(num_items-item_num)<max_items?(num_items-item_num):max_items  ;
    int num_bytes = nif*ps;
    
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d putv checking send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld, pls=%d \n", mype,ctxt,ipe,bid,ioff,ibc, ps);fflush(stdout);
#endif
    
    
  // We check the buff to see if it is full right away.
  // Habu can be called recursively so the very act of calling send_buff could result
  // in this routine being called again.  We therefore need to do a while loop on the 
  // buff count because send_buff might both empty the buff and refill it!
  while(buffcount[bid] +size_habu_op_t+size_habu_mem_t+sizeof(ioff)+size_habu_ipe_t  + sizeof(nif) + num_bytes >= buffsize){
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d putv calling send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,ioff,ibc);fflush(stdout);
#endif
    //  This buff is about to overflow, send the buff instead.
    send_buff(bid,ctxt);
  }
  
  //  We are now sure that there is space in sendbuff to place the item.
  //  First set the header information.
  ibc = buffcount[bid];
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("%d %d putv buffering ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,ioff,ibc);fflush(stdout);
#endif  
//   double mt = get_time();
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &opcode, size_habu_op_t);buffcount[bid]+=size_habu_op_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &thandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ioff, sizeof(ioff)); buffcount[bid]+=sizeof(ioff);
  if((bid-buffid(0,0,0,ctxt))>=ngroups){memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ipe, size_habu_ipe_t); buffcount[bid]+=size_habu_ipe_t;}

  
  //  Need to also send how many items are in this batch.
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &nif,sizeof(nif));buffcount[bid]+=sizeof(nif);
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &item[item_num*ps],num_bytes);
  buffcount[bid]+=num_bytes;
//   memcpytime[ctxt*8] += get_time()-mt;
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    int64_t *it = item;
    printf("%d %d putv buffed bid=%d ioff=%ld val=%ld ibc=%ld\n", mype,ctxt,bid,ioff,it[item_num],buffcount[bid]);fflush(stdout);
#endif  
  }
  return;
}

void habu_getv(habu_mem_t rethandle,int64_t retoffset,int retpe,habu_mem_t thandle,int64_t index,int target_pe,int64_t num_items, int ctxt){
//  This is the getv habu routine that can be thouht of the fetching version of putv.
  
  int ipe = target_pe;
  int64_t offset = index;
  habu_op_t opcode = HABU_GETV;
  
  if(ipe==HABU_CDPE){
    //  We have a word cyclic distribution and the offset passed in was a global offset.
    ipe  = index % npes;
    offset = index / npes;
//     printf("%d distribution is WC %d %ld %d %d %ld\n",mype,idistribution[thandle],index,npes,ipe,offset);
  }

  int bid,target;
  int tgroup=-1;
  
  int sliceid = offset*(num_active_ctxt*inv_numelems[thandle]);;
//  If inv_numelems is negative, atomicity is off.  Just set sliceid to this ctxt
  sliceid = sliceid<0.?ctxt:sliceid;   

  int tmid = ipe%npes_my_group;
  if((tmid != mymid) || (sliceid!=ctxt)){
    //  If the target member and slice don't match, then we are in the intragroup phase.
    //  Shift groupid to ngroups and use targte memberid and sliceid to determine target buffid.
    bid = buffid(ngroups,tmid,sliceid,ctxt);
  }else{
    //  We are in the intergroup phase. Determine the target group for the groupid.
    //  memberid and sliceid should be zero.
    tgroup = ipe/npes_my_group;
    bid = buffid(tgroup,0,0,ctxt);
  }
    int64_t ioff = offset ;
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS  
    printf("%d %d checking send_buff to prevent overflow ipe=%d tgroup=%d bid=%d ioff=%ld ibc=%ld, pls=%d  inv=%lf\n", mype,ctxt,ipe,tgroup,bid,ioff,buffcount[bid], habu_payload_size[opcode],inv_numelems[thandle]);fflush(stdout);
#endif

  //  We first check to see if the buff will overflow if we add this item.
  //  If we are going to overflow we instead pause to send the buff instead.
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d getv checking send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld, num_items=%ld pls=%d \n", mype,ctxt,ipe,bid,offset,buffcount[bid], num_items,item_size[thandle]);fflush(stdout);
#endif
  while(buffcount[bid] +size_habu_op_t+size_habu_mem_t+sizeof(offset)+size_habu_ipe_t+sizeof(retpe)+size_habu_mem_t+sizeof(retoffset)  + sizeof(num_items) >= buffsize){
    //  With recursive HABU, it is possible to not only send the buff but also refill it before 
    //  returning from send_buff, we therefore cannot assume the buff is empty when it returns
    //  from send_buff or that it even has enought space for this item.  A while loop is 
    //  necessary instead.
#ifdef HABU_DEBUG_PRINT_TRACK_BUCKETS
    printf("%d %d calling send_buff to prevent overflow ipe=%d bid=%d ioff=%ld ibc=%ld\n", mype,ctxt,ipe,bid,offset,buffcount[bid]);fflush(stdout);
#endif
    //  This buff is about to overflow, send the buff instead.
    send_buff(bid,ctxt);
  }
  
  //  Now that we know the buff has enough space to place this item, we need to place the
  //  item into the buff.
  //  First thing is to form the item header and place it at the beginning of the packet.
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &opcode, size_habu_op_t);buffcount[bid]+=size_habu_op_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &thandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &offset, sizeof(offset));buffcount[bid]+=sizeof(offset);
  if((bid-buffid(0,0,0,ctxt))>=ngroups){memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &ipe, size_habu_ipe_t); buffcount[bid]+=size_habu_ipe_t;}

#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("%d %d buffed ipe=%d bid=%d ioff=%ld ibc=%ld opcode=%u\n", mype,ctxt,ipe,bid,offset,buffcount[bid],opcode);fflush(stdout);
#endif  
  //  This is a fetching operation which means that we need to form the reture header now
  //  so the target PE will know where to return the fetched data.
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &retpe,sizeof(retpe));buffcount[bid]+=sizeof(retpe);
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &rethandle,size_habu_mem_t);buffcount[bid]+=size_habu_mem_t;
  memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &retoffset,sizeof(retoffset)); buffcount[bid]+=sizeof(retoffset);
    //  Place the number of items that needs to be returned into the buffer.
 
#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
  printf("%d %d buffed returnheader retpe=%d reth=%u ro=%ld bid=%d ioff=%ld \n", mype,ctxt,retpe,rethandle,retoffset,bid,offset);fflush(stdout);
#endif  

#ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
    printf("%d %d buffed bid=%d ioff=%ld  ibc=%ld\n", mype,ctxt,bid,offset,buffcount[bid]);fflush(stdout);
#endif 
    memcpy(&sendbuff[buffcount[bid]+buffsize*bid], &num_items,sizeof(num_items));buffcount[bid]+=sizeof(num_items);
  return;
}
