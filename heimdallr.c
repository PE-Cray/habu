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

//  Heimdallr65 is a driver that implements a number of tests that perform
//  random "updates" to a global memory space.  Heimdallr is can be used for both
//  performance and functional testing.  Tests often comes in pairs, once implemented 
//  using shmem atomics, and once using the HABU library.  When possible, the results
//  of the two tests are compared for correctness.  In many (most) cases we expect zero errors,
//  but in some cases we do expect a very small number of errors. 

//  One can look at only the shmem performance, or both shmem and HABU performance.


#include <stdio.h>
#include <string.h>
#include <time.h>
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
#include <assert.h>
#include "habu.h"


/* Package Information */
#define PACKAGE_NAME "Heimdallr65"
#define PACKAGE_VERSION "2.0"

/* Macros for timing */
struct tms tt;
#define WSEC() (times(&tt) / (double)sysconf(_SC_CLK_TCK))
#define CPUSEC() (clock() / (double)CLOCKS_PER_SEC)


#define GIBI     1073741824L
#define MIBI     1048576L

#define OPT_HELP 1
//#define MAXBLKSIZE 2048
#define MAXBLKSIZE 4096
//#define MAXBLKSIZE 610
// #define MAXBLKSIZE 61
// #define nrepeats 4

#define THREAD_HOT

int64_t ONE=1;
int64_t NEGONE=-1;


typedef struct opts_t {
  int check;
} opts_t;

//  Define a local structure to be used in the HABU user defined function
struct hrp_local{
  habu_op_t HABU_RP;
  int64_t tabsize;
  int64_t npes;
};

void hrp(habu_mem_t thandle, int64_t ioffset,void *arg_v,void *largs,int ctxt){
  int64_t *array = habu_handle_to_pointer(thandle);
  int64_t *val = arg_v;
  int ipe=shmem_my_pe();
  struct hrp_local *hrpl = largs;
//   printf("mype= %d %d hrp ioff= %ld array=%ld val = %ld %ld\n", ipe,omp_get_thread_num(), ioffset,array[ioffset],val[0],val[1]);fflush(stdout);
//   printf("HABU_RP %ld %ld\n",HABU_RP,hrpl->HABU_RP);
  

  if(array[ioffset]==val[0]){
    array[ioffset]=val[1];
  }else if(array[ioffset]==val[1]){
    ipe = (ipe+919)%hrpl->npes;
    ioffset = (ioffset+907)&(hrpl->tabsize-1);
    habu_op( thandle,ioffset,ipe, hrpl->HABU_RP,val,ctxt); 
  }else{
    //  This should only happen if there is an error in our test
    array[ioffset]=-val[1];
  }
}
void fhrp(habu_mem_t rhandle,int64_t return_index, int return_pe, habu_mem_t thandle, int64_t ioffset,void *arg_v,void *largs,int ctxt){
  int64_t *array = habu_handle_to_pointer(thandle);
  int64_t *val = arg_v;
  int ipe=shmem_my_pe();
  struct hrp_local *hfrpl = largs;
//   printf("mype= %d %d hrp ioff= %ld array=%ld val = %ld %ld\n", ipe,omp_get_thread_num(), ioffset,array[ioffset],val[0],val[1]);fflush(stdout);
//   printf("HABU_RP %ld %ld\n",HABU_RP,hrpl->HABU_RP);
  

  if(array[ioffset]==val[0]){
    array[ioffset]=val[1];
    habu_op( rhandle,return_index,return_pe, HABU_PUT,&ioffset,ctxt); 

  }else if(array[ioffset]==val[1]){
    ipe = (ipe+919)%hfrpl->npes;
    ioffset = (ioffset+907)&(hfrpl->tabsize-1);
    habu_fop(rhandle,return_index,return_pe, thandle,ioffset,ipe, hfrpl->HABU_RP,val,ctxt); 
  }else{
    //  This should only happen if there is an error in our test
    array[ioffset]=-val[1];
  }
}

typedef struct
{
  size_t   nrepeats;
  size_t   l2tabsize;
  size_t   l2nupdates;
  size_t   minwords;
  size_t   maxwords;
  size_t   ranpelist;
  int      msgrowth;
  char     singletest[14];
} options_t;

#define numtests 33
char tnames[numtests][14];
int nbytes[numtests];
double nups[numtests];
int skiptest[numtests];
int END_SW_ITEST=0;
int END_CORREST_TEST=0;

  
options_t opts;

void print_usage(char ***argv, FILE *file)
{
  char *name = strrchr((*argv)[0], '/');
  name = (name == NULL ? (*argv)[0] : name+1);
  int mype = 0;
  mype = shmem_my_pe();
  if (mype == 0)
    printf(
            "Usage: %s [OPTION..]\n"
            "Options:\n"
            "  -t, \t\t Table size (as log base 2) per PE\n"
            "  -n, \t\t Nupdates   (as log base 2) per PE\n"
            "  -r, \t\t Number of Repeats completed within a timed test\n"
            "  -T, \t\t Name of single test to be executed\n"
            "  -w, \t\t Min Message size (in 8B words)\n"
            "  -W, \t\t Max Message size (in 8B words)\n"
            "  -G, \t\t Message size growth factor (integer)\n"
            "  -h, \t\t Display this information\n",
            name);
}

options_t check_args(int *argc, char***argv)
{
  uint64_t seed = 0;
  seed = time(NULL);

  options_t opts =
  {
    .nrepeats  = (size_t)4,
    .l2tabsize = (size_t)25,
    .l2nupdates = (size_t)23,
    .maxwords  = (size_t)1 << 13,
    .minwords  = (size_t)1 ,
    .msgrowth  = (int)1,
  };

  strncpy(opts.singletest, "ALL",14);  
  int i=0;
  strncpy(tnames[i], "ATOMIC_NA_INC",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "HABU_NA_INC",14);   nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "ATOMIC_NA_ADD",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "HABU_NA_ADD",14);   nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "ATOMIC_INC",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_INC",14);      nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "ATOMIC_ADD",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_ADD",14);      nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "AMO_PEADD",14);     nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "HABU_PEADD",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=1; i++;
  strncpy(tnames[i], "ATOMIC_ADD2",14);   nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_I&A",14);      nups[i]=2; nbytes[i]= 16; skiptest[i]=0; i++;
  strncpy(tnames[i], "ATOMIC_FADD",14);   nups[i]=1; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_FADD",14);     nups[i]=1; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "ATOMIC_RP",14);       nups[i]=opts.nrepeats*(opts.nrepeats+1.0)/2.0/(opts.nrepeats*2.0)/opts.nrepeats; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_RP",14);       nups[i]=opts.nrepeats*(opts.nrepeats+1.0)/2.0/(opts.nrepeats*2.0)/opts.nrepeats; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "ATOMIC_FRP",14);       nups[i]=opts.nrepeats*(opts.nrepeats+1.0)/2.0/(opts.nrepeats*2.0)/opts.nrepeats; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_FRP",14);       nups[i]=opts.nrepeats*(opts.nrepeats+1.0)/2.0/(opts.nrepeats*2.0)/opts.nrepeats; nbytes[i]= 2*8;skiptest[i]=0; i++;
  strncpy(tnames[i], "GET_NB_lsheap",14); nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_GET",14);      nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; END_SW_ITEST=i; i++;
  strncpy(tnames[i], "GET_NB_lsheap",14); nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_GETV",14);     nups[i]=1; nbytes[i]= 8;  skiptest[i]=1;END_CORREST_TEST=i;  i++;
  strncpy(tnames[i], "PUT_lsheap",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "HABU_PUTV",14);     nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "GET_NB_lheap",14);  nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "GET_NB_lstack",14); nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "GET_lsheap",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_NB_lsheap",14); nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_NB_lheap",14);  nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_NB_lstack",14); nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_SIG",14);       nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_SIG_NB",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  strncpy(tnames[i], "PUT_manSIG",14);    nups[i]=1; nbytes[i]= 8;  skiptest[i]=0; i++;
  
  opterr = 0;
  int opt;
  int firstst=0;
  while ((opt = getopt(*argc, *argv, "hr:T:t:w:W:Lr:G:n:")) != -1)
  {
    switch (opt)
    {
    case 'r':
      opts.nrepeats  = strtoul(optarg, NULL, 0);
      break;
    case 'T':
      strncpy(opts.singletest, optarg,14);   
      if(strncmp(opts.singletest,"ALL",14)!=0 ){
        for(int itest=0;itest<numtests;itest++){
          if(firstst==0) skiptest[itest]=1;
          if(strncmp(tnames[itest],opts.singletest,strlen(opts.singletest))==0 )skiptest[itest]=0;
        }
        firstst=1;
      }
      break;
    case 't':
      opts.l2tabsize = strtoul(optarg, NULL, 0);
      break;
    case 'n':
      opts.l2nupdates= strtoul(optarg, NULL, 0);
      break;
    case 'w':
      opts.minwords  = strtoul(optarg, NULL, 0);
      break;
    case 'W':
      opts.maxwords  = strtoul(optarg, NULL, 0);
      break;
    case 'G':
      opts.msgrowth  = strtoul(optarg, NULL, 0);
      break;
    case 'h':
      print_usage(argv, stdout);
      shmem_finalize();
      exit(EXIT_SUCCESS);
    default:
      print_usage(argv, stderr);
      shmem_finalize();
      exit(EXIT_FAILURE);
    }
  }
  for(int itest=0;itest<numtests;itest++){
      if(itest<=END_SW_ITEST && opts.minwords>1) skiptest[itest]=1; // Skip this test
    }
  return opts;
}



int
handle_options(int argc, char *argv[], int rank, opts_t *bench)
{

  while (1) {
    static struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"check", no_argument, 0, 'c'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    int c = getopt_long(argc, argv, "hc",
                        long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 'h':
      if (rank == 0)
      return OPT_HELP;

    case 'c':
      bench->check = 1;
      break;

    case '?':
      /* getopt_long already printed an error message. */
      if (rank == 0)
      return -1;
      break;

    default:
      abort();
    }
  }

  return 0;
}


/********************************
 divide up total size (loop iters or space amount) in a blocked way
********************************/
void
Block(int myblock, int nblocks, int64_t totalsize, int64_t * start, int64_t * stop,
      int64_t * size)
{
  int64_t div;
  int64_t rem;

  div = totalsize / nblocks;
  rem = totalsize % nblocks;

  if (myblock < rem) {
    *start = myblock * (div + 1);
    *stop = *start + div;
    *size = div + 1;
  }
  else {
    *start = myblock * div + rem;
    *stop = *start + div - 1;
    *size = div;
  }
}


/********************************
  update_table 
********************************/
void
update_table(int64_t tabsize, int64_t nupdate, int64_t *table, int64_t *index, int nrepeats)
{
  uint64_t ran;             /* Current random numbers */
  uint64_t temp;
  double icputime;              /* CPU time to init table */
  double is;
  double cputime;               /* CPU time to update table */
  double s;
  uint64_t *local_table;
  int64_t i, j;
  int itest;
  int64_t one=1;
  double lrwbw, lrbbw, lwmups, lbmups;
  int numthreads;
    int64_t sumval=0;
  int npes,mype;
  double nGbytes;
  int64_t *error_count;
  int64_t *maxval;
  int64_t stackmbuf[nupdate+opts.maxwords];
  int64_t *sheapmbuf;
  int64_t *heapmbuf;
  uint64_t *signal;
  int64_t oldvalue;
  
  int l2tabsize=0;
  npes = shmem_n_pes();
  mype = shmem_my_pe();
//   if (mype == 0) {printf("in update_table\n");fflush(stdout);}
  int lpes = shmemx_local_npes();
  int nnodes = (npes+lpes-1)/lpes;
  if(mype==0){
//     if(lpes*nnodes != npes )printf("*** WARNING:  lpes*nnodes != npes.  Per node calculations may be incorrect.   ***\n");
  }
#pragma omp parallel
    {
    numthreads = omp_get_num_threads();
    }
  habu_set_num_contexts(numthreads);

  error_count = (int64_t*)shmem_malloc((1)*8);
//   if (mype == 0){ printf("before maxval shmalloc\n");fflush(stdout);}
  maxval= (int64_t*)shmem_malloc((1)*8);
  signal = (uint64_t*)shmem_malloc((npes*numthreads)*8);
//   if (mype == 0){ printf("before sheapmbuf shmalloc\n");fflush(stdout);}
  sheapmbuf = (int64_t*)shmem_malloc((nupdate+MAXBLKSIZE)*8);
  if(sheapmbuf==NULL){
    printf("Error. Allocation of sheapmbuf was unsuccessful. nupdate=%ld\n",nupdate);
    return;
  }
//   if (mype == 0){ printf("after sheapmbuf shmalloc\n");fflush(stdout);}
  heapmbuf = (int64_t*)malloc((nupdate+MAXBLKSIZE)*8);
  if(heapmbuf==NULL){
    printf("Error. Allocation of heapmbuf was unsuccessful. nupdate=%ld\n",nupdate);
    return;
  }

  for(i=0;i<nupdate;i++) stackmbuf[i]=0;
//   if (mype == 0){ printf("got to before l2tabsize while\n");fflush(stdout);}
  while((1L<<l2tabsize)<tabsize) l2tabsize +=1;
//   if (mype == 0) printf("got to after l2tabsize while\n");
//   printf("* Table length/rank (elements)   = %ld words\n", tabsize);
//   printf("* In update Log2 Table length/rank         = %d \n", l2tabsize);

  
//  Define the names of tests and control which tests should be run and which should be 
//  skipped.  Skiptest==0 means execute the test.
  int mbsize = tabsize/4;
  mbsize = mbsize<opts.maxwords?mbsize:opts.maxwords;
//  if(tabsize<(MAXBLKSIZE*2)){
//    if(mype==0)printf("ERROR!  Table size is to small for MAXBLKSIZE.\nIncrease table size argument or decrease MAXBLKSIZE in source!\n");
//    return;
    
//  }

    /* Initialize index*/
  int64_t nthreads;  
#pragma omp parallel
  {
  int64_t MYTHREAD = omp_get_thread_num();
  nthreads = omp_get_num_threads();
//     printf("%ld thinks there are %ld threads\n",MYTHREAD,nthreads);
  int64_t start, stop, size;
  int seed;
  unsigned short randstate[3];
  randstate[0] = 5639*(mype*nthreads + MYTHREAD+1);
  randstate[1] = 5827*(mype*nthreads + MYTHREAD+1);
  randstate[2] = 4951*(mype*nthreads + MYTHREAD+1);
  start = stop = 0;
  size = nupdate  / nthreads;
  Block(MYTHREAD, nthreads, nupdate, &start, &stop, &size);
  
//   printf("%d %ld start= %ld stop= %ld size= %ld nupdate=%ld\n",mype, MYTHREAD,start,stop,size,nupdate);
  for (int i=start; i<=stop; i++) {
    index[i] = (int64_t) (erand48 (randstate) * (npes*tabsize));
//     printf("mype= %d index %ld= %ld\n",mype,i,index[i]);
    if(((index[i]&(tabsize-1))+mbsize)>=tabsize) {
//       printf("mype= %d adjust index %ld .and.tabsize = %ld %d \n",mype,i,((index[i]&(tabsize-1))+mbsize),mbsize);
      
      index[i] = index[i]- mbsize ;
    }
    int ipe = index[i]>>l2tabsize;
    int64_t ioff = index[i]&(tabsize-1);
//     printf("init index mype= %d index %ld= %ld ipe= %d ioff= %ld\n",mype,i,index[i],ipe,ioff);
    
  }
  }

  if (mype == 0) {printf("%-14s %7s %7s %8s %6s %8s %8s\n","Test","Bytes", "Time","MUPS","MUPS/N","GiB/s","GiB/s/N");fflush(stdout);}
  
  int tcount=0;
  double geobw=1.0;
  double geomups=1.0;
  double geosecs=1.0;
    int64_t maxrlevel = 0;
//   for(itest=0;itest<2;itest++){
  for(itest=0;itest<numtests;itest++){
    if(skiptest[itest]==1) continue;  // Skip this test
    int blksize=opts.minwords;
    //int blksize=8;
    int blksize2=opts.minwords;
    int lastblksize=1;
//     blksize=600;
//     lastblksize=144;
    int topblksize=mbsize;
//     int topblksize=blksize;
    int sumblksize=0;
    while(blksize<=topblksize){

    /* Initialize main table */
    if(itest<=END_SW_ITEST){
      topblksize=1;
    }
    if((itest%2)==0){
      #pragma omp parallel for        
      for(i = 0; i < tabsize; i += 1){
        if(strncmp(tnames[itest],"ATOMIC_RP",14)==0 || strncmp(tnames[itest],"ATOMIC_FRP",14)==0  ){ table[i] = 0;}
        else{ table[i] = mype+i;}
      }
    }
    if(strncmp(tnames[itest],"GET_NB_lsheap",14)==0){
      #pragma omp parallel for        
      for(i = 0; i < nupdate; i += 1){
        sheapmbuf[i] = -1;
        heapmbuf[i] = -1;
      }
    }
    sumval=0;
    sumblksize += blksize;
    /* Begin timing here */
    icputime = -CPUSEC();
    is = -WSEC();
//  We need to register memory with habu so that it know how to update it  
    habu_mem_t habu_table_handle;
    if(itest>3){
      //  Ask for atomicity.
      habu_table_handle = habu_register_memory(table,sizeof(table[0]),tabsize);
    }else{
      //  Test NON-atomicity.
      habu_table_handle = habu_register_memory(table,sizeof(table[0]),0);
    }
    habu_mem_t habu_local_handle = habu_register_memory(heapmbuf,sizeof(heapmbuf),nupdate);
//     habu_mem_t habu_local_handle = habu_register(heapmbuf,sizeof(heapmbuf),0,nthreads);

    struct hrp_local hrpl = {-1,tabsize,npes};
    struct hrp_local hfrpl = {-1,tabsize,npes};
    hrpl.HABU_RP  = habu_register_op(hrp,16,&hrpl);
    hfrpl.HABU_RP  = habu_register_fop(fhrp,16,&hfrpl);
    double hbartime=0.;


    if (mype == 0) printf("%-14s ",tnames[itest]);fflush(stdout);
    
//     hrpl.HABU_RP = HABU_RP;
//     hrpl.tabsize = tabsize;
//     hrpl.npes = npes;
    
//     hop_tabsize = tabsize;
//     hop_npes = npes;
//     if(mype==0) printf("HABU_RP %ld %ld\n",HABU_RP,hrpl.HABU_RP);
    
    shmem_barrier_all();

    /* Begin timing here */
    icputime += CPUSEC();
    is += WSEC();

    cputime = -CPUSEC();
    s = -WSEC();
#pragma omp parallel
    {
      int MYTHREAD = omp_get_thread_num();
  //     printf("%d thinks there are %d threads\n",MYTHREAD,nthreads);
      int64_t start, stop, size;
      start = stop = 0;
      Block(MYTHREAD, nthreads, nupdate, &start, &stop, &size);
      int64_t mytabstart = (tabsize/nthreads)*MYTHREAD+1;
      
//       printf("%d start= %ld stop= %ld size= %ld nupdate=%ld\n",MYTHREAD,start,stop,size,nupdate);
      

      int64_t i;
      int64_t mype64 = mype;
      int64_t pos;
      int64_t val=0;
      double invslice_size = 1.0/((1.0*tabsize)/nthreads+1);
      for (int ir=0; ir<nrepeats; ir+=1) {
//         for (i=start; i<=stop; i+=blksize) {
//           int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
          switch(itest){
            case 0: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
                #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
                printf("mype= %d %d i= %ld ipe= %d ioff= %ld\n", mype,omp_get_thread_num(),i,ipe,ioff);fflush(stdout);
                #endif
                shmemx_long_add_nb( &table[ioff], -1,ipe,NULL); 
              }
              break;
            case 1: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
                #ifdef HABU_DEBUG_PRINT_TRACK_ITEMS
                printf("mype= %d %d i= %ld ipe= %d ioff= %ld\n", mype,omp_get_thread_num(),i,ipe,ioff);fflush(stdout);
                #endif
                habu_op( habu_table_handle,ioff,ipe, HABU_INC,&ONE,MYTHREAD); } break;
            case 2: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_add_nb( &table[ioff], -1,ipe,NULL); } break;
            case 3: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_op( habu_table_handle,ioff,ipe, HABU_ADD,&ONE,MYTHREAD); } break;
            case 4: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_add_nb( &table[ioff], -1,ipe,NULL); }break;
            case 5: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_op( habu_table_handle,ioff,ipe, HABU_INC,&ONE,MYTHREAD); } break;
            case 6: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_add_nb( &table[ioff], -1,ipe,NULL); } break;
            case 7: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_op( habu_table_handle,ioff,ipe, HABU_ADD,&ONE,MYTHREAD); } break;
            case 8: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_add_nb( &table[mytabstart], -1,ipe,NULL); } break;
            case 9: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_op( habu_table_handle,mytabstart,ipe, HABU_ADD,&ONE,MYTHREAD); } break;
            case 10: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_add_nb( &table[ioff], -2,ipe,NULL); } break;
            case 11: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_op( habu_table_handle,ioff,ipe, HABU_ADD,&ONE,MYTHREAD); 
              habu_op( habu_table_handle,ioff,ipe, HABU_INC,&ONE,MYTHREAD);; 
            } break;
            case 12: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_long_fadd_nb( &sheapmbuf[i], &table[ioff], -1,ipe,NULL); }   break;
            case 13: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_fop( habu_local_handle,i,mype,habu_table_handle,ioff,ipe, HABU_FADD,&ONE,MYTHREAD); } break;
            case 14: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              if(i%(nrepeats*2)>0)continue;
              int64_t rlevel = 1;
                      while(0 != shmem_long_cswap( &table[ioff],0L, 1L,ipe)){
                        ipe = (ipe+919)%npes;
                        ioff = (ioff+907)&(tabsize-1);
                        rlevel++;
                        //printf("mype= %d %d rp ir=%d i= %ld ipe= %d ioff= %ld\n", mype,omp_get_thread_num(),ir,i,ipe,ioff);fflush(stdout);
                        if(rlevel>1000){
                          printf("mype= %d %d detected excessive rlevel rp ir=%d i= %ld ipe= %d ioff= %ld\n", mype,omp_get_thread_num(),ir,i,ipe,ioff);fflush(stdout);
                          break;

                        }
                      }
                      maxrlevel = maxrlevel<rlevel?rlevel:maxrlevel;
              }   break;
            case 15:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              if(i%(nrepeats*2)>0)continue;
              int64_t pack[2]={1,2}; 
              habu_op( habu_table_handle,ioff,ipe, hrpl.HABU_RP,pack,MYTHREAD); 
              
              } break;
            case 16: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              if(i%(nrepeats*2)>0)continue;
              int64_t rlevel = 1;
              while(0 != shmem_long_cswap( &table[ioff],0L, 1L,ipe)){
                ipe = (ipe+919)%npes;
                ioff = (ioff+907)&(tabsize-1);
                rlevel++;
//                         printf("mype= %d %d rp ir=%d i= %ld ipe= %d ioff= %ld\n", mype,omp_get_thread_num(),ir,i,ipe,ioff);fflush(stdout);
              }
              sheapmbuf[i]=ioff;
              maxrlevel = maxrlevel<rlevel?rlevel:maxrlevel;
              }   break;
            case 17:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              if(i%(nrepeats*2)>0)continue;
              int64_t pack[2]={1,2}; 
              habu_fop( habu_local_handle,i,mype,habu_table_handle,ioff,ipe, hfrpl.HABU_RP,pack,MYTHREAD); 
              
              } break;
            case 18: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_get64_nbi( &sheapmbuf[i], &table[ioff], icount,ipe); }  break;
            case 19: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              for(int ii=0;ii<icount;ii++)
                habu_fop( habu_local_handle,i+ii,mype,habu_table_handle,ioff+ii,ipe, HABU_GET,0,MYTHREAD);
              }   break;
            case 20: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_get64_nbi( &sheapmbuf[i], &table[ioff], icount,ipe); }  break;
            case 21:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              habu_getv(habu_local_handle,i,mype,habu_table_handle,ioff,ipe, icount,MYTHREAD);
//               for(int ii=0;ii<icount;ii++)
//                 habu_fop( habu_local_handle,i+ii,mype,habu_table_handle,ioff+ii,ipe, HABU_GET,0,MYTHREAD);
              }   break;
            case 22: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_put64( &table[ioff],&sheapmbuf[i],  icount,ipe); }          break;
            case 23: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
                habu_putv( habu_table_handle, ioff,ipe, &sheapmbuf[i], icount,MYTHREAD); 
//               for(int ii=0;ii<icount;ii++)
//                 habu_op( habu_table_handle,ioff+ii,ipe, HABU_PUT,&sheapmbuf[i+ii],MYTHREAD);
              }          break;
            case 24: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_get64_nbi( &heapmbuf[i], &table[ioff], icount,ipe); }  break;
            case 25: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_get64_nbi( &stackmbuf[i], &table[ioff], icount,ipe); }  break;
            case 26: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_get64( &sheapmbuf[i], &table[ioff], icount,ipe);     }   break;
            case 27:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_put64_nbi( &table[ioff],&sheapmbuf[i], icount,ipe); }   break;
            case 28:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_put64_nbi( &table[ioff],&heapmbuf[i], icount,ipe); }   break;
            case 29:
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_put64_nbi( &table[ioff],&stackmbuf[i], icount,ipe); }   break;
            case 30: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_put64_signal(&table[ioff],&sheapmbuf[i],icount,&signal[mype],1,ipe); } break;
            case 31: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmemx_put64_signal_nb( &table[ioff],&sheapmbuf[i], icount,&signal[mype+MYTHREAD],1,ipe,NULL); }   break;
            case 32: 
              for (i=start; i<=stop; i+=blksize) {
                int ipe = index[i]>>l2tabsize;  int64_t ioff = index[i]&(tabsize-1); int64_t icount = (i+blksize)<=stop?blksize:(stop-i+1);
              shmem_put64( &table[ioff],&sheapmbuf[i], icount,ipe); 
              shmemx_thread_quiet();
              shmem_put64( &signal[mype+MYTHREAD],&ONE,1,ipe); 
          }   break;
          }
        }
//       printf("mype= %d tid= %d done0 generating updates\n",mype,MYTHREAD);fflush(stdout);
//         if(itest%2==1){ 
//           habu_barrier(MYTHREAD);
// //           printf("mype= %d tid= %d after0 habu_barrier\n",mype,MYTHREAD);fflush(stdout);
//         }
//       }
      if(strncmp(tnames[itest],"HABU",4)==0){
        hbartime=-WSEC();
      //if(itest%2==1){
//         if(mype==0) printf("mype= %d tid= %d done generating updates bf habu barrier itest=%d\n",mype,MYTHREAD,itest);fflush(stdout);
        habu_barrier(MYTHREAD);
//         printf("mype= %d tid= %d done generating updates af barrier itest=%d\n",mype,MYTHREAD,itest);fflush(stdout);
        hbartime+=WSEC();
      }
//       habu_barrier(MYTHREAD,nthreads);
//       habu_barrier(MYTHREAD,nthreads);
//       if(itest==0 ||itest==2 ||itest==4 ||itest==6||itest==8) habu_barrier(MYTHREAD,nthreads);
//       if(itest%2==1) printf("mype= %d tid= %d after habu_barrier itest=%d\n",mype,MYTHREAD,itest);
    }
    
    //  Unregister the handles so that those handles can be recycled.
    habu_unregister_memory(habu_table_handle);
    habu_unregister_memory(habu_local_handle);
    habu_unregister_op(hrpl.HABU_RP);
    habu_unregister_op(hfrpl.HABU_RP);
    shmem_barrier_all();
    

    /* End timed section */
    cputime += CPUSEC();
    s += WSEC();
    
    if (mype == 0){
      double mups = nups[itest]*(nrepeats*(double) npes*(double)nupdate / s/1000000);
      //       double mpc = ((double) npes*nupdate / s/1000000)/(npes*numthreads);
      nGbytes= (nups[itest]*nrepeats*(double)npes*nupdate*nbytes[itest]) /GIBI;
      double bw = nGbytes/s;

      printf("%7d %7.2lf %8.1lf %6.1lf %8.2lf %8.2lf ",
              nbytes[itest]*blksize,s,mups,mups/nnodes,bw,bw/nnodes);
//       printf("hbartime=%lf ",hbartime);
      if(strncmp(tnames[itest],"ATOMIC_RP",14)==0) {printf("maxrlevel=%ld ",maxrlevel); maxrlevel=0;}
    }
    if(1 && (blksize+lastblksize > topblksize)){ //  Set if test to true if you want simple error checking on updates
      int error_print_cnt = 0;
      if(itest<=END_CORREST_TEST && ((itest%2)==1)){
        error_count[0]=0;
        maxval[0]=0;
        int64_t ncheck = nupdate;
        //  Run an error check on the target table.
        for (int i = 0; i < tabsize; i++){
          int64_t ref = mype+i;
          if(strncmp(tnames[itest],"HABU_RP",14)==0 || strncmp(tnames[itest],"HABU_FRP",14)==0) {
            if(table[i]!=0 && table[i]!=2) {
              error_count[0] += 1;
              if(error_count[0]<error_print_cnt) printf("%d table %d = %ld \n",mype,i,table[i]);
            }
          }
          if(strncmp(tnames[itest],"HABU_RP",14)!=0 && strncmp(tnames[itest],"HABU_FRP",14)!=0 ) {
            if(table[i]!=ref) {
              error_count[0] += 1;
              if(error_count[0]<error_print_cnt) printf("%d table %d = %ld \n",mype,i,table[i]);
            }
          }
          if(table[i]<maxval[0]) maxval[0]=table[i];
        }
//         if(strncmp(tnames[itest],"HABU_FADD",14)==0 && nrepeats==1){
        if(strncmp(tnames[itest],"HABU_FADD",14)==0 || strncmp(tnames[itest],"HABU_FRP",14)==0 ){
          //  Run an error check on the fetched values.  Only works when nrepeats==1.
          ncheck += nupdate;
          int64_t wiggle = 5;
          if(strncmp(tnames[itest],"HABU_FRP",14)==0) wiggle = 1000;
          for (int i = 0; i < nupdate; i++) {
            if(labs(sheapmbuf[i]-heapmbuf[i])>wiggle*nrepeats){
              //  If the check differs by a handful we can figure that was a simple and expect race
              error_count[0] += 1;
              if(error_count[0]<error_print_cnt) printf("%d sheapmbuf %d = %ld heapmbuf= %ld\n",mype,i,sheapmbuf[i],heapmbuf[i]);
            }
          }
        }
        if(strncmp(tnames[itest],"HABU_GET",14)==0 || strncmp(tnames[itest],"HABU_GETV",14)==0 ){
          //  Run an error check on the fetched values.  Only works when nrepeats==1.
//           ncheck += nupdate;
          for (int i = 0; i < nupdate; i++) {
            if(sheapmbuf[i]!=heapmbuf[i]){
              error_count[0] += 1;
              if(error_count[0]<error_print_cnt) printf("%d sheapmbuf %d = %ld heapmbuf= %ld\n",mype,i,sheapmbuf[i],heapmbuf[i]);
            }
          }
        }
        
        shmem_barrier_all();
        int64_t val;
        if(mype==0){
//           printf("PE %d error_count= %ld\n",0,error_count[0]);
          for(i=1;i<npes;i++) {
            shmem_get64(&val,&error_count[0],1,i);
//             printf("PE %ld error_count= %ld\n",i,val);
            error_count[0] += val;
            shmem_get64(&val,&maxval[0],1,i);
//             printf("PE %ld error_count= %ld\n",i,val);
            if(val<maxval[0]) maxval[0]= val;
          }
          double error_rate = (100.0*error_count[0])/(npes*ncheck);
          if(error_count[0]==0) {
            //  If atomicity is working most tests should pass with 100% correctness.
            printf("PASSED");
          }else{
            //  If atomicity is off, or you are fetching or other cases, we might still count
            //  and error rate of <1% as passing.  But if we are getting any errors we
            //  want to do about it so we have a different print.
            if(error_rate < 1){
              printf("PASSED with %.4g%% error rate",error_rate);
            }else{
              //  Too many errors means we failed the check!
              printf("FAILED with %.4g%% error rate",error_rate);
            }
          }
        }
      }
    }
    int b=blksize;
    blksize+=lastblksize;
    lastblksize=b;
    if(opts.msgrowth>1) blksize=b*opts.msgrowth;
    if(opts.msgrowth<0) {
      if(8*b<(-1*opts.msgrowth)&&b>=8){blksize=b+4;blksize2=blksize;}
      else           {
        blksize=b*1.5;
        if(blksize>=(2*blksize2) || blksize==1){
          blksize2=blksize2*2;
          blksize=blksize2;
        }
      }
    }

//    lastblksize=b;
    if(mype==0) printf("\n");
    }
    if(mype==0 && itest%2==1) printf("\n");
    shmem_barrier_all();  //  Prevents wrap around on the tests

  }
#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  habu_stats(tid);
  }
}

/********************************
  main routine
********************************/
int
main(int argc, char *argv[])
{
  int64_t tabsize;
  int64_t local_tabsize;
  int64_t nupdate;
  int numthreads,mytid;
  int npes,mype;
  int provided;

  int64_t *table;
  int64_t *index;

  double GiB;

  int opt_status;
#ifdef THREAD_HOT
  int requested = SHMEM_THREAD_MULTIPLE;
#if 0
  /* to be used only with OpenSHMEM compliant implementation */
  shmem_init_thread(requested, &provided);/* Old Cray SHMEMX API */
  assert(requested == provided);
#else
  shmemx_init_thread(requested);
#endif
#else  
  shmem_init();
#endif  
  npes = shmem_n_pes();
  mype = shmem_my_pe();
  int lpes = shmemx_local_npes();
  int nnodes = (npes+lpes-1)/lpes;
  if(mype==0){
//     if(lpes*nnodes != npes )printf("*** WARNING:  lpes*nnodes != npes.  Per node calculations may be incorrect.   ***\n");
  }


  opts = check_args(&argc, &argv);
  tabsize = (1L << opts.l2tabsize);
  nupdate = (1L << opts.l2nupdates);
  int nrepeats = opts.nrepeats;
 
#pragma omp parallel
    {
    int64_t MYTHREAD = omp_get_thread_num();
    mytid = MYTHREAD;
#ifdef THREAD_HOT  
    shmemx_thread_register();
#endif    
    numthreads = omp_get_num_threads();
    }
//     printf("%d thinks there are %d threads\n",mytid,numthreads);


  if (!tabsize || !nupdate) {
    if (mype == 0) {
      fprintf(stderr, "ERROR: Incorrect command line argument format.\n");
    }
    exit(1);
  }


  table = (int64_t*)shmem_malloc((tabsize)*8);
  
  if(table==NULL){
    printf("Error. Allocation of table was unsuccessful. \n");
    return 1;
  }
  index = (int64_t*)malloc(nupdate*8);
  if(index==NULL){
    printf("Error. Allocation of index was unsuccessful. \n");
    return 1;
  }
  GiB = tabsize * 8.0 / GIBI;

  if (mype == 0) {
    printf("****************************************************\n");
    printf("* %s version %s  \n*\n", PACKAGE_NAME, PACKAGE_VERSION);
    printf("* NPES                           = %d\n", npes);
    printf("* NNODES (N)                     = %d\n", nnodes);
    printf("* Threads                        = %d\n", numthreads);
    printf("* Tests                          = %s\n", opts.singletest);
    printf("* Table size/PE (GiB)            = %.3f\n", GiB);
    printf("* Table length/PE (elements)     = %ld words\n", tabsize);
    printf("* Log2 Table length/PE           = %zu \n", opts.l2tabsize);
    printf("* Number of updates/PE           = %ld\n", nupdate);
    printf("* nrepeats                       = %d\n", nrepeats);
    if(opts.msgrowth==1){printf("* Msg Size Growth                = Fibonacci\n");}
    else{                printf("* Msg Size Growth                = %d\n", opts.msgrowth);}
    printf("* Index array size/PE (MiB)      = %.3f\n", (double)nupdate*8.0/MIBI);
    printf("* Table array size/PE (MiB)      = %.3f\n", (double)tabsize*8.0/MIBI);
    printf("* Est Memory footprint/PE (MiB)  = %.3f\n", (tabsize+4*nupdate)*8.0/MIBI);
    printf("* Index array size/NODE (MiB)    = %.3f\n", lpes*(double)nupdate*8.0/MIBI);
    printf("* Table array size/NODE (MiB)    = %.3f\n", lpes*(double)tabsize*8.0/MIBI);
    printf("* Est Memory footprint/NODE (MiB)= %.3f\n", lpes*(tabsize+4*nupdate)*8.0/MIBI);
    printf("****************************************************\n");
  }
  
  habu_init(numthreads);

  update_table(tabsize, nupdate, table, index, nrepeats);
  if(mype==0)printf("\n\nHeimdallr65 has seen your network!\n\n");

  shmem_finalize();

  return 0;
}
