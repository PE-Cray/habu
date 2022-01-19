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
// //  
// author:  Nathan Wichmann  (wichmann@hpe.com)
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 

#include <limits.h>



#define habu_mem_t    unsigned char
#define size_habu_mem_t 1

#define habu_op_t unsigned char
#define size_habu_op_t 1
#define size_int64_t 8


#define HABU_CYCLIC_DISTRIBUTION_PE (INT_MIN+65)
#define HABU_CDPE HABU_CYCLIC_DISTRIBUTION_PE
#define HABU_NULL HABU_CYCLIC_DISTRIBUTION_PE

#define habu_max_opcodes UCHAR_MAX  //  Half are ops and half are fops
#define habu_max_memory_register UCHAR_MAX

int habu_init(int max_contexts);
void habu_stats(int ctxt_id);
void habu_barrier(int myctxt);
int64_t habu_poll(int ctxt);

void habu_set_num_contexts(int nc);
int habu_get_max_contexts();
int habu_get_num_contexts();


void habu_op(habu_mem_t target_memory,int64_t index,int target_pe,habu_op_t opcode,void *item_payload,int myctxt);
void habu_fop(habu_mem_t return_memory,int64_t return_index,int reture_pe,habu_mem_t target_memory,int64_t index,int target_pe,habu_op_t opcode, void *item_payload,int threadid);
void habu_putv(habu_mem_t target_memory,int64_t index,int target_pe, void *item_payload, int64_t num_items, int myctxt);
void habu_getv(habu_mem_t return_memory,int64_t return_index,int reture_pe,habu_mem_t target_memory,int64_t index,int target_pe,int64_t num_items, int myctxt);

habu_mem_t habu_register_memory(void *memory, int this_item_size, int64_t num_local_elements);
void habu_unregister_memory(habu_mem_t memory_handle);

int64_t habu_local_reference( habu_mem_t local_memory_handle, int64_t index);

habu_op_t habu_register_op(void (*op_ptr_arr)(habu_mem_t target_memory, int64_t lindex, void *payload,void *local_args, const int ctxt), int payload_size, void *local_args);
habu_op_t habu_register_fop(void (*fop_ptr_arr)(habu_mem_t return_memory,int64_t return_lindex,int return_pe,habu_mem_t target_memory, int64_t target_lindex, void *payload,void *local_args,const int context ), int payload_size, void * local_args);

void habu_unregister_op(habu_op_t opcode);
void habu_unregister_fop(habu_op_t opcode);
void *habu_handle_to_pointer( habu_mem_t memory_handle);
int habu_sizeof_item( habu_mem_t memory_handle);


//  20 opcodes might cover all of the necessary 64-bit operations and fetching operations.
//  We might consider natively supporting 32-bit versions that would take it up to 40 opcodes.
  
extern habu_op_t HABU_PUT;
extern habu_op_t HABU_ADD;
extern habu_op_t HABU_INC;
extern habu_op_t HABU_GET;
extern habu_op_t HABU_FADD;
//  atomic swap
// atomic cswap
// atomic and
// atomic fand
// atomic or
// atomic for
// atomic xor
// atomic fxor
// atomic and.xor
// atomic fand.xor
// atomic imin
// atomic fimin
// atomic imax
// atomic fimax



// #define HABU_DEBUG_PRINT_TRACK_ITEMS 
// #define HABU_DEBUG_PRINT_TRACDK_BUCKETS  


