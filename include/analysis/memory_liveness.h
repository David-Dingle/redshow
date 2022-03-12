/**
 * @file memory_liveness.h
 * @author @Lin-Mao
 * @brief Split the liveness part from memory profile mode. Faster to get the liveness.
 * @version 0.1
 * @date 2022-03-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef REDSHOW_ANALYSIS_MEMORY_LIVENESS_H
#define REDSHOW_ANALYSIS_MEMORY_LIVENESS_H

#include "analysis.h"
#include "operation/operation.h"
#include "operation/kernel.h"
#include "operation/memcpy.h"
#include "operation/memfree.h"
#include "operation/memset.h"
// #include "operation/memory.h" #included in memfree


namespace redshow {

class MemoryLiveness final : public Analysis {

/********************************************************************
 *                  public area for redshow.cpp
 * ******************************************************************/
public:
  MemoryLiveness() : Analysis(REDSHOW_ANALYSIS_MEMORY_LIVENESS) {}

  virtual ~MemoryLiveness() = default;

  // Coarse-grained
  virtual void op_callback(OperationPtr operation, bool is_submemory = false);

  // Fine-grained
  virtual void analysis_begin(u32 cpu_thread, i32 kernel_id, u64 host_op_id, u32 cubin_id, u32 mod_id,
                              GPUPatchType type);

  virtual void analysis_end(u32 cpu_thread, i32 kernel_id);

  virtual void block_enter(const ThreadId &thread_id);

  virtual void block_exit(const ThreadId &thread_id);

  virtual void unit_access(i32 kernel_id, const ThreadId &thread_id, const AccessKind &access_kind,
                           const Memory &memory, u64 pc, u64 value, u64 addr, u32 index,
                           GPUPatchFlags flags);

  // Flush
  virtual void flush_thread(u32 cpu_thread, const std::string &output_dir,
                            const LockableMap<u32, Cubin> &cubins,
                            redshow_record_data_callback_func record_data_callback);

  virtual void flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                     redshow_record_data_callback_func record_data_callback);

/***********************************************************************
 *                 private elements area
 * ********************************************************************/

private:

  struct MemoryLivenessTrace final : public Trace {
    // only need to know memory access, don't care read or write
    // here use memory range to loge access range but not allocation and sub-allocation

    // u64: Memory:Operation->op_id
    Map<u64, Set<MemoryRange>> read_memory;
    Map<u64, Set<MemoryRange>> write_memory;

    MemoryLivenessTrace() = default;

    virtual ~MemoryLivenessTrace() {}
  };

  std::shared_ptr<MemoryLivenessTrace> _trace;

		// <op_id, ctx_id>
	Map<u64, i32> _op_node;

	// <op_id, memory>   log all allocated memory
	Map<u64, std::shared_ptr<Memory>> _memories;

	// <op_id, memory> log current memory
	Map<u64, std::shared_ptr<Memory>> _current_memories;

	enum memory_operation {ALLOC, SET, COPYT, COPYF, ACCESS, FREE};

	Map<u64, Map<u64, memory_operation>> _operations;

/**
 * @brief Kernel end callback
 * 
 * @param op 
 */
void kernel_op_callback(std::shared_ptr<Kernel> op);

/**
 * @brief Memory register callback function
 * 
 * @param op 
 */
void memory_op_callback(std::shared_ptr<Memory> op, bool is_submemory = false);

/**
 * @brief Memory unregister callback function
 * 
 */
void memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory = false);

/**
 * @brief Memcpy register callback function
 * 
 * @param op 
 */
void memcpy_op_callback(std::shared_ptr<Memcpy> op);

/**
 * @brief Memset register callback function
 * 
 * @param op 
 */
void memset_op_callback(std::shared_ptr<Memset> op);

/**
 * @brief Register operations to _operations
 * 
 * @param memory_op_id 
 * @param op_id 
 * @param mem_op 
 */
void memory_operation_register(u64 memory_op_id, u64 op_id, memory_operation mem_op);



};	// MemoryLiveness

}   // namespace redshow

#endif  // REDSHOW_ANALYSIS_MEMORY_LIVENESS_H

