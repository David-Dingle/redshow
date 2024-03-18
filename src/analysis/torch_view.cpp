//
// Created by xjding on 1/1/24.
//

#include "analysis/torch_view.h"

#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>

#include "torch_monitor.h"

#include "common/utils.h"


namespace redshow {

  void TorchView::update_op_node(u64 op_id, i32 ctx_id) {
    if (op_id > REDSHOW_MEMORY_HOST) {
      // Point the operation to the calling context
      _op_node[op_id] = ctx_id;
    }
  }

  void TorchView::op_callback(OperationPtr op, bool is_submemory /* default = false */) {
        // Add a calling context node
    lock();
    if (op->type == OPERATION_TYPE_KERNEL) {
      kernel_op_callback(std::dynamic_pointer_cast<Kernel>(op));
    } else if (op->type == OPERATION_TYPE_MEMORY) {
      memory_op_callback(std::dynamic_pointer_cast<Memory>(op), is_submemory);
    } else if (op->type == OPERATION_TYPE_MEMFREE) {
      memfree_op_callback(std::dynamic_pointer_cast<Memfree>(op), is_submemory);
    } else if (op->type == OPERATION_TYPE_MEMCPY) {
      memcpy_op_callback(std::dynamic_pointer_cast<Memcpy>(op));
    } else if (op->type == OPERATION_TYPE_MEMSET) {
      memset_op_callback(std::dynamic_pointer_cast<Memset>(op));
    }
    unlock();
  }

  void TorchView::memory_op_callback(std::shared_ptr<Memory> op, bool is_submemory) {
    update_op_node(op->op_id, op->ctx_id);

    if (!is_submemory) {
      _memories.try_emplace(op->op_id, op);
      _current_memories.try_emplace(op->op_id, op);
      _op_node.emplace(op->op_id, op->ctx_id);
      _op_type.emplace(op->op_id, "ALLOC");

      _addresses_map.try_emplace(op->memory_range.start, op->op_id);
      _current_memory_usage += op->len;
      _nums_cudamalloc++;

      if (_current_memory_usage > _memory_peak)
        _memory_peak = _current_memory_usage;
    } else {
      _submemories.try_emplace(op->op_id, op);
      _current_submemories.try_emplace(op->op_id, op);
      _sub_addresses_map.try_emplace(op->memory_range.start, op->op_id);
      _current_submemory_usage += op->len;

      if (_current_submemory_usage > _submemory_peak) {
        _submemory_peak = _current_submemory_usage;
      }
    }
  }

  /**
   *
   * */
  void TorchView::memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory) {
  // update_global_op_id_start(op->op_id);
//  update_op_node(op->op_id, op->ctx_id);
//
//  if (!is_submemory) {
//    u64 address = op->memory_range.start;
//    u64 malloc_op_id = _addresses_map.at(address);
//    _op_node.emplace(op->op_id, op->ctx_id);
//    _op_type.emplace(op->op_id, "FREE");
//
//    _addresses_map.erase(address);
//    _current_memories.erase(malloc_op_id);
//    _current_memory_usage -= op->len;
//    _nums_cudafree++;
//
//  } else {
//    u64 address = op->memory_range.start;
//    u64 sub_alloc_id = _sub_addresses_map.at(address);
//
//    _sub_addresses_map.erase(address);
//    _current_submemories.erase(sub_alloc_id);
//    _current_submemory_usage -= op->len;
//  }
}

/**
 * Need fix
 * */
  void TorchView::kernel_op_callback(std::shared_ptr<Kernel> op) {

//    update_op_node(op->op_id, op->ctx_id);
//
//    if (_trace.get() == NULL) {
//      // If the kernel is sampled
//      return;
//    }
//
//    Map<u64, Set<MemoryRange>> initail_unuse_memory_map;
//    _blank_chunks.emplace(_trace->kernel.op_id, initail_unuse_memory_map);
//
//    auto &unuse_memory_map_in_kernel = _blank_chunks[_trace->kernel.op_id];
//
//    // for access trace, no need to identify read or write
//    for (auto &mem_iter : _trace->access_memory) {
//
//#ifdef SUB_MEMORY
//      auto memory = _sub_memories.at(mem_iter.first);
//#endif
//
//#ifndef SUB_MEMORY
//      auto memory = _memories.at(mem_iter.first);
//#endif
//      if (_accessed_memories.find(mem_iter.first) != _accessed_memories.end()) {
//        auto kid = _accessed_memories.at(mem_iter.first);
//        auto mem_unuse_set = _blank_chunks.at(kid).at(mem_iter.first);
//        unuse_memory_map_in_kernel.emplace(mem_iter.first, mem_unuse_set);
//        // ensure the lastest mem_unuse_set
//        _accessed_memories[mem_iter.first] = _trace->kernel.op_id;
//      } else {
//        _accessed_memories.emplace(mem_iter.first, _trace->kernel.op_id);
//        Set<MemoryRange> mem_unuse_set;
//        mem_unuse_set.insert(memory->memory_range);
//        unuse_memory_map_in_kernel.emplace(mem_iter.first, mem_unuse_set);
//      }
//
//      auto node_id = _op_node.at(memory->op_id);
//
//      int r_count = 0; // for debug count
//      // mem_iter.second is a Set<MemRange>
//      for (auto &range_iter : mem_iter.second) {
//        update_blank_chunks(op->op_id, mem_iter.first, range_iter);
//      }
//    }
//    update_object_fragmentation_per_kernel(_trace->kernel.cpu_thread, op->op_id);
//
//    // reset _trace
//    _trace->access_memory.clear();
//    _trace = NULL;
  }

  void TorchView::memcpy_op_callback(std::shared_ptr<Memcpy> op) {
//    auto overwrite = op->len;
//    auto src_memory = _memories.at(op->src_memory_op_id);
//    auto dst_memory = _memories.at(op->dst_memory_op_id);
//    auto src_len = src_memory->len == 0 ? op->len : src_memory->len;
//    auto dst_len = dst_memory->len == 0 ? op->len : dst_memory->len;
//
//    if (op->dst_memory_op_id == REDSHOW_MEMORY_HOST || op->dst_memory_op_id == REDSHOW_MEMORY_UVM) {
//      // sink edge
//      auto dst_ctx_id = _op_node.at(op->dst_memory_op_id);
//
//    } else {
//      link_op_node(op->dst_memory_op_id, op->ctx_id, dst_memory->ctx_id);
//      update_op_node(op->dst_memory_op_id, op->ctx_id);
//    }
//
//    auto src_ctx_id = _op_node.at(op->src_memory_op_id);
//    auto dst_ctx_id = _op_node.at(op->dst_memory_op_id);
//
//    // Update host
//    memory_copy(reinterpret_cast<void *>(op->dst_shadow_start), reinterpret_cast<void *>(op->src_shadow_start),
//                op->len);
//    u64 host = 0;
//    if (op->dst_memory_op_id == REDSHOW_MEMORY_HOST || op->dst_memory_op_id == REDSHOW_MEMORY_UVM) {
//      host = op->dst_shadow_start;
//    } else {
//      host = reinterpret_cast<u64>(dst_memory->value.get());
//    }
//
//#ifdef NDEBUG_TORCH_VIEW
//      std::cout << "ctx: " << op->ctx_id << ", hash: " << hash
//      << " overwrite, " << overwrite << ", memory->len: " << dst_len << std::endl;
//#endif
  }

  void TorchView::memset_op_callback(std::shared_ptr<Memset> op) {
//    u64 overwrite = op->len;
//
//    auto memory = _memories.at(op->memory_op_id);
//    link_op_node(op->memory_op_id, op->ctx_id, memory->ctx_id);
//    update_op_node(op->memory_op_id, op->ctx_id);
//
//    // Update host
//    memset(reinterpret_cast<void *>(op->shadow_start), op->value, op->len);
//    u64 host = reinterpret_cast<u64>(memory->value.get());
//
//    if (_configs[REDSHOW_ANALYSIS_DATA_FLOW_HASH] == true) {
//      std::string hash = compute_memory_hash(host, memory->len);
//      _node_hash[op->ctx_id].emplace(hash);
//
//#ifdef DEBUG_DATA_FLOW
//      std::cout << "ctx: " << op->ctx_id << ", hash: " << hash
//      << " overwrite, " << overwrite << ", memory->len: " << memory->len << std::endl;
//#endif
//    }
  }

  void TorchView::analysis_begin(u32 cpu_thread, i32 kernel_id, u64 host_op_id, u32 stream_id,
                                u32 cubin_id, u32 mod_id, GPUPatchType type, void* trace_data) {
//    assert(type == GPU_PATCH_TYPE_ADDRESS_PATCH || type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS);
//
//    lock();
//
//    if (!this->_kernel_trace[cpu_thread].has(host_op_id)) {
//      auto trace = std::make_shared<MemoryProfileTrace>();
//      trace->kernel.ctx_id = kernel_id;
//      trace->kernel.cubin_id = cubin_id;
//      trace->kernel.mod_id = mod_id;
//      trace->kernel.op_id = host_op_id;
//      this->_kernel_trace[cpu_thread][host_op_id] = trace;
//    }
//    _trace = std::dynamic_pointer_cast<MemoryProfileTrace>(this->_kernel_trace[cpu_thread][host_op_id]);
//
//    unlock();
  }

  void TorchView::analysis_end(u32 cpu_thread, i32 kernel_id) {}

  void TorchView::block_enter(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::block_exit(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::unit_access(i32 kernel_id, u64 host_op_id, const ThreadId &thread_id,
                                 const AccessKind &access_kind, const Memory &memory, u64 pc,
                                 u64 value, u64 addr, u32 index, GPUPatchFlags flags) {
  }

  void TorchView::flush_thread(u32 cpu_thread, const std::string &output_dir,
                                  const LockableMap<u32, Cubin> &cubins,
                                  redshow_record_data_callback_func record_data_callback) {}

  void TorchView::flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                           redshow_record_data_callback_func record_data_callback) {
    std::ofstream output(output_dir + "memory_info.txt");
    output << "GPU memory peak: " << _memory_peak << " B" << std::endl;
    output << "Optimal GPU memory peak: " << _optimal_memory_peak << " B" << std::endl;
    output << "Peak kernel op id: " << _memory_peak_kernel << std::endl;
    output << "Number of cudaMallocs: " << _nums_cudamalloc << std::endl;
    output << "Number of cudaFrees: " << _nums_cudafree << std::endl;
    output << std::endl;

    output << "Submemory peak: " << _submemory_peak << " B" << std::endl;
    output << "Optimal submemory peak: " << _optimal_submemory_peak << " B" << std::endl;
    output << "Peak kernel op id: " << _submemory_peak_kernel << std::endl;

    output.close();

    std::ofstream out(output_dir + "memory_info.csv");
    for (auto op : _op_node) {
      out << "op_id: " << op.first << ", " << _op_type[op.first] << " " << op.second << std::endl;
    }
    out.close();
  }

}  // namespace redshow
