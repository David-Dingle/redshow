//
// Created by xjding on 1/1/24.
//

#include "analysis/torch_view.h"

#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string.h>

#include "torch_monitor.h"

#include "common/utils.h"


namespace redshow {

  void TorchView::update_op_node(u64 op_id, i32 ctx_id) {
//    if (op_id > REDSHOW_MEMORY_HOST) {
//      // Point the operation to the calling context
//      _op_node[op_id] = ctx_id;
//    }
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
    /**
     * Pass
     * In theory, we don't care about CUDA memory allocation
     * May empty this code block later
     */
//    _memories.try_emplace(op->op_id, op);
//    _current_memories.try_emplace(op->op_id, op);
//    _op_node.emplace(op->op_id, op->ctx_id);
//    _op_type.emplace(op->op_id, "ALLOC");
//
//    _addresses_map.try_emplace(op->memory_range.start, op->op_id);
//    _nums_cudamalloc++;
  }

  void TorchView::memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory) {
    /**
     * Pass
     * We don't care about CUDA level memory free operations
     * May empty this code block later
     */
//    u64 address = op->memory_range.start;
//    u64 malloc_op_id = _addresses_map.at(address);
//    _op_node.emplace(op->op_id, op->ctx_id);
//    _op_type.emplace(op->op_id, "FREE");
//    _addresses_map.erase(address);
//    _current_memories.erase(malloc_op_id);
//    _current_memory_usage -= op->len;
//    _nums_cudafree++;
  }

/**
 * Need fix
 * */
  void TorchView::kernel_op_callback(std::shared_ptr<Kernel> op) {
    // std::cout << "Enter TORCH_VIEW Kernel op callback." << std::endl;
    if (_trace.get() == NULL) {
      // If the kernel is sampled
      return;
    }
    if (_delayed_trace.get() != NULL){ // if the previous kernel view-node mapping is delayed
      /** handle delayed unit access:
       *  1. map unit access to the updated forest
       *  2. if the map miss again, attribute the access to PyTorch Allocator's mem-block
       */
      std::cout << "Delayed " <<  _delayed_trace->access_memory.size() << " memory accesses." << std::endl;
      for (auto &trace_iter : _delayed_trace->access_memory) {
        u64 mem_start = trace_iter.first;
        std::vector<ViewNode*> view_node_hit_mem = get_view_nodes_by_mem_addr(mem_start, true);
        update_node_total_access(view_node_hit_mem);
        // Update Call ctc_id to CallPath TODO: use the old python state and then insert ctx_id
        for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
          _delayed_trace->python_state.object_type = VIEW_NODE;
          call_path_map[(*viter)->view_id].push_back(_delayed_trace->python_state);
          call_path_map[(*viter)->view_id].back().ctx_id.push_back(_trace->access_memory[mem_start]);
        }
        std::cout << "Delayed Kernel Access Hits: " << view_node_hit_mem.size() << " View Node(s). :: " << mem_start << std::endl;
        if(view_node_hit_mem.empty()){
          std::vector<MemoryBlock*> mem_blocks_hit = get_mem_block_by_mem_addr(mem_start);
          std::cout << "Memory Block Hit: " << mem_blocks_hit.size() << std::endl;
          // TODO insert mem_block_id, delayed_Python_state, object_type, and ctx_id in the call_path_map
          for (auto miter : mem_blocks_hit) {
            _delayed_trace->python_state.object_type = MEMORY_BLOCK;
            call_path_map[(*miter).block_id].push_back(_delayed_trace->python_state);
            call_path_map[(*miter).block_id].back().ctx_id.push_back(_trace->access_memory[mem_start]);
          }
        }
      }
      _delayed_trace = NULL; // reset delayed trace to NULL
    }
    if (!_delayed_trace) {
      _delayed_trace = std::make_shared<TorchViewDelayedTrace>();
      // TODO(): assign current Python State to it's field for delayed useage
      PyStateCTX _state{-1, num_states, python_states};
      _delayed_trace->python_state = _state;
    }
    std::cout << "We Got " <<  _trace->access_memory.size() << " memory accesses." << std::endl;
    for (auto &trace_iter : _trace->access_memory) {
      u64 mem_start = trace_iter.first;
      std::vector<ViewNode*> view_node_hit_mem = get_view_nodes_by_mem_addr(mem_start);
      update_node_total_access(view_node_hit_mem);
      // Update Call ctc_id to CallPath
      for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
        call_path_map[(*viter)->view_id].back().ctx_id.push_back(_trace->access_memory[mem_start]);
      } // TODO need to be tested
      std::cout << "Kernel Access Hits: " << view_node_hit_mem.size() << " View Node(s). :: " << mem_start << std::endl;
      if (view_node_hit_mem.empty()){
        if (!_delayed_trace->access_memory.has(trace_iter.first)) {
          _delayed_trace->access_memory.emplace(trace_iter.first, trace_iter.second);
        }
      }
    }
    std::cout << "Will delay mem size: " << _delayed_trace->access_memory.size() << std::endl;
    // check if any unit access has been delayed
    if(_delayed_trace->access_memory.empty()){
      _delayed_trace = NULL;
    }
    _trace->access_memory.clear();
    _trace = NULL;
  }

  void TorchView::memcpy_op_callback(std::shared_ptr<Memcpy> op) {
    if ((op->dst_memory_op_id != REDSHOW_MEMORY_HOST && op->dst_memory_op_id != REDSHOW_MEMORY_UVM) // im case dst on device
         ||
        (op->src_memory_op_id != REDSHOW_MEMORY_HOST && op->src_memory_op_id != REDSHOW_MEMORY_UVM)) { // in case src on device
      // auto dst_ctx_id = _op_node.at(op->dst_memory_op_id);
      u64 overwrite_len = op->len;
      u64 src_start = op->src_start;
      u64 dst_start = op->dst_start;
      u64 dst_shadow_start = op->dst_shadow_start;

      // (mem_range_t)mem_range{mem_addrs, mem_addrs + op->len};
      std::vector<ViewNode*> view_node_hit_src = get_view_nodes_by_mem_addr(src_start);
      std::vector<ViewNode*> view_node_hit_dst = get_view_nodes_by_mem_addr(dst_start);
      std::vector<ViewNode*> view_node_hit_shadow = get_view_nodes_by_mem_addr(dst_shadow_start);

      std::cout << "memcpy hit: " << view_node_hit_src.size() << " " <<
                                     view_node_hit_dst.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
      update_node_total_access(view_node_hit_src);
      update_node_total_access(view_node_hit_dst);
      update_node_total_access(view_node_hit_shadow);

      // TODO update call_path_map
      if(!view_node_hit_src.empty()) {
        for (auto viter: view_node_hit_src) {
          call_path_map[(*viter).view_id].back().ctx_id.push_back(op->ctx_id);
        }
      }
      if(!view_node_hit_dst.empty()) {
        for (auto viter: view_node_hit_dst) {
          call_path_map[(*viter).view_id].back().ctx_id.push_back(op->ctx_id);
        }
      }
      if(!view_node_hit_shadow.empty()) {
        for (auto viter: view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctx_id.push_back(op->ctx_id);
        }
      }
    }
  }

  void TorchView::memset_op_callback(std::shared_ptr<Memset> op) {
    if (op->memory_op_id != REDSHOW_MEMORY_HOST && op->memory_op_id != REDSHOW_MEMORY_UVM) {
      u64 overwrite = op->len;
      u64 start = op->start;
      u64 dst_shadow_start = op->shadow_start;
      u64 value = op->value;

      std::vector<ViewNode*> view_node_hit_start = get_view_nodes_by_mem_addr(start);
      std::vector<ViewNode*> view_node_hit_shadow = get_view_nodes_by_mem_addr(dst_shadow_start);

      std::cout << "memset hit: " << view_node_hit_start.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
      update_node_total_access(view_node_hit_start);
      update_node_total_access(view_node_hit_shadow);

      if(!view_node_hit_start.empty()){
        for (auto viter : view_node_hit_start) {
          call_path_map[(*viter).view_id].back().ctx_id.push_back(op->ctx_id);
        }
      }
      if(!view_node_hit_shadow.empty()){
        for (auto viter : view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctx_id.push_back(op->ctx_id);
        }
      }
    }
  }

  void TorchView::analysis_begin(u32 cpu_thread, i32 kernel_id, u64 host_op_id, u32 stream_id,
                                u32 cubin_id, u32 mod_id, GPUPatchType type, void* trace_data) {
    // configured in sanitizer-api.c:sanitizer_torch_view_analysis_enable()
//    if(type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS)
//      return;
    assert(type == GPU_PATCH_TYPE_ADDRESS_PATCH || type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS);
    // gpu_patch_buffer_t* buffer = static_cast<gpu_patch_buffer_t*>(trace_data);
    lock();
    // ?? How to make sure this _trace are the same with the _trace in kernel_op_callback
    if (!_trace) {
      _trace = std::make_shared<TorchViewTrace>();
    }
    unlock();
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
    std::cout << "ENTER TORCH VIEW UNIT ACCESS: " << memory.memory_range.start << std::endl;
    if (!_trace->access_memory.has(memory.memory_range.start)) {
      _trace->access_memory.emplace(memory.memory_range.start, memory.ctx_id);
      std::cout << "Trace mem size: " << _trace->access_memory.size() << std::endl;
    }
  }

  void TorchView::flush_thread(u32 cpu_thread, const std::string &output_dir,
                                  const LockableMap<u32, Cubin> &cubins,
                                  redshow_record_data_callback_func record_data_callback) {}

  void TorchView::flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                        redshow_record_data_callback_func record_data_callback)
//  {
//    std::ofstream out(output_dir + "torch_view_report.json");
//    out << "[" << std::endl; // file begin
//    for(auto iter = call_path_map.begin(); iter != call_path_map.end(); ){
//      out << " {" << std::endl; // call_path_map item begin
//        out << "  \"id\": " << iter->first << "," << std::endl;
//        out << "  \"python_state\": [" << std::endl; // Python StateS begin
//        for(auto siter = iter->second.begin(); siter != iter->second.end(); ){
//          out << "   {" << std::endl; // State begin
//          out << "    \"index\": " << siter->index << "," << std::endl;
//          out << "    \"num_states\": " << siter->num_states << "," << std::endl;
//          out << "    \"py_state\": [" << std::endl; // py_state begin
//          size_t state_length = (siter->num_states < MAX_NUM_STATES ? siter->num_states : MAX_NUM_STATES);
//          for(size_t i = 0; i < state_length; ) {
//            out << "    {" << std::endl; // torch_monitor_python_state_t begin
//            out << "     " << "\"file_name\": \""<< siter->py_state[i].file_name << "\"," << std::endl;
//            out << "     " << "\"function_name\": \""<< siter->py_state[i].function_name << "\"," << std::endl;
//            out << "     " << "\"function_first_lineno\": "<< siter->py_state[i].function_first_lineno << ","<< std::endl;
//            out << "     " << "\"lineno\": "<< siter->py_state[i].lineno << std::endl;
//            if (++i != state_length){
//              out << "    }," << std::endl;} // torch_monitor_python_state_t ends
//            else {
//              out << "    }" << std::endl;}
//          }
//          out << "    ]," << std::endl; // py_state end
//          out << "    \"object_type\": " << (int)siter->object_type << "," << std::endl;
//          out << "    \"ctx_id\": [" << std::endl; // ctx_id begin
//          for(auto citer = siter->ctx_id.begin(); citer != siter->ctx_id.end(); ){
//            out << "    " << (*citer);
//            if (++citer != siter->ctx_id.end()){
//              out << ",";
//            }
//            out << std::endl;
//          }
//          out << "    ]" << std::endl; // ctx_id end
//          out << "   }";
//          if (++siter != iter->second.end()){
//            out << ",";
//          }
//          out << std::endl;
//        }
//        out << "   ]" << std::endl; // Python stateS end
//      out << " }";
//      if (++iter != call_path_map.end()) {
//        out << ",";
//      }
//      out << std::endl;
//      //--iter;
//    }
//    out << "]" << std::endl; // file end
//    out.close();
//  }
/**
 * new version
 * */
  {
    std::ofstream out(output_dir + "torch_view_report.csv");

    for(auto iter = call_path_map.begin(); iter != call_path_map.end(); iter++){
      out << "id " << iter->first << std::endl;
      out << "python_state " << std::endl; // Python StateS begin
      for(auto siter = iter->second.begin(); siter != iter->second.end(); siter++){
        out << "index " << siter->index << std::endl;
        out << "num_states " << siter->num_states << std::endl;
        out << "py_state " << std::endl; // py_state begin
        size_t state_length = (siter->num_states < MAX_NUM_STATES ? siter->num_states : MAX_NUM_STATES);
        for(size_t i = 0; i < state_length; i++) {
          out << "file_name "<< siter->py_state[i].file_name << std::endl;
          out << "function_name "<< siter->py_state[i].function_name << std::endl;
          out << "function_first_lineno "<< siter->py_state[i].function_first_lineno << std::endl;
          out << "lineno "<< siter->py_state[i].lineno << std::endl;
        }
        out << "object_type " << (int)siter->object_type << std::endl;
        out << "ctx_id " << std::endl; // ctx_id begin
        for(auto citer = siter->ctx_id.begin(); citer != siter->ctx_id.end(); citer++){
          out << (*citer) << std::endl;
        }
        out << std::endl;
      }
    }

    out.close();
  }

}  // namespace redshow
