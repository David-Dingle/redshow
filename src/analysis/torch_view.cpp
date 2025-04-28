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

static FILE * fp;

namespace redshow {

  void TorchView::update_op_node(u64 op_id, i32 ctx_id) {}

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

  /**
    * Pass
    * In theory, we don't care about CUDA memory allocation
    * May empty this code block later
    */
  void TorchView::memory_op_callback(std::shared_ptr<Memory> op, bool is_submemory) {}

  /**
    * Pass
    * We don't care about CUDA level memory free operations
    * May empty this code block later
    */
  void TorchView::memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory) {}

/**
 * Need fix
 * */
  void TorchView::kernel_op_callback(std::shared_ptr<Kernel> op) {
    // std::cout << "Enter TORCH_VIEW Kernel op callback." << std::endl;
    if (_trace.get() == NULL) {
      // If the kernel is sampled
      return;
    }

    std::map<uint64_t, std::vector<ViewNode*>> _pc_node_cache;

    // STEP 1
    // Initialize the the _delay_trace table
    // _delayed_trace = NULL; // reset delayed trace to NULL
    if (!_delayed_trace) {
      _delayed_trace = std::make_shared<TorchViewDelayedTrace>();
        torch_monitor_python_state_get(MAX_NUM_STATES, delayed_python_states, &num__delayed_states);
        PyStateCTX _state{-1, num__delayed_states, delayed_python_states};
        if (num__delayed_states > 0) {
         if (!_domain_name.empty()) {
           strcat(_state.py_state[0].function_name, "^");
           strcat(_state.py_state[0].function_name, _domain_name.top().c_str());
         }
        }
        _delayed_trace->python_state = _state;
    }

    // STEP 2
    // Normal update on call_path_map
# ifdef DEBUG
    std::cout << "We Got " <<  _trace->access_memory.size() << " memory accesses." << std::endl;
# endif

    for (auto & [pc, m_c] : _trace->access_memory) {
      for (auto & [m, c] : m_c) {  // m is real mem_start addr, c is 0 place holder. Use op->ctx_id instead
        u64 mem_start = m;
        std::vector<ViewNode*> view_node_hit_mem;

        if(_pc_node_cache.find(pc) != _pc_node_cache.end()) {
          update_node_total_access(_pc_node_cache[pc], pc);
          continue; // just update access counter, but dont add callpath again and again
        } else {
          view_node_hit_mem = get_view_nodes_by_mem_addr(mem_start, false);
          update_node_total_access(view_node_hit_mem, pc);
          _pc_node_cache[pc] = view_node_hit_mem;
        }

        // Update Call ctc_id to CallPath
        for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
          call_path_map[(*viter)->view_id].back().ctxid_pcs[op->ctx_id].push_back(pc);

          // if this is a aten::copy_ domain
          std::cout << "Domain: " << _domain_name.top() << std::endl;
          if (is_aten_copy_domain) {
            if (aten_copy_tar == 0 || aten_copy_src == 0) {
              std::string all_states("");
              if (num__delayed_states > 0 && num__delayed_states <= MAX_NUM_STATES) {
                for(size_t i = 0; i < num__delayed_states; i++) {
                  all_states.append(delayed_python_states[i].file_name);
                  all_states.append(delayed_python_states[i].function_name);
                  all_states.append(std::to_string(delayed_python_states[i].function_first_lineno));
                  all_states.append(std::to_string(delayed_python_states[i].lineno));
                }
              }
              // std::cout << "all_states: " << all_states << std::endl;
              aten_copy_pystate_hash = (std::size_t)std::hash<std::string>{}(all_states);
              std::cout << "unit access: " << aten_copy_pystate_hash << std::endl;
              if (call_path_map[(*viter)->view_id].back().index == 0) {
                aten_copy_tar = (*viter)->view_id;
              } else if (call_path_map[(*viter)->view_id].back().index == 1) {
                aten_copy_src = (*viter)->view_id;
              }
              if(aten_copy_tar != 0 && aten_copy_src != 0) {
                device_view_copy_map[aten_copy_pystate_hash][aten_copy_tar] = aten_copy_src;
              }
            }
          } // Set aten_copy_tar & aten_copy_src
        } // TODO need to be tested
# ifdef DEBUG
        std::cout << "Kernel Access Hits: " << view_node_hit_mem.size() << " View Node(s). :: " << mem_start << std::endl;
# endif
        if (view_node_hit_mem.empty()){
          // if (!_delayed_trace->access_memory.has(pc)) {
          if (true) {
            _delayed_trace->access_memory[pc].emplace(mem_start, op->ctx_id);
          }
        }
      }
    }

# ifdef DEBUG
    std::cout << "Will delay mem size: " << _delayed_trace->access_memory.size() << std::endl;
# endif
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

# ifdef DEBUG
      std::cout << "memcpy hit: " << view_node_hit_src.size() << " " <<
                                     view_node_hit_dst.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
# endif
      update_node_total_access(view_node_hit_src, 0);
      update_node_total_access(view_node_hit_dst, 0);
      update_node_total_access(view_node_hit_shadow, 0);

      // TODO update call_path_map
      if(!view_node_hit_src.empty()) {
        for (auto viter: view_node_hit_src) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_dst.empty()) {
        for (auto viter: view_node_hit_dst) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_shadow.empty()) {
        for (auto viter: view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
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

# ifdef DEBUG
      std::cout << "memset hit: " << view_node_hit_start.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
# endif
      update_node_total_access(view_node_hit_start, 0);
      update_node_total_access(view_node_hit_shadow, 0);

      if(!view_node_hit_start.empty()){
        for (auto viter : view_node_hit_start) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_shadow.empty()){
        for (auto viter : view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
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
    if (_domain_name.top().compare("aten::copy_") == 0){
      is_aten_copy_domain = true;
      aten_copy_pystate_hash = 0;
      aten_copy_tar = 0;
      aten_copy_src = 0;
    }
# ifdef DEBUG
    std::cout << "analysis_begin Kernel ID: " << std::hex << kernel_id << std::dec << std::endl;
# endif
    unlock();
  }

  void TorchView::analysis_end(u32 cpu_thread, i32 kernel_id) {
# ifdef DEBUG
    std::cout << "analysis_end Kernel ID: " << std::hex << kernel_id << std::dec << std::endl;
# endif
    if (is_aten_copy_domain == true && aten_copy_src != 0 && aten_copy_src != 0) {
      // device_view_copy_map[aten_copy_pystate_hash][aten_copy_tar] = aten_copy_src;
      // std::cout << "kernel end aten_copy_pystate_hash : " <<  aten_copy_pystate_hash << std::endl;

      is_aten_copy_domain = false;
      aten_copy_pystate_hash = 0;
      aten_copy_tar = 0;
      aten_copy_src = 0;
    }
  }

  void TorchView::block_enter(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::block_exit(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::unit_access(i32 kernel_id, u64 host_op_id, const ThreadId &thread_id,
                                 const AccessKind &access_kind, const Memory &memory, u64 pc,
                                 u64 value, u64 addr, u32 index, GPUPatchFlags flags) {
    // std::cout << "ENTER TORCH VIEW UNIT ACCESS: " << memory.memory_range.start << " : " << pc << std::endl;
    // if (true) { 
# ifdef DEBUG
      std::cout << "pc: " << std::hex << pc << " mem: " << memory.memory_range.start << std::dec << std::endl;
# endif
    if (!_trace->access_memory.has(pc)) {
      _trace->access_memory[pc].emplace(memory.memory_range.start, 0); // 0 placeholder;
    }
  }

  void TorchView::flush_thread(u32 cpu_thread, const std::string &output_dir,
                                  const LockableMap<u32, Cubin> &cubins,
                                  redshow_record_data_callback_func record_data_callback) {}

  void TorchView::flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                        redshow_record_data_callback_func record_data_callback)
/**
 * new version
 * */
  {
    // clear delayed unit access trace
    map_delayed_access();
    clear_delayed_trace();

    // lock();
    // Log device_view_copy_map
    std::ofstream copy_out(output_dir + "aten_copy_map.txt");
    copy_out << "Target   :   Source" << std::endl;
    for(auto& [_hash, t_s] : device_view_copy_map) {
      copy_out << _hash << std::endl;
      for(auto& [tar, src] : t_s) {
        copy_out << "        " << tar << " : " << src << std::endl;
      }
    }

    copy_out.close();

    std::ofstream out(output_dir + "torch_view_report.csv");

    for(auto iter = call_path_map.begin(); iter != call_path_map.end(); iter++){
      out << "id " << iter->first << std::endl;
      out << "python_state " << std::endl; // Python StateS begin
      for(auto siter = iter->second.begin(); siter != iter->second.end(); siter++){
        // skip states with no ctx info
        // if(siter->ctxid_pcs.empty()) {
        //   continue;
        // }
        // end skip states with no ctx info
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
       // start
        std::string fitst_filename;
        std::string all_states("");
        if (state_length > 0) {
          const size_t last_slash_idx = std::string(siter->py_state[0].function_name).rfind('^');
          if (std::string::npos != last_slash_idx) {
            fitst_filename = std::string(siter->py_state[0].function_name).substr(0, last_slash_idx);
          }
        }
        for(size_t i = 0; i < state_length; i++) {
          all_states.append(siter->py_state[i].file_name);
          if (i == 0 && fitst_filename.length() > 0) {
            all_states.append(fitst_filename);
          } else {
            all_states.append(siter->py_state[i].function_name);
          }
          all_states.append(std::to_string(siter->py_state[i].function_first_lineno));
          all_states.append(std::to_string(siter->py_state[i].lineno));
        }
        out << "pytates_hash " << (std::size_t)std::hash<std::string>{}(all_states) << std::endl;
        // end
        out << "object_type " << (int)siter->object_type << std::endl;
        for(auto& [ctx_id, pc_s] : siter->ctxid_pcs){
          // out << "ctx_id " << std::endl; // ctx_id begin
          // out << ctx_id << std::endl;
          for(auto& pc : pc_s) {
            out << "ctx_id " << std::endl; // ctx_id begin
            out << ctx_id << std::endl;
            out << "pc " << std::endl;
            out << pc << std::endl;
          }
        }
        out << std::endl;
      }
    }

    out.close();

    // Log the forest
    for (unsigned i = 0; i < _roots.size(); i++) {
      std::ofstream fout(output_dir + "forest.txt", std::ios::app);
      for (unsigned i = 0; i < _roots.size(); i++) {
        _roots.at(i)->delete_children_nodes(fout);
        fout << '\n';
        _roots.erase(_roots.begin()+i);
        --i;
      }
      fout.close();
    }
    // unlock();
  }

}  // namespace redshow
