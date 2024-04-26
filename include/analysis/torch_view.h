//
// Created by xjding on 1/1/24.
// TODO(XJDing):
//  is callback func mutax needed? (Done)
//  where comes the double free (!prev) error? (disappeared)
//  C/CUDA Call path;
//  Early stop for tensor strides sorting with the aid of transpose flag;
//  GPA;
//

#ifndef REDSHOW_ANALYSIS_TORCH_VIEW_H
#define REDSHOW_ANALYSIS_TORCH_VIEW_H

#include <mutex>
#include <string>
#include <optional>
#include <stack>

#include "analysis.h"
#include "binutils/cubin.h"
#include "common/map.h"
#include "common/utils.h"
#include "operation/kernel.h"
#include "operation/memcpy.h"
#include "operation/memory.h"
#include "operation/memset.h"
#include "operation/memfree.h"
#include "operation/operation.h"
#include "redshow.h"
#include <iostream>
#include <algorithm>

#include "/home/xjding/Projects/new_DrGPUM/DrGPUM/torch-monitor/include/torch_monitor.h"

#include <fstream>

const static size_t MAX_NUM_STATES = 30;
thread_local static size_t num_states;
thread_local static torch_monitor_python_state_t python_states[MAX_NUM_STATES];

namespace redshow {

 class TorchView final : public Analysis {
  public:
   TorchView() : Analysis(REDSHOW_ANALYSIS_TORCH_VIEW) {}

   virtual ~TorchView() = default;

   // Coarse-grained
   virtual void op_callback(OperationPtr operation, bool is_submemory = false);

   // Fine-grained
   virtual void analysis_begin(u32 cpu_thread, i32 kernel_id, u64 host_op_id, u32 stream_id,
                                    u32 cubin_id, u32 mode_id, GPUPatchType type, void* aux = NULL);

   virtual void analysis_end(u32 cpu_thread, i32 kernel_id);

   virtual void block_enter(const ThreadId &thread_id);

   virtual void block_exit(const ThreadId &thread_id);

   virtual void unit_access(i32 kernel_id, u64 host_op_id, const ThreadId &thread_id,
                                 const AccessKind &access_kind, const Memory &memory, u64 pc,
                                 u64 value, u64 addr, u32 index, GPUPatchFlags flags);

   virtual void flush_thread(u32 cpu_thread, const std::string &output_dir,
                                  const LockableMap<u32, Cubin> &cubins,
                                  redshow_record_data_callback_func record_data_callback);

   virtual void flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                           redshow_record_data_callback_func record_data_callback);

   /**
    * Data Structure
    * -------------------------------------------------------------------------------------
    */
   typedef u64 data_ptr_t;
   typedef u64 metadata_ptr_t;
   typedef std::pair<data_ptr_t, data_ptr_t> mem_range_t;

   struct ViewNode {
     u64 view_id;
     int64_t index;
     int64_t numel;  // number of tensor elements
     int64_t dim;
     torch_monitor_scalar_type_t dtype;  // pytorch tensor.dtype (scalar type)
     uint64_t itemsize;
     int64_t storage_offset;
     int64_t sizes[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
     int64_t strides[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
     data_ptr_t data_ptr;
     metadata_ptr_t metadata_ptr;
     mem_range_t mem_block_range;  // <block start, block end>
     int _total_access = 0;
     std::vector<ViewNode*> _children = {};

     public:
       /**temp object for searching*/
      ViewNode(data_ptr_t data_ptr, metadata_ptr_t metadata_ptr, int64_t numel, uint64_t itemsize):data_ptr(data_ptr), metadata_ptr(metadata_ptr), numel(numel), itemsize(itemsize) {}

      /**for root nodes*/
      ViewNode(u64 view_id, int64_t index, int64_t numel, int64_t dim, torch_monitor_scalar_type_t dtype, uint64_t itemsize, int64_t storage_offset,
               int64_t sizes[], int64_t strides[], data_ptr_t data_ptr, metadata_ptr_t metadata_ptr):
               view_id(view_id), index(index), numel(numel), dim(dim), dtype(dtype), itemsize(itemsize), storage_offset(storage_offset), data_ptr(data_ptr),
               metadata_ptr(metadata_ptr){
        for(int i = 0; i < std::min<int>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, dim); i ++){
          this->sizes[i] = sizes[i];
          this->strides[i] = strides[i];
        }
        this->mem_block_range = mem_range_t{data_ptr, data_ptr+(itemsize * numel)};
      }

      /**other nodes*/
      ViewNode(u64 view_id, int64_t index, int64_t numel, int64_t dim, torch_monitor_scalar_type_t dtype, uint64_t itemsize, int64_t storage_offset,
               int64_t sizes[], int64_t strides[], data_ptr_t data_ptr, metadata_ptr_t metadata_ptr,
               mem_range_t mem_block_range):
               view_id(view_id), index(index), numel(numel), dim(dim), dtype(dtype), itemsize(itemsize), storage_offset(storage_offset), data_ptr(data_ptr),
               metadata_ptr(metadata_ptr){
        this->mem_block_range = mem_range_t{mem_block_range};
        for(int i = 0; i < std::min<int>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, dim); i ++){
          this->sizes[i] = sizes[i];
          this->strides[i] = strides[i];
        }
      }

//      ~ViewNode(){
//        std::cout<< "destruct: " << (uint64_t)this->data_ptr << std::endl;
//      }

      bool operator==(const ViewNode& r_val) {
        return ((this->data_ptr == r_val.data_ptr) && (this->metadata_ptr == r_val.metadata_ptr));
      }

      /**
       * recursive member function of view node object
       * return node ptr if the view sits on this branch
       * return null if the view sits nowhere on this branch
       * */
      ViewNode* find_node(ViewNode& r_node) {
        if (*(this) == r_node) {
          return this;
        }
        if (!(this->_children.empty())) {
          ViewNode* res = nullptr;
          for (auto node : this->_children) {
            res = node->find_node(r_node);
            if (res) {
              return res;  // return the node ptr
            }
          }
          return res;  // not found return nullptr
        } else {
          return nullptr;  // this._children is empty
        }
      }

      bool is_leaf_node() {
        return this->_children.empty();
      }

      void delete_children_nodes() {
        if (! this->is_leaf_node()) {
          for (auto child : this->_children) {
            child->delete_children_nodes();
            delete child;
          }
          this->_children.clear();
        }
      }
   };

   std::stack<torch_monitor_op_data_t> _op_stack = std::stack<torch_monitor_op_data_t>();
   std::stack<torch_monitor_op_data_t> _op_stack_temp = std::stack<torch_monitor_op_data_t>();  // Deprecated
   torch_monitor_op_data_t _popped_op; // = torch_monitor_op_data_t();

   struct PyStateCTX {
    public:
     int64_t index;  // arg index in torch-monitor callback inputs list
     size_t num_states;
     torch_monitor_python_state_t py_state[MAX_NUM_STATES];
     std::vector<u64> ctx_id;

     PyStateCTX(int64_t index, size_t num_states, torch_monitor_python_state_t (&arg_py_state)[MAX_NUM_STATES]):
       index(index), num_states(num_states)
     {
       ctx_id = std::vector<u64>{};  // init as empty vector
       for (int i = 0; i < (num_states < MAX_NUM_STATES ? num_states : MAX_NUM_STATES); i++){
         py_state[i].file_name = arg_py_state[i].file_name;
         py_state[i].function_name = arg_py_state[i].function_name;
         py_state[i].function_first_lineno = arg_py_state[i].function_first_lineno;
         py_state[i].lineno = arg_py_state[i].lineno;
       } // TODO(Ding): Verify the correctness of shallow copy
     };
   };

   std::vector<ViewNode*> _roots = {};  // The node forest //TODO: optimize
   std::map<u64, std::vector<PyStateCTX>> call_path_map = {};
   std::vector<mem_range_t> gpu_mem_blocks = {};

   /** find view root from the forest */
   ViewNode* find_root_node(ViewNode r_node) {
     for (auto node : _roots){
       if (r_node.data_ptr >= node->mem_block_range.first && (r_node.data_ptr + r_node.itemsize * r_node.numel) <= node->mem_block_range.second){
         return node;
       }
     }
     return nullptr;
   }

  /**
   * add a node to the forest iif captured a new tensor/view
   * increase the corresponding _total_access by 1 if the tensor/view exists
   * 1. check if the view belongs to any root node?
   *    if so, search on corresponding branch
   *    1.1. not in the branch: add a view node into the chain
   *    1.2. in the branch: do nothing here. _total_access will be added by elsewhere when (mem_rw/view_offset) is captured. (TODO)
   * 2. else, create a root node
   * */
   void update_view_forest(torch_monitor_callback_tensor_data_t& tensor_data, u64 global_id) {
     lock();

     data_ptr_t data_ptr = (data_ptr_t)tensor_data.data_ptr;
     metadata_ptr_t metadata_ptr = (metadata_ptr_t)tensor_data.metadata_ptr;
     ViewNode temp = ViewNode{data_ptr, metadata_ptr, tensor_data.numel, tensor_data.itemsize};
     ViewNode* root_found = find_root_node(temp);
     if(root_found){
       ViewNode* view_existed = root_found->find_node(temp);
       if(view_existed){
         // Update Python State
         PyStateCTX _state{tensor_data.index, num_states, python_states};
         call_path_map[view_existed->view_id].push_back(_state);
       } else { // found the view node in the forest
         /** insert new view node into correct "view_node._children"
           * 1. find the father node from the root/branch
           * 2. create a new node and insert into its "_children"
           * */
         // fetch the closest pytorch callback op inputs from the stack (popped op)
         for (int64_t i = 0; i < _op_stack.top().input_output_data.size; i++) {
           torch_monitor_callback_tensor_data_t _tensor = _op_stack.top().input_output_data.tensor_data[i];
           if (_tensor.index == -1 || _tensor.numel <= 0)
             continue;
           data_ptr_t _tensor_data_ptr = (data_ptr_t)_tensor.data_ptr;
           metadata_ptr_t _tensor_metadata_ptr = (metadata_ptr_t)_tensor.metadata_ptr;
           ViewNode _input_temp = ViewNode{_tensor_data_ptr, _tensor_metadata_ptr, _tensor.numel, _tensor.itemsize};
           ViewNode* _input_root = root_found; //find_root_node(_input_temp);
           ViewNode* _input_view = _input_root->find_node(_input_temp);
           if (!_input_view) // if the candidate father does not exist ; continue and look on the next one
             continue;
           if ((data_ptr_t)tensor_data.data_ptr >= _input_view->mem_block_range.first && (data_ptr_t)tensor_data.data_ptr <= _input_view->mem_block_range.second) {
             // add the tensor_data to _tensor(node)'s children
             ViewNode* _node = new ViewNode(global_id, tensor_data.index, tensor_data.numel, tensor_data.dim, tensor_data.dtype, tensor_data.itemsize,
                                                                          tensor_data.storage_offset, tensor_data.sizes, tensor_data.strides, data_ptr,
                                                                          metadata_ptr, _input_view->mem_block_range);
             _input_view->_children.push_back(_node);
             call_path_map[global_id] = std::vector<PyStateCTX>();
             PyStateCTX _state{tensor_data.index, num_states, python_states};
             call_path_map[global_id].push_back(_state);
             break;
           }
         }
       }
     } else { // add a new root
       ViewNode* _root_node_ptr = new ViewNode(global_id, tensor_data.index, tensor_data.numel, tensor_data.dim, tensor_data.dtype, tensor_data.itemsize,
                                                                               tensor_data.storage_offset, tensor_data.sizes, tensor_data.strides, data_ptr,
                                                                               metadata_ptr);
       _roots.push_back(_root_node_ptr);
       call_path_map[global_id] = std::vector<PyStateCTX>();
       PyStateCTX _state{tensor_data.index, num_states, python_states};
       call_path_map[global_id].push_back(_state);
     } // add a new root
     unlock();
   }

   /**
    * Print out the view forest for debugging use
    */
#ifdef DEBUG
   void visualize_view_forest(ViewNode* root, int indent_level = 0, const char* indent = "    |"){
     for (int i = 0; i < indent_level; i++)
       std::cout << indent;
     std::cout << "ID: " << root->view_id << " D_ptr: " << root->data_ptr << " numel: " << root->numel << " dim: " << root->dim << " M_ptr: " << root->metadata_ptr << " Range_f: " << root->mem_block_range.first << " Range_s: " << root->mem_block_range.second << " Access: " << root->_total_access;
     if (!root->_children.empty()){
       for (ViewNode* citer : root->_children) {
         visualize_view_forest(citer, indent_level + 1, indent);
       }
     } else {
       std::cout << std::endl;
     }
   }
#endif

   /**
    * delete a tree(gaven a tensor ptr) from the forest
    *@param mem_start_addr: starting address of memory range
    */
   void delete_forest_tree(data_ptr_t mem_start_addr, int64_t total_allocated) {
     lock();
     for (unsigned i = 0; i < _roots.size(); i++) {
       if (_roots.at(i)->mem_block_range.first >= mem_start_addr && _roots.at(i)->mem_block_range.second <= (mem_start_addr + total_allocated)){
         _roots.at(i)->delete_children_nodes();
         _roots.erase(_roots.begin()+i);
         break;
       }
     }
     unlock();
   }

   /**
    * add view_node _total_access by one
    * @param a list of view_nodes
    * TODO: call this func at project stage 3, (use data_ptr, offset, stride, itemsize to spot the view_node which needs update)
    */
   void update_node_total_access(std::vector<ViewNode*> view_nodes) {
     // lock(); // TODO: Try fix later: another lock; Or try to make it sequential
     for (auto viter : view_nodes) {
       ViewNode* _root_node = find_root_node(*viter);
       if (!_root_node) {
         continue;
       }
       ViewNode* _node = _root_node->find_node(*viter);
       _node->_total_access++;
     }
     // unlock();
   }

   /**
    * Bubble sort tensor 'sizes' and 'strides' with descending ordered strides
    * as tensor dimension shouldn't be extremely high, bubble sort is fine.
    * @param tensor: a PyTorch tensor view
    * @return a view copy with the same info, but sorted strides and their corresponding sizes
    */
   torch_monitor_callback_tensor_data_t sort_tensor_strides(torch_monitor_callback_tensor_data_t const tensor) {
     torch_monitor_callback_tensor_data_t res = tensor;
     int64_t dim = tensor.dim;
     bool swapped;
     for(int64_t i = 0; i < dim - 1; i++) {
       swapped = false;
       int64_t temp;
       for(int64_t j = 0; j < dim - i - i; j++) {
         if(res.strides[j] < res.strides[j + 1]){
           temp = res.strides[j];
           res.strides[j] = res.strides[j + 1];
           res.strides[j + 1] = temp;
           temp = res.sizes[j];
           res.sizes[j] = res.sizes[j + 1];
           res.sizes[j + 1] = temp;
           swapped = true;
         }
         if (swapped == false)
           break;
       }
     }
     return res;
   }

   /**
    * Binary search on the given dimension
    * @param size: size on given dimension
    * @param stride stride in given dimension
    * @param itemsize sizeof(tensor.dtype)
    * @param mem_addr_hit
    * @return
    */
   u64 find_local_starting_address(u64 init_address, int64_t size, int64_t stride, uint64_t itemsize, u64 mem_addr_hit) {
     if (size >= 1){
       u64 sit = init_address + ((size-1) / 2) * itemsize * stride;
       if (sit == mem_addr_hit) {
         return sit;
       } else if (sit > mem_addr_hit){
         return find_local_starting_address(init_address, ((size-1) / 2), stride, itemsize, mem_addr_hit);
       } else { // sit < mem_addr_hit
         if ((sit + itemsize * stride) > mem_addr_hit) {
           return sit;
         } else {
           return find_local_starting_address((sit + itemsize * stride), (size - (size / 2)), stride, itemsize, mem_addr_hit);
         }
       }
     } else {
       return 0;
     }
   }
   /**
    * recursive function that always return the updated dimensional starting address
    *
    * @param tensor: a temp value with gradually smaller dimension while recursion
    * @param itemsize: sizeof(tensor.dtype) c10::scalarTypeToTypeMeta(c10::ScalarType(dtype)).itemsize();
    * @return the address of
    */
   u64 find_closest_starting_address(torch_monitor_callback_tensor_data_t temp_tensor, uint64_t itemsize, u64 mem_addr_hit){
     int64_t * sizes = temp_tensor.sizes;
     int64_t * strides = temp_tensor.strides;
     int64_t dim = temp_tensor.dim;
     if (dim <= 1)
       return (u64)temp_tensor.data_ptr;
     else {
       u64 sit = (u64)temp_tensor.data_ptr;
       for(int64_t i = 0; i < (dim-1); i++) {
         sit = find_local_starting_address(sit, sizes[i], strides[i], itemsize, mem_addr_hit);
         if(sit == 0)
           break;
       }
       return sit;
     }
   }

  /**return view node ptr by mem address just hit, if sth is found; return nullptr if not.
   *
   * use binary search on the first (dim - 1) dimensions;
   * and mod the [(hit_address - starting_address_of_dim-1) / dtype_size] by stride[dim]
   * if the mod is 0 and the quodient is no greater than the sizes[dim]
   * then the tensor is hit
   * Note: more than one tensors from the top of the stack might be hit at a time,
   * but it's fine, we just return them all.
   *
   * @param memory_address: address visited on GPU
   * @return the tensor/view been accessed
   */
   std::vector<ViewNode*> get_view_nodes_by_mem_addr(u64 mem_addr_hit) {
     std::vector<torch_monitor_callback_tensor_data_t> possible = {};
     std::vector<torch_monitor_callback_tensor_data_t> sorted = {};
     std::vector<torch_monitor_callback_tensor_data_t> hit = {};
     std::vector<ViewNode*> res = {};
     torch_monitor_op_data_t op_info = _op_stack.top();
     torch_monitor_input_output_data_t inputs = op_info.input_output_data;
     for (int64_t i = 0; i < inputs.size; i++) {
       torch_monitor_callback_tensor_data_t _tensor = inputs.tensor_data[i];
       std::cout << "op stack tensor data ptr: " << (u64)_tensor.data_ptr << std::endl;
       if (_tensor.index == -1 || _tensor.numel <= 0)
         continue;
       data_ptr_t _tensor_data_ptr = reinterpret_cast<data_ptr_t>(_tensor.data_ptr);
       metadata_ptr_t _tensor_metadata_ptr = reinterpret_cast<metadata_ptr_t>(_tensor.metadata_ptr);
       ViewNode _input_temp = ViewNode{_tensor_data_ptr, _tensor_metadata_ptr, _tensor.numel, _tensor.itemsize};
       ViewNode* _input_root = find_root_node(_input_temp);
       if (!_input_root) {
//         std::cout << "No root---" << std::endl;
         continue;
       }
       ViewNode* _input_view = _input_root->find_node(_input_temp);
       if (!_input_view) {
//         std::cout << "No node---" << std::endl;
         continue;
       }
       // possible: hit-address roughly fall into view_node.mem_block_range
       if ((data_ptr_t)mem_addr_hit >= _input_view->mem_block_range.first && (data_ptr_t)mem_addr_hit <= _input_view->mem_block_range.second) {
         possible.push_back(_tensor);
       }
     }
     // sort possible tensor strides
     for(int i = 0; i < possible.size(); i++){
       sorted.push_back(sort_tensor_strides(possible[i]));
     }
     // find hit tensor
     for(int i = 0; i < sorted.size(); i++){
       uint64_t itemsize = sorted[i].itemsize;
       u64 sit = find_closest_starting_address(sorted[i], itemsize, mem_addr_hit);
       if(sit != 0){
//         if((mem_addr_hit - sit) / (sorted[i].strides[sorted[i].dim - 1] * itemsize) < sorted[i].sizes[sorted[i].dim - 1]
           if((mem_addr_hit == sit)
             ||
               ((mem_addr_hit - sit) < sorted[i].sizes[sorted[i].dim - 1] * (sorted[i].strides[sorted[i].dim - 1] * itemsize)
               &&
               (mem_addr_hit - sit) % (sorted[i].strides[sorted[i].dim - 1] * itemsize) == 0)){
           hit.push_back(sorted[i]);
         }
       }
     }
     // assemble the results
     for(int64_t i = 0; i < hit.size(); i++){
       data_ptr_t _hit_data_ptr = reinterpret_cast<data_ptr_t>(hit[i].data_ptr);
       metadata_ptr_t _hit_metadata_ptr = reinterpret_cast<metadata_ptr_t>(hit[i].metadata_ptr);
       ViewNode _input_temp = ViewNode{_hit_data_ptr, _hit_metadata_ptr, hit[i].numel, hit[i].itemsize};
       ViewNode* _input_root = find_root_node(_input_temp);
       ViewNode* _input_view = _input_root->find_node(_input_temp);
       res.push_back(_input_view);
     }
     std::cout << "possible size: " << possible.size() << std::endl;
     std::cout << "sorted size: " << sorted.size() << std::endl;
     std::cout << "hit size: " << hit.size() << std::endl;
     return res;
   }

 private:
   struct TorchViewTrace final : public Trace {
     // only need to know memory access, don't care read or write
     // here use memory range to loge access range but not allocation and sub-allocation
     // u64: Memory:Operation->op_id
     // don't care about read or write in this mode, just need to know access or not
     Map<u64, i32> access_memory; // map with sort but vector not

     TorchViewTrace() = default;

     virtual ~TorchViewTrace() {}
   };

   std::shared_ptr<TorchViewTrace> _trace;

   Map<u64, std::shared_ptr<Memory>> _memories;

   Map<u64, std::shared_ptr<Memory>> _current_memories;

   // <start_addr, memory_op_id>
   Map<u64, u64> _addresses_map;

   u64 _current_memory_usage = 0;  // to update _memory_peak
   u64 _memory_peak = 0;
   u64 _optimal_memory_peak = 0;
   u64 _memory_peak_kernel = 0;

   u64 _nums_cudamalloc = 0;
   u64 _nums_cudafree = 0;

   Map<u64, i32> _op_node;
   Map<u64, std::string> _op_type;

   Map<u64, u64> _accessed_memories;
   Map<u64, std::shared_ptr<Memory>> _submemories;

   Map<u64, std::shared_ptr<Memory>> _current_submemories;

   Map<u64, u64> _sub_addresses_map;

   Map<u64, Map<u64, Set<MemoryRange>>> _blank_chunks;

   u64 _current_submemory_usage = 0;  // to update _submemory_peak
   u64 _submemory_peak = 0;
   u64 _optimal_submemory_peak = 0;
   u64 _submemory_peak_kernel = 0;


  private:



   /**
    * Call Path from torch-monitor
    * TODO: find out how to invoke this function
    * @param op_id
    */
//   void update_torch_python_states(u64 op_id) {
//     size_t num_states = 0;
//     torch_monitor_python_state_get(MAX_NUM_STATES, python_states, &num_states);
//     Vector<PythonState> pstates;
//     for (size_t i = 0; i < num_states; ++i) {
//       pstates.push_back(
//         PythonState(
//           std::string(python_states[i].file_name),
//           std::string(python_states[i].function_name),
//           python_states[i].function_first_lineno,
//           python_states[i].lineno)
//       );
//     }
//     _torch_python_states.try_emplace(op_id, pstates);
//   }

   void update_op_node(u64 op_id, i32 ctx_id);

   void memory_op_callback(std::shared_ptr<Memory> op, bool is_submemory = false);

   void memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory = false);

   void kernel_op_callback(std::shared_ptr<Kernel> op);

   void memcpy_op_callback(std::shared_ptr<Memcpy> op);

   void memset_op_callback(std::shared_ptr<Memset> op);
 };  // TorchView

} // namespace redshow

#endif //REDSHOW_ANALYSIS_TORCH_VIEW_H
