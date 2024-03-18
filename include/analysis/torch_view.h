//
// Created by xjding on 1/1/24.
// TODO: Call path; Early stop for tensor strides sorting with the aid of transpose flag; GPA; Debugging
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
#include "c10/core/ScalarTypeToTypeMeta.h"

#include "/home/xjding/Projects/new_DrGPUM/DrGPUM/torch-monitor/include/torch_monitor.h"

#include <fstream>

const static size_t MAX_NUM_STATES = 30;

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
     int64_t index;
     int64_t numel;  // number of tensor elements
     int64_t dim;
     torch_monitor_scalar_type_t dtype;  // pytorch tensor.dtype (scalar type)
     int64_t item_size;
     int64_t storage_offset;
     int64_t sizes[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
     int64_t strides[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
     data_ptr_t data_ptr;
     metadata_ptr_t metadata_ptr;
     mem_range_t mem_block_range;  // <block start, block end>
     int _total_access = 0;
     std::vector<std::shared_ptr<ViewNode>> _children = {};

     public:
       /**temp object for searching*/
      ViewNode(data_ptr_t data_ptr, metadata_ptr_t metadata_ptr):data_ptr(data_ptr), metadata_ptr(metadata_ptr) {}

      /**for root nodes*/
      ViewNode(int64_t index, int64_t numel, int64_t dim, torch_monitor_scalar_type_t dtype, int64_t storage_offset,
               int64_t sizes[], int64_t strides[], data_ptr_t data_ptr, metadata_ptr_t metadata_ptr):
               index(index), numel(numel), dim(dim), dtype(dtype), storage_offset(storage_offset), data_ptr(data_ptr),
               metadata_ptr(metadata_ptr){
        this->item_size = c10::scalarTypeToTypeMeta(c10::ScalarType(dtype)).itemsize();
        for(int i = 0; i < std::min<int>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, dim); i ++){
          this->sizes[i] = sizes[i];
          this->strides[i] = strides[i];
        }
        this->mem_block_range = mem_range_t{data_ptr, data_ptr+(item_size * numel)};
      }

      /**other nodes*/
      ViewNode(int64_t index, int64_t numel, int64_t dim, torch_monitor_scalar_type_t dtype, int64_t storage_offset,
               int64_t sizes[], int64_t strides[], data_ptr_t data_ptr, metadata_ptr_t metadata_ptr,
               mem_range_t mem_block_range):
               index(index), numel(numel), dim(dim), dtype(dtype), storage_offset(storage_offset), data_ptr(data_ptr),
               metadata_ptr(metadata_ptr){
        this->item_size = c10::scalarTypeToTypeMeta(c10::ScalarType(dtype)).itemsize();
        this->mem_block_range = mem_range_t{mem_block_range};
        for(int i = 0; i < std::min<int>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, dim); i ++){
          this->sizes[i] = sizes[i];
          this->strides[i] = strides[i];
        }
      }

      bool operator==(const ViewNode& r_val) {
        return ((this->data_ptr == r_val.data_ptr) && (this->metadata_ptr == r_val.metadata_ptr));
      }

      /**
       * recursive member function of view node object
       * return node ptr if the view sits on this branch
       * return null if the view sits nowhere on this branch
       * */
      std::shared_ptr<ViewNode> find_node(ViewNode r_node) {
        if (*(this) == r_node)
          return (std::shared_ptr<ViewNode>) this;
        else if (!this->_children.empty()) {
          std::shared_ptr<ViewNode> res = std::shared_ptr<ViewNode>();
          for (auto node : this->_children) {
            res = node->find_node(r_node);
            if (!res)
              return res;  // return the node ptr
          }
          return res;  // not found return nullptr
        } else {
          return std::shared_ptr<ViewNode>();  // this._children is empty
        }
      }

      bool is_leaf_node() {
        return this->_children.empty();
      }

      void delete_children_nodes() {
        if (! this->is_leaf_node()) {
          for (auto child : this->_children) {
            child->delete_children_nodes();
          }
          this->_children.clear();
        }
      }
   };

   std::vector<std::shared_ptr<ViewNode>> _roots = {};  // The node forest //TODO: optimize

   // push <op_name, visible inputs(tensors)> while domain enter
   // pop op_name while domain exit
   std::stack<torch_monitor_op_data_t> _op_stack = std::stack<torch_monitor_op_data_t>();
   torch_monitor_op_data_t _popped_op = torch_monitor_op_data_t();

   /** find view root from the forest */
   std::shared_ptr<ViewNode> find_root_node(ViewNode r_node) {
     for (auto node : _roots){
       if (r_node.data_ptr >= node->mem_block_range.first && r_node.data_ptr <= node->mem_block_range.second){
         return node;
       }
     }
       return std::shared_ptr<ViewNode>();
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
   void update_view_forest(torch_monitor_callback_tensor_data_t& tensor_data) {
     data_ptr_t data_ptr = reinterpret_cast<data_ptr_t>(tensor_data.data_ptr);
     metadata_ptr_t metadata_ptr = reinterpret_cast<metadata_ptr_t>(tensor_data.metadata_ptr);
     ViewNode temp = ViewNode{data_ptr, metadata_ptr};
     std::shared_ptr<ViewNode> root_found = find_root_node(temp);
     if(root_found){
       std::shared_ptr<ViewNode> view_existed = root_found->find_node(temp);
       if(!view_existed){
           /** insert new view node into correct "view_node._children"
            * 1. find the father node from the root/branch
            * 2. create a new node and insert into its "_children"
            * */
           // fetch the closest pytorch callback op inputs from the stack
         for (int64_t i = 0; i < _popped_op.input_output_data.size; i++) {
           torch_monitor_callback_tensor_data_t _tensor = _popped_op.input_output_data.tensor_data[i];
           if (_tensor.index == -1 || _tensor.numel <= 0)
               continue;
           data_ptr_t _tensor_data_ptr = reinterpret_cast<data_ptr_t>(_tensor.data_ptr);
           metadata_ptr_t _tensor_metadata_ptr = reinterpret_cast<metadata_ptr_t>(_tensor.metadata_ptr);
           ViewNode _input_temp = ViewNode{_tensor_data_ptr, _tensor_metadata_ptr};
           std::shared_ptr<ViewNode> _input_root = find_root_node(_input_temp);
           std::shared_ptr<ViewNode> _input_view = _input_root->find_node(_input_temp);
           if ((data_ptr_t)tensor_data.data_ptr >= _input_view->mem_block_range.first && (data_ptr_t)tensor_data.data_ptr <= _input_view->mem_block_range.second) {
               // add the tensor_data to _tensor(node)'s children
             std::shared_ptr<ViewNode> _node = std::make_shared<ViewNode>(tensor_data.index, tensor_data.numel, tensor_data.dim, tensor_data.dtype,
                                                                          tensor_data.storage_offset, tensor_data.sizes, tensor_data.strides, data_ptr,
                                                                          metadata_ptr, _input_view->mem_block_range);
             _input_view->_children.push_back(_node);
           }
         }
       }
     } else {
         ViewNode _node = ViewNode{tensor_data.index, tensor_data.numel, tensor_data.dim, tensor_data.dtype,
                                   tensor_data.storage_offset, tensor_data.sizes, tensor_data.strides, data_ptr,
                                   metadata_ptr};
         std::shared_ptr<ViewNode> _node_ptr = std::make_shared<ViewNode>(_node);
         _roots.push_back(_node_ptr);  // add a new root
       }
   }

   /**
    * delete a tree(gaven a tensor ptr) from the forest
    *@param mem_start_addr: starting address of memory range
    */
   void delete_forest_tree(data_ptr_t mem_start_addr) {
     for (int i =0; i < _roots.size(); i++) {
       if (_roots.at(i)->mem_block_range.first == mem_start_addr){
         _roots.at(i)->delete_children_nodes();
         _roots.erase(_roots.begin()+i);
         break;
       }
     }
   }

   /**
    * add view_node _total_access by one
    * @param a list of view_nodes
    * TODO: call this func at project stage 3, (use data_ptr, offset, stride, item_size to spot the view_node which needs update)
    */
   void update_node__total_access(std::vector<std::shared_ptr<ViewNode>> view_nodes) {
     for (auto viter : view_nodes) {
       std::shared_ptr<ViewNode> _root_node = find_root_node(*(viter.get()));
       std::shared_ptr<ViewNode> _node = _root_node->find_node(*(viter.get()));
       _node->_total_access++;
     }
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
    * @param item_size sizeof(tensor.dtype)
    * @param mem_addr_hit
    * @return
    */
   u64 find_local_starting_address(u64 init_address, int64_t size, int64_t stride, int64_t item_size, u64 mem_addr_hit) {
     if (size >= 1){
       u64 sit = init_address + ((size-1) / 2) * item_size * stride;
       if (sit == mem_addr_hit) {
         return sit;
       } else if (sit > mem_addr_hit){
         return find_local_starting_address(init_address, ((size-1) / 2), stride, item_size, mem_addr_hit);
       } else { // sit < mem_addr_hit
         if ((sit + item_size * stride) > mem_addr_hit) {
           return sit;
         } else {
           return find_local_starting_address((sit + item_size * stride), (size - (size / 2)), stride, item_size, mem_addr_hit);
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
    * @param item_size: sizeof(tensor.dtype) c10::scalarTypeToTypeMeta(c10::ScalarType(dtype)).itemsize();
    * @return the address of
    */
   u64 find_closest_starting_address(torch_monitor_callback_tensor_data_t temp_tensor, int64_t item_size, u64 mem_addr_hit){
     int64_t * sizes = temp_tensor.sizes;
     int64_t * strides = temp_tensor.strides;
     int64_t dim = temp_tensor.dim;
     if (dim <= 1)
       return (u64)temp_tensor.data_ptr;
     else {
       u64 sit = (u64)temp_tensor.data_ptr;
       for(int64_t i = 0; i < (dim-1); i++) {
         sit = find_local_starting_address(sit, sizes[i], strides[i], item_size, mem_addr_hit);
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
   std::vector<std::shared_ptr<ViewNode>> get_view_nodes_by_mem_addr(u64 mem_addr_hit) {
     std::vector<torch_monitor_callback_tensor_data_t> possible = {};
     std::vector<torch_monitor_callback_tensor_data_t> sorted = {};
     std::vector<torch_monitor_callback_tensor_data_t> hit = {};
     std::vector<std::shared_ptr<ViewNode>> res = {};
     torch_monitor_op_data_t op_info = _op_stack.top();
     torch_monitor_input_output_data_t inputs = op_info.input_output_data;
     for (int64_t i = 0; i < inputs.size; i++) {
       torch_monitor_callback_tensor_data_t _tensor = inputs.tensor_data[i];
       if (_tensor.index == -1 || _tensor.numel <= 0)
         continue;
       data_ptr_t _tensor_data_ptr = reinterpret_cast<data_ptr_t>(_tensor.data_ptr);
       metadata_ptr_t _tensor_metadata_ptr = reinterpret_cast<metadata_ptr_t>(_tensor.metadata_ptr);
       ViewNode _input_temp = ViewNode{_tensor_data_ptr, _tensor_metadata_ptr};
       std::shared_ptr<ViewNode> _input_root = find_root_node(_input_temp);
       std::shared_ptr<ViewNode> _input_view = _input_root->find_node(_input_temp);
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
       int64_t item_size = c10::scalarTypeToTypeMeta(c10::ScalarType(sorted[i].dtype)).itemsize();
       u64 sit = find_closest_starting_address(sorted[i], item_size, mem_addr_hit);
       if(sit != 0){
         if((mem_addr_hit - sit) / (sorted[i].strides[sorted[i].dim - 1] * item_size) < sorted[i].sizes[sorted[i].dim - 1]
           &&
           (mem_addr_hit - sit) % (sorted[i].strides[sorted[i].dim - 1] * item_size) == 0){
           hit.push_back(sorted[i]);
         }
       }
     }
     // assemble the results
     for(int64_t i = 0; i < hit.size(); i++){
       data_ptr_t _hit_data_ptr = reinterpret_cast<data_ptr_t>(hit[i].data_ptr);
       metadata_ptr_t _hit_metadata_ptr = reinterpret_cast<metadata_ptr_t>(hit[i].metadata_ptr);
       ViewNode _input_temp = ViewNode{_hit_data_ptr, _hit_metadata_ptr};
       std::shared_ptr<ViewNode> _input_root = find_root_node(_input_temp);
       std::shared_ptr<ViewNode> _input_view = _input_root->find_node(_input_temp);
       res.push_back(_input_view);
     }
     return res;
   }

 private:
   struct TorchViewTrace final : public Trace {
     // only need to know memory access, don't care read or write
     // here use memory range to loge access range but not allocation and sub-allocation

     // u64: Memory:Operation->op_id
     // @Lin-Mao: don't care about read or write in this mode, just need to know access or not
     Map<u64, bool> access_memory; // map with sort but vector not

     Map<u64, bool> access_submemory;

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
