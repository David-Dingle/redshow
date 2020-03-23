//
// Created by find on 19-7-1.
//

#ifndef REDSHOW_COMMON_LIB_H
#define REDSHOW_COMMON_LIB_H

#include <string>
#include <regex>
#include <list>
#include <map>
#include <numeric>
#include <fstream>
#include <set>
#include <regex>
#include <queue>

#include <tuple>
#include <algorithm>

#include "instruction.h"
#include "utils.h"
#include "redshow.h"

/*
 * Data type definition
 */

struct ThreadId {
  u32 flat_block_id;
  u32 flat_thread_id;

  bool operator<(const ThreadId &o) const {
    return (this->flat_block_id < o.flat_block_id) ||
           (this->flat_block_id == o.flat_block_id && this->flat_thread_id < o.flat_thread_id);
  }

  bool operator==(const ThreadId &o) const {
    return this->flat_thread_id == o.flat_thread_id &&
           this->flat_block_id == o.flat_block_id;
  }
};

// {<memory_op_id, AccessKind::DataType> : {pc: {value: count}}}
typedef std::map<std::pair<u64, AccessKind>, std::map<u64, std::map<u64, u64>>> SpatialTrace;

// {memory_op_id: {value: count}}
typedef std::map<std::pair<u64, AccessKind>, std::map<u64, u64>> SpatialStatistic;

// {ThreadId : {address : {<pc, value>}}}
typedef std::map<ThreadId, std::map<u64, std::pair<u64, u64>>> TemporalTrace;

// {pc1 : {pc2 : {<value, AccessKind::DataType> : count}}}
typedef std::map<u64, std::map<u64, std::map<std::pair<u64, AccessKind>, u64>>> PCPairs;
// {pc: access_sum_count}
typedef std::map<u64, u64> PCAccessSum;
// <pc_from, pc_to, value, AccessKind, count>
typedef std::tuple<u64, u64, u64, AccessKind, u64> TopPair;

struct CompareView {
  bool operator()(redshow_record_view_t const &d1, redshow_record_view_t const &d2) {
    return d1.count > d2.count;
  }
};

struct CompareStatistic {
  bool operator()(std::tuple<u64, u64, AccessKind> const &d1, std::tuple<u64, u64, AccessKind> const &d2) {
    return std::get<1>(d1) > std::get<1>(d2);
  }
};

struct CompareTopPairs {
  bool operator()(TopPair const &d1, TopPair const &d2) {
    return std::get<4>(d1) > std::get<4>(d2);
  }
};

typedef std::priority_queue<redshow_record_view_t, std::vector<redshow_record_view_t>, CompareView> TopViews;
// {value, count, AccessKind}
typedef std::priority_queue<std::tuple<u64, u64, AccessKind>,
        std::vector<std::tuple<u64, u64, AccessKind>>, CompareStatistic> TopStatistic;
// <pc_from, pc_to, value, AccessKind, count>
typedef std::priority_queue<TopPair, std::vector<TopPair>, CompareTopPairs> TopPairs;

/*
 * Interface:
 *
 * Each analysis type has three methods
 * 1. get_<analysis_type>
 * Analyze a segment of trace
 *
 * 2. record_<analysis_type>
 * Attribute high level analysis result to runtime
 *
 * 3. show_<analysis_type>
 * Debug/output detailed analysis result to file
 */

/*
 * Analyze temporal trace
 *
 * pc:
 * Current record's pc
 *
 * tid:
 * Current record's global thread index
 *
 * addr:
 * Current record's address
 *
 * access_type:
 * How a thread accesses memory (e.g. float/int, vector/scalar)
 *
 * temporal_trace:
 * {ThreadId : {address : {<pc, value>}}}
 *
 * pc_pairs:
 * {pc1 : {pc2 : {<value, type>}}}
 */
void get_temporal_trace(u64 pc, ThreadId tid, u64 addr, u64 value, AccessKind access_type,
                        TemporalTrace &temporal_trace, PCPairs &pc_pairs);

/*
 * Record frequent temporal records
 *
 * pc_pairs:
 * {pc1 : {pc2 : {<value, type>}}}
 *
 * record_data:
 * Data returned to the runtime
 *
 * num_views_limit:
 * Number of entries the runtime needs to know
 */
void record_temporal_trace(PCPairs &pc_pairs, PCAccessSum &pc_access_sum, redshow_record_data_t &record_data,
                           uint32_t num_views_limit, uint64_t &kernel_red_count, uint64_t &kernel_count);

void show_temporal_trace(u64 kernel_id, PCPairs &pc_pairs, PCAccessSum &pc_access_sum, bool is_read,
                         uint32_t num_views_limit, uint32_t thread_id, uint64_t &kernel_red_count,
                         uint64_t &kernel_count);

/*
 * Analyze spatial trace
 *
 * pc:
 * Current record's pc
 *
 * memory_op_id:
 * Current record's memory identifier
 *
 * access_type:
 * How a thread accesses memory (e.g. float/int, vector/scalar)
 *
 * spatial_trace:
 * {<memory_op_id, AccessKind::DataType> : {pc: {value: count}}}
 *
 */
void get_spatial_trace(u64 pc, u64 value, u64 memory_op_id, AccessKind access_type,
                       SpatialTrace &spatial_trace);

/*
 * Record frequent spatial records
 *
 * spatial_trace:
 * {<memory_op_id, AccessKind::DataType> : {pc: {value: count}}}
 *
 * record_data:
 * Data returned to the runtime
 *
 * num_views_limit:
 * Number of entries the runtime needs to know
 */
void record_spatial_trace(SpatialTrace &spatial_trace,
                          redshow_record_data_t &record_data, uint32_t num_views_limit,
                          SpatialStatistic &spatial_statistic, SpatialStatistic &thread_spatial_statistic);

/**
 * Write array's value statistic data into files.
 * @arg thread_id: cpu thread id
 * @arg spatial_statistic: {memory_op_id: {value: count}}
 * @arg num_write_limit: numer of entries will be written into files.
 * @arg is_read: the spatial_statistic is for reading or writing accesses.
 * */
void show_spatial_trace(uint32_t thread_id, uint64_t kernel_id, SpatialStatistic &spatial_statistic,
                        uint32_t num_write_limit, bool is_read, bool is_kernel);

/**
 * Use decimal_degree_f32 bits to cut the valid floating number bits.
 * @arg decimal_degree_f32: the valid bits. The floating numbers have 23-bit fractions.
 * */
u64 store2float(u64 a, int decimal_degree_f32);

/**
 * @arg decimal_degree_f64: the valid bits. The float64 numbers have 52-bit fractions.
 * */
u64 store2double(u64 a, int decimal_degree_f64);

/**
 * Change raw data to formatted value.
 * */
u64 store2basictype(u64 a, AccessKind akind, int decimal_degree_f32, int decimal_degree_f64);

void output_corresponding_type_value(u64 a, AccessKind akind, std::streambuf *buf, bool is_signed);

std::string combine_type_unitsize(AccessKind akind);

#endif  // REDSHOW_COMMON_LIB_H


