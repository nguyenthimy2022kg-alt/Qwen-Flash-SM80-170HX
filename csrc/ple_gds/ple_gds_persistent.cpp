#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda.h>
#include <cufile.h>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Range {
    std::uint64_t file_offset;
    std::uint64_t size;
    std::uint64_t staging_offset;
};

struct IdBuffer {
    const void* data = nullptr;
    std::size_t count = 0;
    std::size_t item_size = 0;

    std::int64_t at(std::size_t index) const {
        if (item_size == sizeof(std::int32_t)) {
            return static_cast<const std::int32_t*>(data)[index];
        }
        return static_cast<const std::int64_t*>(data)[index];
    }
};

double seconds_since(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_cuda(CUresult value, const char* operation) {
    if (value == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(value, &name);
    cuGetErrorString(value, &text);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA_ERROR") +
                             " (" + (text ? text : "unknown") + ")");
}

void check_cufile(CUfileError_t value, const char* operation) {
    if (value.err == CU_FILE_SUCCESS) return;
    throw std::runtime_error(std::string(operation) + ": " +
                             cufileop_status_error(static_cast<CUfileOpError>(std::abs(value.err))));
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string normalize_bdf(const std::string& value) {
    unsigned int domain = 0, bus = 0, device = 0, function = 0;
    if (std::sscanf(value.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
        throw std::runtime_error("invalid GPU BDF: " + value);
    }
    char result[32];
    std::snprintf(result, sizeof(result), "%04x:%02x:%02x.%x", domain, bus, device, function);
    return result;
}

std::string format_uuid(const CUuuid& uuid) {
    const auto* b = reinterpret_cast<const unsigned char*>(uuid.bytes);
    char result[64];
    std::snprintf(result, sizeof(result),
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
                  b[10], b[11], b[12], b[13], b[14], b[15]);
    return result;
}

constexpr const char* kGatherPtx = R"PTX(
.version 7.0
.target sm_80
.address_size 64

.visible .entry gather_spans(
    .param .u64 p_staging,
    .param .u64 p_output,
    .param .u64 p_sources,
    .param .u64 p_destinations,
    .param .u64 p_lengths,
    .param .u32 p_count)
{
    .reg .pred %p<3>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<20>;
    ld.param.u64 %rd1, [p_staging];
    ld.param.u64 %rd2, [p_output];
    ld.param.u64 %rd3, [p_sources];
    ld.param.u64 %rd4, [p_destinations];
    ld.param.u64 %rd5, [p_lengths];
    ld.param.u32 %r1, [p_count];
    mov.u32 %r2, %ctaid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra GATHER_DONE;
    mul.wide.u32 %rd6, %r2, 8;
    add.u64 %rd7, %rd3, %rd6;
    add.u64 %rd8, %rd4, %rd6;
    add.u64 %rd9, %rd5, %rd6;
    ld.global.u64 %rd10, [%rd7];
    ld.global.u64 %rd11, [%rd8];
    ld.global.u64 %rd12, [%rd9];
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd13, %r3;
    mov.u32 %r4, %ntid.x;
    cvt.u64.u32 %rd14, %r4;
GATHER_LOOP:
    setp.ge.u64 %p2, %rd13, %rd12;
    @%p2 bra GATHER_DONE;
    add.u64 %rd15, %rd10, %rd13;
    add.u64 %rd16, %rd11, %rd13;
    add.u64 %rd17, %rd1, %rd15;
    add.u64 %rd18, %rd2, %rd16;
    ld.global.u8 %r5, [%rd17];
    st.global.u8 [%rd18], %r5;
    add.u64 %rd13, %rd13, %rd14;
    bra GATHER_LOOP;
GATHER_DONE:
    ret;
}
)PTX";

struct ContextGuard {
    explicit ContextGuard(CUcontext context) { check_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent"); }
    ~ContextGuard() {
        CUcontext ignored = nullptr;
        cuCtxPopCurrent(&ignored);
    }
};

struct Slot {
    CUdeviceptr staging = 0;
    CUdeviceptr output = 0;
    std::uint64_t staging_capacity = 0;
    std::uint64_t output_capacity = 0;
    CUdeviceptr sources = 0;
    CUdeviceptr destinations = 0;
    CUdeviceptr lengths = 0;
    CUfileBatchHandle_t batch = nullptr;
    CUfileHandle_t batch_file_handle = nullptr;
    CUevent gather_start = nullptr;
    CUevent gather_end = nullptr;
    CUevent consumer_done = nullptr;
    std::size_t registered_lanes = 0;
    bool released = true;
    bool consumer_event_recorded = false;
    bool gather_event_recorded = false;
    void* host_metadata_workspace = nullptr;
    std::size_t host_metadata_workspace_bytes = 0;
    std::uint64_t* host_sources = nullptr;
    std::uint64_t* host_destinations = nullptr;
    std::uint64_t* host_lengths = nullptr;
    std::vector<CUfileIOParams_t> batch_params;
    std::vector<std::uint64_t> batch_expected;
    std::vector<CUfileIOEvents_t> batch_events;
    std::vector<unsigned char> batch_completed;
    std::vector<std::uint64_t> plan_pages;
    std::vector<std::uint64_t> plan_gather_offsets;
    std::vector<Range> plan_ranges;
    std::vector<std::size_t> plan_range_order;
    std::vector<std::uint64_t> plan_lane_loads;
    std::vector<std::size_t> async_sizes;
    std::vector<off_t> async_file_offsets;
    std::vector<off_t> async_buffer_offsets;
    std::vector<ssize_t> async_results;
    CUstream async_stream = nullptr;
    bool async_stream_registered = false;
    std::mutex mutex;
};

struct ReadMetrics {
    std::uint64_t bytes_read = 0;
    std::uint64_t io_count = 0;
    std::uint64_t batch_chunks = 0;
    double slot_wait_seconds = 0.0;
    double submission_seconds = 0.0;
    double io_wait_seconds = 0.0;
    double metadata_seconds = 0.0;
    double gather_launch_seconds = 0.0;
    double gather_gpu_seconds = 0.0;
    double native_seconds = 0.0;
    double planning_seconds = 0.0;
    std::uint64_t requested_count = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t unique_page_count = 0;
};

class PersistentState {
public:
    PersistentState(const std::string& path, const std::vector<std::uint64_t>& staging_ptrs,
                    std::uint64_t staging_capacity, const std::vector<std::uint64_t>& output_ptrs,
                    std::uint64_t output_capacity, const std::string& expected_uuid,
                    const std::string& expected_bdf, std::uint64_t max_spans,
                    unsigned int requested_batch_capacity, unsigned int worker_count,
                    bool pin_workers, bool poll_mode)
        : path_(path), expected_uuid_(expected_uuid), expected_bdf_(normalize_bdf(expected_bdf)),
          max_spans_(max_spans), requested_worker_count_(worker_count),
          pin_workers_(pin_workers), poll_mode_(poll_mode) {
        if (staging_ptrs.empty() || staging_ptrs.size() != output_ptrs.size()) {
            throw std::runtime_error("staging/output slot counts must match and be nonzero");
        }
        if (!staging_capacity || !output_capacity || !max_spans) {
            throw std::runtime_error("buffer capacities and max_spans must be positive");
        }
        if (!worker_count || worker_count > 128) {
            throw std::runtime_error("worker_count must be in [1, 128]");
        }
        lane_capacity_ = staging_capacity / worker_count / 4096 * 4096;
        if (!lane_capacity_) {
            throw std::runtime_error("staging slot is too small for one aligned lane per worker");
        }
        check_cuda(cuInit(0), "cuInit");
        int count = 0;
        check_cuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
        bool found = false;
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            CUdevice candidate = 0;
            CUuuid uuid{};
            char bdf[32]{};
            check_cuda(cuDeviceGet(&candidate, ordinal), "cuDeviceGet");
            check_cuda(cuDeviceGetUuid(&uuid, candidate), "cuDeviceGetUuid");
            check_cuda(cuDeviceGetPCIBusId(bdf, sizeof(bdf), candidate), "cuDeviceGetPCIBusId");
            if (lower(format_uuid(uuid)) == lower(expected_uuid_) && normalize_bdf(bdf) == expected_bdf_) {
                device_ = candidate;
                device_ordinal_ = ordinal;
                gpu_uuid_ = format_uuid(uuid);
                gpu_bdf_ = normalize_bdf(bdf);
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("expected GPU UUID/BDF was not found among visible CUDA devices");
        check_cuda(cuCtxGetCurrent(&context_), "cuCtxGetCurrent");
        if (!context_) throw std::runtime_error("no current PyTorch CUDA context");
        CUdevice context_device = 0;
        check_cuda(cuCtxGetDevice(&context_device), "cuCtxGetDevice");
        if (context_device != device_) {
            throw std::runtime_error("current PyTorch context is not on the expected GPU");
        }
        try {
            ContextGuard guard(context_);
            fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd_ < 0) {
                throw std::runtime_error("open(O_DIRECT) failed: " + std::string(std::strerror(errno)));
            }
            file_open_count_ = 1;
            check_cufile(cuFileDriverOpen(), "cuFileDriverOpen");
            driver_open_ = true;
            driver_open_count_ = 1;
            check_cufile(cuFileDriverSetPollMode(poll_mode_, 4), "cuFileDriverSetPollMode");
            CUfileDrvProps_t properties{};
            check_cufile(cuFileDriverGetProperties(&properties), "cuFileDriverGetProperties");
            driver_control_flags_ = properties.nvfs.dcontrolflags;
            if (driver_control_flags_ & (1U << CU_FILE_ALLOW_COMPAT_MODE)) {
                throw std::runtime_error("cuFile compatibility mode is enabled");
            }
            const unsigned int driver_limit = properties.max_batch_io_size ? properties.max_batch_io_size : 128;
            batch_capacity_ = std::min({128U, driver_limit, requested_batch_capacity});
            if (!batch_capacity_) throw std::runtime_error("cuFile batch capacity is zero");
            CUfileDescr_t descriptor{};
            descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            descriptor.handle.fd = fd_;
            check_cufile(cuFileHandleRegister(&file_handle_, &descriptor), "cuFileHandleRegister");
            handle_registration_count_ = 1;
            check_cuda(cuModuleLoadData(&module_, kGatherPtx), "cuModuleLoadData(gather PTX)");
            module_load_count_ = 1;
            check_cuda(cuModuleGetFunction(&gather_, module_, "gather_spans"), "cuModuleGetFunction(gather)");
            const std::size_t metadata_workspace_bytes = metadata_workspace_bytes_for(max_spans_);
            for (std::size_t index = 0; index < staging_ptrs.size(); ++index) {
                auto slot = std::make_unique<Slot>();
                slot->staging = static_cast<CUdeviceptr>(staging_ptrs[index]);
                slot->output = static_cast<CUdeviceptr>(output_ptrs[index]);
                slot->staging_capacity = staging_capacity;
                slot->output_capacity = output_capacity;
                slots_.push_back(std::move(slot));
                Slot& managed = *slots_.back();
                verify_pointer(managed.staging, staging_capacity, "staging");
                verify_pointer(managed.output, output_capacity, "output");
                for (unsigned int lane = 0; lane < requested_worker_count_; ++lane) {
                    const CUdeviceptr lane_base = managed.staging + lane * lane_capacity_;
                    check_cufile(cuFileBufRegister(reinterpret_cast<void*>(lane_base), lane_capacity_, 0),
                                 "cuFileBufRegister(torch worker lane)");
                    ++managed.registered_lanes;
                    ++buffer_registration_count_;
                }
                check_cuda(cuMemAlloc(&managed.sources, max_spans * sizeof(std::uint64_t)), "cuMemAlloc(sources)");
                ++metadata_allocation_count_;
                check_cuda(cuMemAlloc(&managed.destinations, max_spans * sizeof(std::uint64_t)),
                           "cuMemAlloc(destinations)");
                ++metadata_allocation_count_;
                check_cuda(cuMemAlloc(&managed.lengths, max_spans * sizeof(std::uint64_t)), "cuMemAlloc(lengths)");
                ++metadata_allocation_count_;
                check_cufile(cuFileBatchIOSetUp(&managed.batch, batch_capacity_), "cuFileBatchIOSetUp");
                ++batch_setup_count_;
                check_cuda(cuMemHostAlloc(&managed.host_metadata_workspace, metadata_workspace_bytes, 0),
                           "cuMemHostAlloc(metadata workspace)");
                managed.host_metadata_workspace_bytes = metadata_workspace_bytes;
                auto* metadata_words = static_cast<std::uint64_t*>(managed.host_metadata_workspace);
                managed.host_sources = metadata_words;
                managed.host_destinations = metadata_words + max_spans_;
                managed.host_lengths = metadata_words + 2 * max_spans_;
                ++pinned_metadata_allocation_count_;
                pinned_metadata_allocation_bytes_ += metadata_workspace_bytes;
                managed.batch_params.resize(batch_capacity_);
                managed.batch_expected.resize(batch_capacity_);
                managed.batch_events.resize(batch_capacity_);
                managed.batch_completed.resize(batch_capacity_);
                managed.plan_pages.reserve(max_spans_ * 2);
                managed.plan_gather_offsets.reserve(max_spans_);
                managed.plan_ranges.reserve(max_spans_ * 2);
                managed.plan_range_order.reserve(max_spans_ * 2);
                managed.plan_lane_loads.resize(requested_worker_count_);
                managed.async_sizes.resize(max_spans_ * 2);
                managed.async_file_offsets.resize(max_spans_ * 2);
                managed.async_buffer_offsets.resize(max_spans_ * 2);
                managed.async_results.resize(max_spans_ * 2);
                host_workspace_allocation_count_ += 16;
                check_cuda(cuEventCreate(&managed.gather_start, CU_EVENT_DEFAULT), "cuEventCreate(gather_start)");
                check_cuda(cuEventCreate(&managed.gather_end, CU_EVENT_DEFAULT), "cuEventCreate(gather_end)");
                check_cuda(cuEventCreate(&managed.consumer_done, CU_EVENT_DISABLE_TIMING),
                           "cuEventCreate(consumer_done)");
            }
            start_workers();
            for (std::size_t index = 0; index < slots_.size(); ++index) {
                slots_[index]->batch_file_handle = workers_[index % workers_.size()]->file_handle;
            }
            initialization_count_ = 1;
        } catch (...) {
            close_noexcept();
            throw;
        }
    }

    ~PersistentState() { close_noexcept(); }

    ReadMetrics read_ids(std::size_t slot_index, const IdBuffer& ids, std::uint64_t row_count,
                         std::uint64_t base_offset, std::uint64_t block_rows,
                         std::uint64_t block_stride, std::uint64_t row_bytes,
                         std::uint64_t alignment, std::uint64_t max_gap_pages,
                         std::uint64_t max_range_bytes, std::uint64_t output_bytes,
                         std::uint64_t output_offset_bytes, CUstream stream,
                         const std::string& mode, bool measure_gpu) {
        if (slot_index >= slots_.size()) throw std::runtime_error("slot index is out of range");
        if (!ids.count) throw std::runtime_error("read request must contain at least one row");
        if (ids.count > max_spans_) throw std::runtime_error("row ID capacity exceeded");
        if (!row_count || !block_rows || !row_bytes || !alignment || alignment != 4096) {
            throw std::runtime_error("invalid artifact planning geometry");
        }
        if (output_bytes != ids.count * row_bytes) {
            throw std::runtime_error("requested output byte count mismatch");
        }
        if (!max_range_bytes) max_range_bytes = std::numeric_limits<std::uint64_t>::max();
        if (max_range_bytes < alignment) {
            throw std::runtime_error("max range size must be at least one aligned page");
        }
        Slot& slot = *slots_[slot_index];
        const auto planning_start = Clock::now();
        slot.plan_pages.clear();
        slot.plan_gather_offsets.clear();
        slot.plan_ranges.clear();
        slot.plan_range_order.clear();
        if (slot.plan_pages.capacity() < ids.count * 2 ||
            slot.plan_gather_offsets.capacity() < ids.count) {
            throw std::runtime_error("fixed planning workspace capacity exceeded");
        }

        auto absolute_offset = [&](std::uint64_t row_id) {
            const std::uint64_t block = row_id / block_rows;
            const std::uint64_t within = row_id % block_rows;
            if (block > (std::numeric_limits<std::uint64_t>::max() - base_offset) / block_stride) {
                throw std::runtime_error("row offset overflow");
            }
            const std::uint64_t block_offset = base_offset + block * block_stride;
            if (within > (std::numeric_limits<std::uint64_t>::max() - block_offset) / row_bytes) {
                throw std::runtime_error("row offset overflow");
            }
            return block_offset + within * row_bytes;
        };

        for (std::size_t index = 0; index < ids.count; ++index) {
            const std::int64_t signed_id = ids.at(index);
            if (signed_id < 0 || static_cast<std::uint64_t>(signed_id) >= row_count) {
                throw std::runtime_error("row ID is outside artifact bounds");
            }
            const std::uint64_t offset = absolute_offset(static_cast<std::uint64_t>(signed_id));
            const std::uint64_t first_page = offset / alignment;
            const std::uint64_t last_page = (offset + row_bytes - 1) / alignment;
            slot.plan_pages.push_back(first_page);
            if (last_page != first_page) slot.plan_pages.push_back(last_page);
        }
        std::sort(slot.plan_pages.begin(), slot.plan_pages.end());
        slot.plan_pages.erase(std::unique(slot.plan_pages.begin(), slot.plan_pages.end()),
                              slot.plan_pages.end());

        std::uint64_t staging_cursor = 0;
        const std::uint64_t effective_max_range = std::min(max_range_bytes, lane_capacity_);
        for (const std::uint64_t page : slot.plan_pages) {
            if (!slot.plan_ranges.empty()) {
                Range& previous = slot.plan_ranges.back();
                const std::uint64_t previous_first = previous.file_offset / alignment;
                const std::uint64_t previous_last = previous_first + previous.size / alignment - 1;
                const std::uint64_t gap = page - previous_last - 1;
                const std::uint64_t extended_size = (page - previous_first + 1) * alignment;
                if (gap <= max_gap_pages && extended_size <= effective_max_range) {
                    staging_cursor += extended_size - previous.size;
                    previous.size = extended_size;
                    continue;
                }
            }
            slot.plan_ranges.push_back(Range{page * alignment, alignment, staging_cursor});
            staging_cursor += alignment;
        }
        if (staging_cursor > slot.staging_capacity) {
            throw std::runtime_error("planned pages exceed fixed staging slot");
        }

        std::fill(slot.plan_lane_loads.begin(), slot.plan_lane_loads.end(), 0);
        for (std::size_t index = 0; index < slot.plan_ranges.size(); ++index) {
            slot.plan_range_order.push_back(index);
        }
        std::sort(slot.plan_range_order.begin(), slot.plan_range_order.end(),
                  [&slot](std::size_t left, std::size_t right) {
                      const Range& lhs = slot.plan_ranges[left];
                      const Range& rhs = slot.plan_ranges[right];
                      return lhs.size != rhs.size ? lhs.size > rhs.size
                                                  : lhs.file_offset < rhs.file_offset;
                  });
        for (const std::size_t range_index : slot.plan_range_order) {
            Range& range = slot.plan_ranges[range_index];
            std::size_t selected = slot.plan_lane_loads.size();
            for (std::size_t lane = 0; lane < slot.plan_lane_loads.size(); ++lane) {
                if (slot.plan_lane_loads[lane] + range.size <= lane_capacity_ &&
                    (selected == slot.plan_lane_loads.size() ||
                     slot.plan_lane_loads[lane] < slot.plan_lane_loads[selected])) {
                    selected = lane;
                }
            }
            if (selected == slot.plan_lane_loads.size()) {
                throw std::runtime_error("planned pages cannot fit fixed registered worker lanes");
            }
            range.staging_offset = selected * lane_capacity_ + slot.plan_lane_loads[selected];
            slot.plan_lane_loads[selected] += range.size;
        }

        for (std::size_t index = 0; index < ids.count; ++index) {
            const std::uint64_t offset = absolute_offset(static_cast<std::uint64_t>(ids.at(index)));
            const std::uint64_t page = offset / alignment;
            const auto range_it = std::upper_bound(
                slot.plan_ranges.begin(), slot.plan_ranges.end(), page,
                [alignment](std::uint64_t wanted, const Range& range) {
                    return wanted < range.file_offset / alignment;
                });
            if (range_it == slot.plan_ranges.begin()) {
                throw std::runtime_error("planned row page has no staging range");
            }
            const Range& range = *std::prev(range_it);
            const std::uint64_t first_page = range.file_offset / alignment;
            const std::uint64_t last_page = first_page + range.size / alignment - 1;
            if (page > last_page) throw std::runtime_error("planned row page is not covered");
            const std::uint64_t staging = range.staging_offset + (page - first_page) * alignment +
                                          offset % alignment;
            if (staging + row_bytes > slot.staging_capacity) {
                throw std::runtime_error("planned gather span exceeds staging slot");
            }
            slot.plan_gather_offsets.push_back(staging);
        }
        const double planning_seconds = seconds_since(planning_start);
        ReadMetrics metrics = read(slot_index, slot.plan_ranges, slot.plan_gather_offsets,
                                   row_bytes, output_bytes, output_offset_bytes, stream,
                                   mode, measure_gpu);
        metrics.planning_seconds = planning_seconds;
        metrics.requested_count = ids.count;
        metrics.requested_bytes = output_bytes;
        metrics.unique_page_count = slot.plan_pages.size();
        return metrics;
    }

    ReadMetrics read(std::size_t slot_index, const std::vector<Range>& ranges,
                     const std::vector<std::uint64_t>& gather_offsets, std::uint64_t row_bytes,
                     std::uint64_t output_bytes, std::uint64_t output_offset_bytes,
                     CUstream stream, const std::string& mode, bool measure_gpu) {
        const auto native_start = Clock::now();
        std::lock_guard<std::mutex> binding_lock(output_binding_mutex_);
        if (slot_index >= slots_.size()) throw std::runtime_error("slot index is out of range");
        if (!row_bytes || output_bytes != gather_offsets.size() * row_bytes) {
            throw std::runtime_error("invalid gather/output geometry");
        }
        if (ranges.empty() || gather_offsets.empty()) {
            throw std::runtime_error("read request must contain at least one range and row");
        }
        if (gather_offsets.size() > max_spans_) throw std::runtime_error("gather span capacity exceeded");
        Slot& slot = *slots_[slot_index];
        std::unique_lock<std::mutex> lock(slot.mutex);
        if (!slot.released) throw std::runtime_error("slot is still owned by the previous consumer");
        if (output_offset_bytes > slot.output_capacity ||
            output_bytes > slot.output_capacity - output_offset_bytes) {
            throw std::runtime_error("output slot capacity exceeded");
        }
        std::uint64_t expected_read = 0;
        for (const auto& range : ranges) {
            if ((range.file_offset | range.size | range.staging_offset) & 4095U) {
                throw std::runtime_error("cuFile range is not 4096-byte aligned");
            }
            if (!range.size || range.staging_offset + range.size > slot.staging_capacity) {
                throw std::runtime_error("cuFile range exceeds staging slot");
            }
            const std::uint64_t lane = range.staging_offset / lane_capacity_;
            const std::uint64_t lane_offset = range.staging_offset % lane_capacity_;
            if (lane >= requested_worker_count_ || lane_offset + range.size > lane_capacity_) {
                throw std::runtime_error("cuFile range crosses a registered worker lane");
            }
            expected_read += range.size;
        }
        for (auto offset : gather_offsets) {
            if (offset + row_bytes > slot.staging_capacity) throw std::runtime_error("gather exceeds staging slot");
        }
        ContextGuard guard(context_);
        ReadMetrics metrics;
        if (slot.consumer_event_recorded) {
            const auto wait_start = Clock::now();
            check_cuda(cuEventSynchronize(slot.consumer_done), "cuEventSynchronize(consumer_done)");
            metrics.slot_wait_seconds = seconds_since(wait_start);
            slot.consumer_event_recorded = false;
            slot.gather_event_recorded = false;
        } else if (slot.gather_event_recorded) {
            const auto wait_start = Clock::now();
            check_cuda(cuEventSynchronize(slot.gather_end), "cuEventSynchronize(gather_end)");
            metrics.slot_wait_seconds = seconds_since(wait_start);
            slot.gather_event_recorded = false;
        }
        slot.released = false;
        try {
            if (mode == "sync") {
                const auto io_start = Clock::now();
                for (const auto& range : ranges) {
                    const std::uint64_t lane = range.staging_offset / lane_capacity_;
                    const std::uint64_t lane_offset = range.staging_offset % lane_capacity_;
                    const CUdeviceptr lane_base = slot.staging + lane * lane_capacity_;
                    const ssize_t got = cuFileRead(file_handle_, reinterpret_cast<void*>(lane_base),
                                                   range.size, range.file_offset, lane_offset);
                    if (got < 0) {
                        throw std::runtime_error("cuFileRead failed: " +
                                                 std::string(CUFILE_ERRSTR(static_cast<int>(got))) +
                                                 " (" + std::to_string(got) + ")");
                    }
                    if (static_cast<std::uint64_t>(got) != range.size) {
                        throw std::runtime_error("cuFileRead short read");
                    }
                    metrics.bytes_read += static_cast<std::uint64_t>(got);
                }
                metrics.io_wait_seconds = seconds_since(io_start);
            } else if (mode == "sync_mt") {
                sync_mt_read(slot, ranges, &metrics);
            } else if (mode == "async") {
                if (!measure_gpu) {
                    throw std::runtime_error("diagnostic async requires synchronous result validation");
                }
                async_read(slot, ranges, stream, &metrics);
                metrics.bytes_read = expected_read;
            } else if (mode == "batch") {
                batch_read(slot, ranges, &metrics);
            } else {
                throw std::runtime_error("mode must be 'sync', 'sync_mt', 'async', or 'batch'");
            }
            if (metrics.bytes_read != expected_read) throw std::runtime_error("aggregate I/O byte count mismatch");
            metrics.io_count = ranges.size();
            const auto metadata_start = Clock::now();
            const std::size_t count = gather_offsets.size();
            if (!slot.host_sources || !slot.host_destinations || !slot.host_lengths) {
                throw std::runtime_error("pinned metadata workspace is not initialized");
            }
            for (std::size_t index = 0; index < count; ++index) {
                slot.host_sources[index] = gather_offsets[index];
                slot.host_destinations[index] = output_offset_bytes + index * row_bytes;
                slot.host_lengths[index] = row_bytes;
            }
            check_cuda(cuMemcpyHtoDAsync(slot.sources, slot.host_sources,
                                         count * sizeof(std::uint64_t), stream),
                       "cuMemcpyHtoDAsync(sources)");
            check_cuda(cuMemcpyHtoDAsync(slot.destinations, slot.host_destinations,
                                         count * sizeof(std::uint64_t), stream),
                       "cuMemcpyHtoDAsync(destinations)");
            check_cuda(cuMemcpyHtoDAsync(slot.lengths, slot.host_lengths,
                                         count * sizeof(std::uint64_t), stream),
                       "cuMemcpyHtoDAsync(lengths)");
            metrics.metadata_seconds = seconds_since(metadata_start);
            const auto gather_start = Clock::now();
            unsigned int count_u32 = static_cast<unsigned int>(count);
            if (measure_gpu) check_cuda(cuEventRecord(slot.gather_start, stream), "cuEventRecord(gather_start)");
            void* args[] = {&slot.staging, &slot.output, &slot.sources, &slot.destinations,
                            &slot.lengths, &count_u32};
            check_cuda(cuLaunchKernel(gather_, count_u32, 1, 1, 256, 1, 1, 0, stream, args, nullptr),
                       "cuLaunchKernel(gather)");
            check_cuda(cuEventRecord(slot.gather_end, stream), "cuEventRecord(gather_end)");
            slot.gather_event_recorded = true;
            metrics.gather_launch_seconds = seconds_since(gather_start);
            if (measure_gpu) {
                check_cuda(cuEventSynchronize(slot.gather_end), "cuEventSynchronize(gather benchmark)");
                if (mode == "async") validate_async_results(slot, ranges);
                float milliseconds = 0.0F;
                check_cuda(cuEventElapsedTime(&milliseconds, slot.gather_start, slot.gather_end),
                           "cuEventElapsedTime(gather)");
                metrics.gather_gpu_seconds = milliseconds / 1000.0;
            }
        } catch (...) {
            slot.released = true;
            throw;
        }
        metrics.native_seconds = seconds_since(native_start);
        calls_.fetch_add(1, std::memory_order_relaxed);
        bytes_read_.fetch_add(metrics.bytes_read, std::memory_order_relaxed);
        if (mode == "batch") batch_calls_.fetch_add(1, std::memory_order_relaxed);
        else if (mode == "sync_mt") sync_mt_calls_.fetch_add(1, std::memory_order_relaxed);
        else sync_calls_.fetch_add(1, std::memory_order_relaxed);
        return metrics;
    }

    void rebind_outputs(const std::vector<std::uint64_t>& output_ptrs,
                        std::uint64_t output_capacity) {
        std::lock_guard<std::mutex> binding_lock(output_binding_mutex_);
        if (closed_) throw std::runtime_error("persistent reader is closed");
        if (output_ptrs.size() != slots_.size() || !output_capacity) {
            throw std::runtime_error("replacement output slot geometry is invalid");
        }
        ContextGuard guard(context_);
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            Slot& slot = *slots_[index];
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (!slot.released) {
                throw std::runtime_error("cannot rebind an active output slot");
            }
            if (slot.consumer_event_recorded) {
                check_cuda(cuEventSynchronize(slot.consumer_done),
                           "cuEventSynchronize(output rebind)");
                slot.consumer_event_recorded = false;
                slot.gather_event_recorded = false;
            } else if (slot.gather_event_recorded) {
                check_cuda(cuEventSynchronize(slot.gather_end),
                           "cuEventSynchronize(output rebind gather)");
                slot.gather_event_recorded = false;
            }
            verify_pointer(static_cast<CUdeviceptr>(output_ptrs[index]), output_capacity,
                           "replacement output");
        }
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            slots_[index]->output = static_cast<CUdeviceptr>(output_ptrs[index]);
            slots_[index]->output_capacity = output_capacity;
        }
        ++output_rebind_count_;
    }

    void release(std::size_t slot_index, CUstream stream) {
        if (slot_index >= slots_.size()) throw std::runtime_error("slot index is out of range");
        Slot& slot = *slots_[slot_index];
        std::lock_guard<std::mutex> lock(slot.mutex);
        if (slot.released) throw std::runtime_error("slot is not owned by a consumer");
        ContextGuard guard(context_);
        check_cuda(cuEventRecord(slot.consumer_done, stream), "cuEventRecord(consumer_done)");
        slot.consumer_event_recorded = true;
        slot.released = true;
        releases_.fetch_add(1, std::memory_order_relaxed);
    }

    void close() {
        std::lock_guard<std::mutex> close_lock(close_mutex_);
        if (closed_) return;
        stop_workers();
        std::lock_guard<std::mutex> binding_lock(output_binding_mutex_);
        ContextGuard guard(context_);
        for (auto& owned_slot : slots_) {
            Slot& slot = *owned_slot;
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (slot.gather_event_recorded && slot.gather_end) cuEventSynchronize(slot.gather_end);
            if (slot.consumer_event_recorded && slot.consumer_done) cuEventSynchronize(slot.consumer_done);
            if (slot.batch) cuFileBatchIODestroy(slot.batch);
            if (slot.async_stream_registered) cuFileStreamDeregister(slot.async_stream);
            for (std::size_t lane = 0; lane < slot.registered_lanes; ++lane) {
                cuFileBufDeregister(reinterpret_cast<void*>(slot.staging + lane * lane_capacity_));
            }
            if (slot.sources) cuMemFree(slot.sources);
            if (slot.destinations) cuMemFree(slot.destinations);
            if (slot.lengths) cuMemFree(slot.lengths);
            if (slot.host_metadata_workspace) {
                check_cuda(cuMemFreeHost(slot.host_metadata_workspace),
                           "cuMemFreeHost(metadata workspace)");
                ++pinned_metadata_free_count_;
                pinned_metadata_free_bytes_ += slot.host_metadata_workspace_bytes;
                slot.host_metadata_workspace = nullptr;
                slot.host_metadata_workspace_bytes = 0;
                slot.host_sources = nullptr;
                slot.host_destinations = nullptr;
                slot.host_lengths = nullptr;
            }
            if (slot.gather_start) cuEventDestroy(slot.gather_start);
            if (slot.gather_end) cuEventDestroy(slot.gather_end);
            if (slot.consumer_done) cuEventDestroy(slot.consumer_done);
        }
        slots_.clear();
        if (module_) cuModuleUnload(module_);
        if (file_handle_) cuFileHandleDeregister(file_handle_);
        if (driver_open_) cuFileDriverClose();
        if (fd_ >= 0) ::close(fd_);
        module_ = nullptr;
        file_handle_ = nullptr;
        driver_open_ = false;
        fd_ = -1;
        closed_ = true;
        close_count_ = 1;
    }

    void close_noexcept() noexcept {
        try { close(); } catch (...) {}
    }

    const std::string& gpu_uuid() const { return gpu_uuid_; }
    const std::string& gpu_bdf() const { return gpu_bdf_; }
    int device_ordinal() const { return device_ordinal_; }
    unsigned int batch_capacity() const { return batch_capacity_; }
    unsigned int driver_control_flags() const { return driver_control_flags_; }
    std::size_t slot_count() const { return slots_.size(); }
    bool closed() const { return closed_; }
    std::uint64_t calls() const { return calls_.load(); }
    std::uint64_t sync_calls() const { return sync_calls_.load(); }
    std::uint64_t sync_mt_calls() const { return sync_mt_calls_.load(); }
    std::uint64_t batch_calls() const { return batch_calls_.load(); }
    std::uint64_t bytes_read() const { return bytes_read_.load(); }
    std::uint64_t releases() const { return releases_.load(); }
    std::uint64_t initialization_count() const { return initialization_count_; }
    std::uint64_t close_count() const { return close_count_; }
    std::uint64_t file_open_count() const { return file_open_count_; }
    std::uint64_t driver_open_count() const { return driver_open_count_; }
    std::uint64_t handle_registration_count() const { return handle_registration_count_; }
    std::uint64_t buffer_registration_count() const { return buffer_registration_count_; }
    std::uint64_t metadata_allocation_count() const { return metadata_allocation_count_; }
    std::uint64_t module_load_count() const { return module_load_count_; }
    std::uint64_t batch_setup_count() const { return batch_setup_count_; }
    std::uint64_t host_workspace_allocation_count() const { return host_workspace_allocation_count_; }
    std::uint64_t pinned_metadata_allocation_count() const { return pinned_metadata_allocation_count_; }
    std::uint64_t pinned_metadata_free_count() const { return pinned_metadata_free_count_; }
    std::uint64_t pinned_metadata_allocation_bytes() const { return pinned_metadata_allocation_bytes_; }
    std::uint64_t pinned_metadata_free_bytes() const { return pinned_metadata_free_bytes_; }
    std::uint64_t output_rebind_count() const { return output_rebind_count_; }
    unsigned int worker_count() const { return requested_worker_count_; }
    bool pin_workers() const { return pin_workers_; }
    bool poll_mode() const { return poll_mode_; }

private:
    static std::size_t metadata_workspace_bytes_for(std::uint64_t max_spans) {
        constexpr std::size_t kMetadataArrays = 3;
        constexpr std::size_t kBytesPerSpan = kMetadataArrays * sizeof(std::uint64_t);
        if (max_spans > std::numeric_limits<std::size_t>::max() / kBytesPerSpan) {
            throw std::runtime_error("pinned metadata workspace size overflow");
        }
        return static_cast<std::size_t>(max_spans) * kBytesPerSpan;
    }

    struct Worker {
        std::thread thread;
        int fd = -1;
        CUfileHandle_t file_handle = nullptr;
        bool initialized = false;
        int cpu = -1;
        std::string initialization_error;
    };

    void start_workers() {
        workers_.reserve(requested_worker_count_);
        for (unsigned int index = 0; index < requested_worker_count_; ++index) {
            workers_.push_back(std::make_unique<Worker>());
            Worker* worker = workers_.back().get();
            worker->thread = std::thread([this, worker, index]() { worker_loop(worker, index); });
        }
        std::unique_lock<std::mutex> lock(work_mutex_);
        init_cv_.wait(lock, [this]() {
            return std::all_of(workers_.begin(), workers_.end(),
                               [](const auto& worker) { return worker->initialized; });
        });
        for (const auto& worker : workers_) {
            if (!worker->initialization_error.empty()) {
                const std::string error = worker->initialization_error;
                lock.unlock();
                stop_workers();
                throw std::runtime_error("sync_mt worker initialization failed: " + error);
            }
        }
        file_open_count_ += requested_worker_count_;
        handle_registration_count_ += requested_worker_count_;
    }

    void worker_loop(Worker* worker, unsigned int index) noexcept {
        try {
            ContextGuard guard(context_);
            if (pin_workers_) {
                const long online = sysconf(_SC_NPROCESSORS_ONLN);
                const unsigned int physical = static_cast<unsigned int>(std::max(1L, online / 2));
                worker->cpu = static_cast<int>(index % physical);
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(worker->cpu, &set);
                const int affinity_error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                if (affinity_error != 0) {
                    throw std::runtime_error("pthread_setaffinity_np: " +
                                             std::string(std::strerror(affinity_error)));
                }
            }
            worker->fd = open(path_.c_str(), O_RDONLY | O_DIRECT);
            if (worker->fd < 0) {
                throw std::runtime_error("worker open(O_DIRECT): " + std::string(std::strerror(errno)));
            }
            CUfileDescr_t descriptor{};
            descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            descriptor.handle.fd = worker->fd;
            check_cufile(cuFileHandleRegister(&worker->file_handle, &descriptor),
                         "worker cuFileHandleRegister");
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                worker->initialized = true;
            }
            init_cv_.notify_all();

            std::uint64_t observed_generation = 0;
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(work_mutex_);
                    work_cv_.wait(lock, [&]() {
                        return stopping_workers_ || work_generation_ != observed_generation;
                    });
                    if (stopping_workers_) break;
                    observed_generation = work_generation_;
                }
                std::uint64_t local_bytes = 0;
                std::uint64_t local_ios = 0;
                if (!active_ranges_) {
                    work_failed_.store(true, std::memory_order_relaxed);
                } else {
                    for (const Range& range : *active_ranges_) {
                        if (work_failed_.load(std::memory_order_relaxed)) break;
                        if (range.staging_offset / lane_capacity_ != index) continue;
                        const CUdeviceptr lane_base = active_staging_ + index * lane_capacity_;
                        const std::uint64_t lane_offset = range.staging_offset % lane_capacity_;
                        const ssize_t got = cuFileRead(worker->file_handle,
                                                       reinterpret_cast<void*>(lane_base),
                                                       range.size, range.file_offset,
                                                       lane_offset);
                        if (got < 0 || static_cast<std::uint64_t>(got) != range.size) {
                            std::lock_guard<std::mutex> lock(work_mutex_);
                            if (!work_failed_.exchange(true)) {
                                work_error_ = got < 0
                                    ? "sync_mt cuFileRead failed: " +
                                          std::string(CUFILE_ERRSTR(static_cast<int>(got))) +
                                          " (" + std::to_string(got) + ")"
                                    : "sync_mt cuFileRead short read";
                            }
                            break;
                        }
                        local_bytes += static_cast<std::uint64_t>(got);
                        ++local_ios;
                    }
                }
                work_bytes_.fetch_add(local_bytes, std::memory_order_relaxed);
                work_ios_.fetch_add(local_ios, std::memory_order_relaxed);
                {
                    std::lock_guard<std::mutex> lock(work_mutex_);
                    if (--workers_pending_ == 0) work_done_cv_.notify_one();
                }
            }
            if (worker->file_handle) cuFileHandleDeregister(worker->file_handle);
            worker->file_handle = nullptr;
            if (worker->fd >= 0) ::close(worker->fd);
            worker->fd = -1;
        } catch (const std::exception& exc) {
            if (worker->file_handle) cuFileHandleDeregister(worker->file_handle);
            if (worker->fd >= 0) ::close(worker->fd);
            worker->file_handle = nullptr;
            worker->fd = -1;
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                worker->initialization_error = exc.what();
                worker->initialized = true;
            }
            init_cv_.notify_all();
        }
    }

    void stop_workers() noexcept {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            stopping_workers_ = true;
        }
        work_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker->thread.joinable()) worker->thread.join();
        }
        workers_.clear();
    }

    void sync_mt_read(Slot& slot, const std::vector<Range>& ranges, ReadMetrics* metrics) {
        const auto io_start = Clock::now();
        std::unique_lock<std::mutex> request_lock(mt_request_mutex_);
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            active_ranges_ = &ranges;
            active_staging_ = slot.staging;
            work_bytes_.store(0, std::memory_order_relaxed);
            work_ios_.store(0, std::memory_order_relaxed);
            work_failed_.store(false, std::memory_order_relaxed);
            work_error_.clear();
            workers_pending_ = workers_.size();
            ++work_generation_;
        }
        work_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lock(work_mutex_);
            work_done_cv_.wait(lock, [this]() { return workers_pending_ == 0; });
            active_ranges_ = nullptr;
            active_staging_ = 0;
            if (work_failed_.load(std::memory_order_relaxed)) {
                throw std::runtime_error(work_error_);
            }
        }
        metrics->bytes_read = work_bytes_.load(std::memory_order_relaxed);
        if (work_ios_.load(std::memory_order_relaxed) != ranges.size()) {
            throw std::runtime_error("sync_mt completed I/O count mismatch");
        }
        metrics->io_wait_seconds = seconds_since(io_start);
    }

    void async_read(Slot& slot, const std::vector<Range>& ranges, CUstream stream,
                    ReadMetrics* metrics) {
        if (ranges.size() > slot.async_sizes.size()) {
            throw std::runtime_error("fixed async parameter workspace exceeded");
        }
        if (!slot.async_stream_registered || slot.async_stream != stream) {
            if (slot.async_stream_registered) {
                check_cufile(cuFileStreamDeregister(slot.async_stream), "cuFileStreamDeregister");
            }
            check_cufile(cuFileStreamRegister(stream, CU_FILE_STREAM_PAGE_ALIGNED_INPUTS),
                         "cuFileStreamRegister(PAGE_ALIGNED)");
            slot.async_stream = stream;
            slot.async_stream_registered = true;
        }
        const auto submit_start = Clock::now();
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            const std::uint64_t lane = ranges[index].staging_offset / lane_capacity_;
            const CUdeviceptr lane_base = slot.staging + lane * lane_capacity_;
            slot.async_sizes[index] = ranges[index].size;
            slot.async_file_offsets[index] = static_cast<off_t>(ranges[index].file_offset);
            slot.async_buffer_offsets[index] =
                static_cast<off_t>(ranges[index].staging_offset % lane_capacity_);
            slot.async_results[index] = 0;
            check_cufile(cuFileReadAsync(file_handle_, reinterpret_cast<void*>(lane_base),
                                         &slot.async_sizes[index], &slot.async_file_offsets[index],
                                         &slot.async_buffer_offsets[index], &slot.async_results[index],
                                         stream),
                         "cuFileReadAsync");
        }
        metrics->submission_seconds = seconds_since(submit_start);
    }

    void validate_async_results(const Slot& slot, const std::vector<Range>& ranges) {
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            if (slot.async_results[index] < 0) {
                throw std::runtime_error("cuFileReadAsync completion failed: " +
                                         std::string(CUFILE_ERRSTR(
                                             static_cast<int>(slot.async_results[index]))));
            }
            if (static_cast<std::uint64_t>(slot.async_results[index]) != ranges[index].size) {
                throw std::runtime_error("cuFileReadAsync completion was short");
            }
        }
    }

    void verify_pointer(CUdeviceptr pointer, std::uint64_t capacity, const char* name) {
        CUcontext pointer_context = nullptr;
        std::size_t allocation_size = 0;
        CUdeviceptr allocation_base = 0;
        check_cuda(cuPointerGetAttribute(&pointer_context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer),
                   "cuPointerGetAttribute(context)");
        if (pointer_context != context_) throw std::runtime_error(std::string(name) + " pointer context mismatch");
        check_cuda(cuMemGetAddressRange(&allocation_base, &allocation_size, pointer), "cuMemGetAddressRange");
        if (pointer < allocation_base || pointer + capacity > allocation_base + allocation_size) {
            throw std::runtime_error(std::string(name) + " capacity exceeds torch allocation");
        }
        if (pointer & 4095U) throw std::runtime_error(std::string(name) + " pointer is not 4096-byte aligned");
    }

    void batch_read(Slot& slot, const std::vector<Range>& ranges, ReadMetrics* metrics) {
        for (std::size_t start = 0; start < ranges.size(); start += batch_capacity_) {
            const unsigned int count = static_cast<unsigned int>(
                std::min<std::size_t>(batch_capacity_, ranges.size() - start));
            for (unsigned int index = 0; index < count; ++index) {
                const Range& range = ranges[start + index];
                const std::uint64_t lane = range.staging_offset / lane_capacity_;
                slot.batch_params[index].mode = CUFILE_BATCH;
                slot.batch_params[index].u.batch.devPtr_base =
                    reinterpret_cast<void*>(slot.staging + lane * lane_capacity_);
                slot.batch_params[index].u.batch.file_offset = range.file_offset;
                slot.batch_params[index].u.batch.devPtr_offset = range.staging_offset % lane_capacity_;
                slot.batch_params[index].u.batch.size = range.size;
                slot.batch_params[index].fh = slot.batch_file_handle;
                slot.batch_params[index].opcode = CUFILE_READ;
                slot.batch_params[index].cookie =
                    reinterpret_cast<void*>(static_cast<std::uintptr_t>(index + 1));
                slot.batch_expected[index] = range.size;
            }
            const auto submit_start = Clock::now();
            check_cufile(cuFileBatchIOSubmit(slot.batch, count, slot.batch_params.data(), 0),
                         "cuFileBatchIOSubmit");
            metrics->submission_seconds += seconds_since(submit_start);
            const auto wait_start = Clock::now();
            std::fill_n(slot.batch_completed.begin(), count, 0);
            unsigned int completed_count = 0;
            while (completed_count < count) {
                unsigned int available = count - completed_count;
                timespec timeout{10, 0};
                check_cufile(cuFileBatchIOGetStatus(slot.batch, 1, &available,
                                                    slot.batch_events.data(), &timeout),
                             "cuFileBatchIOGetStatus");
                if (!available) throw std::runtime_error("cuFileBatchIOGetStatus returned no completion");
                for (unsigned int event_index = 0; event_index < available; ++event_index) {
                    const auto& event = slot.batch_events[event_index];
                    const auto cookie = reinterpret_cast<std::uintptr_t>(event.cookie);
                    if (!cookie || cookie > count) throw std::runtime_error("invalid cuFile batch cookie");
                    const std::size_t index = cookie - 1;
                    if (slot.batch_completed[index]) {
                        throw std::runtime_error("duplicate cuFile batch completion");
                    }
                    if (event.status != CUFILE_COMPLETE) {
                        const auto signed_ret = static_cast<ssize_t>(event.ret);
                        const std::string detail = signed_ret < 0
                            ? CUFILE_ERRSTR(static_cast<int>(signed_ret))
                            : "non-complete status";
                        throw std::runtime_error("cuFile batch item failed: " + detail + ", status=" +
                                                 std::to_string(event.status) +
                                                 " ret=" + std::to_string(signed_ret));
                    }
                    if (event.ret != slot.batch_expected[index]) {
                        throw std::runtime_error("cuFile batch item short read: " +
                                                 std::to_string(event.ret) + " != " +
                                                 std::to_string(slot.batch_expected[index]));
                    }
                    slot.batch_completed[index] = 1;
                    ++completed_count;
                    metrics->bytes_read += event.ret;
                }
            }
            metrics->io_wait_seconds += seconds_since(wait_start);
            ++metrics->batch_chunks;
        }
    }

    std::string path_;
    std::string expected_uuid_;
    std::string expected_bdf_;
    std::string gpu_uuid_;
    std::string gpu_bdf_;
    CUdevice device_ = 0;
    int device_ordinal_ = -1;
    CUcontext context_ = nullptr;
    int fd_ = -1;
    CUfileHandle_t file_handle_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction gather_ = nullptr;
    bool driver_open_ = false;
    bool closed_ = false;
    unsigned int batch_capacity_ = 0;
    unsigned int driver_control_flags_ = 0;
    std::uint64_t max_spans_ = 0;
    std::uint64_t lane_capacity_ = 0;
    unsigned int requested_worker_count_ = 0;
    bool pin_workers_ = false;
    bool poll_mode_ = false;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex close_mutex_;
    std::mutex output_binding_mutex_;
    std::mutex mt_request_mutex_;
    std::mutex work_mutex_;
    std::condition_variable init_cv_;
    std::condition_variable work_cv_;
    std::condition_variable work_done_cv_;
    bool stopping_workers_ = false;
    std::uint64_t work_generation_ = 0;
    std::size_t workers_pending_ = 0;
    const std::vector<Range>* active_ranges_ = nullptr;
    CUdeviceptr active_staging_ = 0;
    std::atomic<std::uint64_t> work_bytes_{0};
    std::atomic<std::uint64_t> work_ios_{0};
    std::atomic<bool> work_failed_{false};
    std::string work_error_;
    std::atomic<std::uint64_t> calls_{0};
    std::atomic<std::uint64_t> sync_calls_{0};
    std::atomic<std::uint64_t> sync_mt_calls_{0};
    std::atomic<std::uint64_t> batch_calls_{0};
    std::atomic<std::uint64_t> bytes_read_{0};
    std::atomic<std::uint64_t> releases_{0};
    std::uint64_t initialization_count_ = 0;
    std::uint64_t close_count_ = 0;
    std::uint64_t file_open_count_ = 0;
    std::uint64_t driver_open_count_ = 0;
    std::uint64_t handle_registration_count_ = 0;
    std::uint64_t buffer_registration_count_ = 0;
    std::uint64_t metadata_allocation_count_ = 0;
    std::uint64_t module_load_count_ = 0;
    std::uint64_t batch_setup_count_ = 0;
    std::uint64_t host_workspace_allocation_count_ = 0;
    std::uint64_t pinned_metadata_allocation_count_ = 0;
    std::uint64_t pinned_metadata_free_count_ = 0;
    std::uint64_t pinned_metadata_allocation_bytes_ = 0;
    std::uint64_t pinned_metadata_free_bytes_ = 0;
    std::uint64_t output_rebind_count_ = 0;
};

typedef struct {
    PyObject_HEAD
    PersistentState* state;
} PersistentObject;

bool parse_u64(PyObject* value, std::uint64_t* output) {
    const unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return false;
    *output = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parse_u64_sequence(PyObject* object, const char* message, std::vector<std::uint64_t>* output) {
    PyObject* sequence = PySequence_Fast(object, message);
    if (!sequence) return false;
    bool ok = true;
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(sequence) && ok; ++index) {
        std::uint64_t value = 0;
        ok = parse_u64(PySequence_Fast_GET_ITEM(sequence, index), &value);
        if (ok) output->push_back(value);
    }
    Py_DECREF(sequence);
    return ok;
}

bool parse_ranges(PyObject* object, std::vector<Range>* output) {
    PyObject* sequence = PySequence_Fast(object, "ranges must be a sequence");
    if (!sequence) return false;
    bool ok = true;
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(sequence) && ok; ++index) {
        PyObject* tuple = PySequence_Fast(PySequence_Fast_GET_ITEM(sequence, index),
                                          "each range must be a sequence");
        if (!tuple) { ok = false; break; }
        if (PySequence_Fast_GET_SIZE(tuple) != 3) {
            PyErr_SetString(PyExc_ValueError, "each range needs file_offset, size, staging_offset");
            ok = false;
        } else {
            Range range{};
            ok = parse_u64(PySequence_Fast_GET_ITEM(tuple, 0), &range.file_offset) &&
                 parse_u64(PySequence_Fast_GET_ITEM(tuple, 1), &range.size) &&
                 parse_u64(PySequence_Fast_GET_ITEM(tuple, 2), &range.staging_offset);
            if (ok) output->push_back(range);
        }
        Py_DECREF(tuple);
    }
    Py_DECREF(sequence);
    return ok;
}

bool parse_id_buffer(PyObject* object, Py_buffer* view, IdBuffer* output) {
    if (PyObject_GetBuffer(object, view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        PyErr_SetString(PyExc_TypeError, "row IDs must expose a contiguous signed int32/int64 buffer");
        return false;
    }
    const bool one_dimensional = view->ndim == 1;
    const bool contiguous = PyBuffer_IsContiguous(view, 'C') != 0;
    const bool signed_size = view->itemsize == 4 || view->itemsize == 8;
    const char* format = view->format ? view->format : "";
    while (*format == '@' || *format == '=' || *format == '<' || *format == '>' || *format == '!') {
        ++format;
    }
    const bool signed_format = (*format == 'i' && view->itemsize == 4) ||
                               ((*format == 'l' || *format == 'q') && view->itemsize == 8);
    if (!one_dimensional || !contiguous || !signed_size || !signed_format || view->len < 0 ||
        view->len % view->itemsize != 0) {
        PyBuffer_Release(view);
        std::memset(view, 0, sizeof(*view));
        PyErr_SetString(PyExc_TypeError, "row IDs must be a contiguous native signed int32/int64 buffer");
        return false;
    }
    output->data = view->buf;
    output->count = static_cast<std::size_t>(view->len / view->itemsize);
    output->item_size = static_cast<std::size_t>(view->itemsize);
    return true;
}

PyObject* metrics_dict(const ReadMetrics& metrics, const std::string& mode, std::size_t slot) {
    PyObject* result = PyDict_New();
    auto set = [&](const char* key, PyObject* value) {
        PyDict_SetItemString(result, key, value);
        Py_DECREF(value);
    };
    set("mode", PyUnicode_FromString(mode.c_str()));
    set("slot", PyLong_FromSize_t(slot));
    set("bytes_read", PyLong_FromUnsignedLongLong(metrics.bytes_read));
    set("io_count", PyLong_FromUnsignedLongLong(metrics.io_count));
    set("batch_chunks", PyLong_FromUnsignedLongLong(metrics.batch_chunks));
    set("slot_wait_seconds", PyFloat_FromDouble(metrics.slot_wait_seconds));
    set("submission_seconds", PyFloat_FromDouble(metrics.submission_seconds));
    set("io_wait_seconds", PyFloat_FromDouble(metrics.io_wait_seconds));
    set("metadata_seconds", PyFloat_FromDouble(metrics.metadata_seconds));
    set("gather_launch_seconds", PyFloat_FromDouble(metrics.gather_launch_seconds));
    set("gather_gpu_seconds", PyFloat_FromDouble(metrics.gather_gpu_seconds));
    set("native_seconds", PyFloat_FromDouble(metrics.native_seconds));
    set("planning_seconds", PyFloat_FromDouble(metrics.planning_seconds));
    set("requested_count", PyLong_FromUnsignedLongLong(metrics.requested_count));
    set("requested_bytes", PyLong_FromUnsignedLongLong(metrics.requested_bytes));
    set("unique_page_count", PyLong_FromUnsignedLongLong(metrics.unique_page_count));
    return result;
}

int persistent_init(PersistentObject* self, PyObject* args, PyObject*) {
    const char* path = nullptr;
    PyObject* staging_object = nullptr;
    unsigned long long staging_capacity = 0;
    PyObject* output_object = nullptr;
    unsigned long long output_capacity = 0;
    const char* uuid = nullptr;
    const char* bdf = nullptr;
    unsigned long long max_spans = 0;
    unsigned int batch_capacity = 0;
    unsigned int worker_count = 0;
    int pin_workers = 0, poll_mode = 0;
    if (!PyArg_ParseTuple(args, "sOKOKssKIIpp", &path, &staging_object, &staging_capacity,
                          &output_object, &output_capacity, &uuid, &bdf, &max_spans,
                          &batch_capacity, &worker_count, &pin_workers, &poll_mode)) return -1;
    std::vector<std::uint64_t> staging_ptrs, output_ptrs;
    if (!parse_u64_sequence(staging_object, "staging pointers must be a sequence", &staging_ptrs) ||
        !parse_u64_sequence(output_object, "output pointers must be a sequence", &output_ptrs)) return -1;
    try {
        self->state = new PersistentState(path, staging_ptrs, staging_capacity, output_ptrs,
                                          output_capacity, uuid, bdf, max_spans, batch_capacity,
                                          worker_count, pin_workers != 0, poll_mode != 0);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return -1;
    }
    return 0;
}

void persistent_dealloc(PersistentObject* self) {
    delete self->state;
    self->state = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* persistent_read(PersistentObject* self, PyObject* args) {
    int slot = 0;
    PyObject* ranges_object = nullptr;
    PyObject* offsets_object = nullptr;
    unsigned long long row_bytes = 0, output_bytes = 0, stream = 0;
    const char* mode = nullptr;
    int measure = 0;
    if (!PyArg_ParseTuple(args, "iOOKKKsp", &slot, &ranges_object, &offsets_object,
                          &row_bytes, &output_bytes, &stream, &mode, &measure)) return nullptr;
    if (!self->state || self->state->closed()) {
        PyErr_SetString(PyExc_RuntimeError, "persistent reader is closed");
        return nullptr;
    }
    std::vector<Range> ranges;
    std::vector<std::uint64_t> offsets;
    if (!parse_ranges(ranges_object, &ranges) ||
        !parse_u64_sequence(offsets_object, "gather offsets must be a sequence", &offsets)) return nullptr;
    ReadMetrics metrics;
    std::string failure;
    PyThreadState* thread = PyEval_SaveThread();
    try {
        metrics = self->state->read(slot, ranges, offsets, row_bytes, output_bytes, 0,
                                    reinterpret_cast<CUstream>(stream), mode, measure != 0);
    } catch (const std::exception& exc) {
        failure = exc.what();
    }
    PyEval_RestoreThread(thread);
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    return metrics_dict(metrics, mode, slot);
}

PyObject* persistent_read_ids_common(PersistentObject* self, PyObject* args,
                                     bool with_output_offset) {
    int slot = 0;
    PyObject* ids_object = nullptr;
    unsigned long long row_count = 0, base_offset = 0, block_rows = 0, block_stride = 0;
    unsigned long long row_bytes = 0, alignment = 0, max_gap_pages = 0;
    unsigned long long max_range_bytes = 0, output_bytes = 0;
    unsigned long long output_offset_bytes = 0, stream = 0;
    const char* mode = nullptr;
    int measure = 0;
    const bool parsed = with_output_offset
        ? PyArg_ParseTuple(args, "iOKKKKKKKKKKKsp", &slot, &ids_object, &row_count,
                           &base_offset, &block_rows, &block_stride, &row_bytes, &alignment,
                           &max_gap_pages, &max_range_bytes, &output_bytes,
                           &output_offset_bytes, &stream, &mode, &measure)
        : PyArg_ParseTuple(args, "iOKKKKKKKKKKsp", &slot, &ids_object, &row_count,
                           &base_offset, &block_rows, &block_stride, &row_bytes, &alignment,
                           &max_gap_pages, &max_range_bytes, &output_bytes, &stream,
                           &mode, &measure);
    if (!parsed) return nullptr;
    if (!self->state || self->state->closed()) {
        PyErr_SetString(PyExc_RuntimeError, "persistent reader is closed");
        return nullptr;
    }
    Py_buffer view{};
    IdBuffer ids;
    if (!parse_id_buffer(ids_object, &view, &ids)) return nullptr;
    ReadMetrics metrics;
    std::string failure;
    PyThreadState* thread = PyEval_SaveThread();
    try {
        metrics = self->state->read_ids(
            static_cast<std::size_t>(slot), ids, row_count, base_offset, block_rows,
            block_stride, row_bytes, alignment, max_gap_pages, max_range_bytes,
            output_bytes, output_offset_bytes, reinterpret_cast<CUstream>(stream),
            mode, measure != 0);
    } catch (const std::exception& exc) {
        failure = exc.what();
    }
    PyEval_RestoreThread(thread);
    PyBuffer_Release(&view);
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    return metrics_dict(metrics, mode, static_cast<std::size_t>(slot));
}

PyObject* persistent_read_ids(PersistentObject* self, PyObject* args) {
    return persistent_read_ids_common(self, args, false);
}

PyObject* persistent_read_ids_at(PersistentObject* self, PyObject* args) {
    return persistent_read_ids_common(self, args, true);
}

PyObject* persistent_rebind_outputs(PersistentObject* self, PyObject* args) {
    PyObject* output_object = nullptr;
    unsigned long long output_capacity = 0;
    if (!PyArg_ParseTuple(args, "OK", &output_object, &output_capacity)) return nullptr;
    if (!self->state || self->state->closed()) {
        PyErr_SetString(PyExc_RuntimeError, "persistent reader is closed");
        return nullptr;
    }
    std::vector<std::uint64_t> output_ptrs;
    if (!parse_u64_sequence(output_object, "output pointers must be a sequence",
                            &output_ptrs)) {
        return nullptr;
    }
    std::string failure;
    PyThreadState* thread = PyEval_SaveThread();
    try {
        self->state->rebind_outputs(output_ptrs, output_capacity);
    } catch (const std::exception& exc) {
        failure = exc.what();
    }
    PyEval_RestoreThread(thread);
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* persistent_release(PersistentObject* self, PyObject* args) {
    int slot = 0;
    unsigned long long stream = 0;
    if (!PyArg_ParseTuple(args, "iK", &slot, &stream)) return nullptr;
    try {
        self->state->release(slot, reinterpret_cast<CUstream>(stream));
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* persistent_close(PersistentObject* self, PyObject*) {
    try {
        if (self->state) self->state->close();
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* persistent_stats(PersistentObject* self, PyObject*) {
    if (!self->state) return PyDict_New();
    PyObject* result = PyDict_New();
    auto set = [&](const char* key, PyObject* value) {
        PyDict_SetItemString(result, key, value);
        Py_DECREF(value);
    };
    set("gpu_uuid", PyUnicode_FromString(self->state->gpu_uuid().c_str()));
    set("gpu_bdf", PyUnicode_FromString(self->state->gpu_bdf().c_str()));
    set("device_ordinal", PyLong_FromLong(self->state->device_ordinal()));
    set("batch_capacity", PyLong_FromUnsignedLong(self->state->batch_capacity()));
    set("driver_control_flags", PyLong_FromUnsignedLong(self->state->driver_control_flags()));
    set("slot_count", PyLong_FromSize_t(self->state->slot_count()));
    set("closed", PyBool_FromLong(self->state->closed()));
    set("calls", PyLong_FromUnsignedLongLong(self->state->calls()));
    set("sync_calls", PyLong_FromUnsignedLongLong(self->state->sync_calls()));
    set("sync_mt_calls", PyLong_FromUnsignedLongLong(self->state->sync_mt_calls()));
    set("batch_calls", PyLong_FromUnsignedLongLong(self->state->batch_calls()));
    set("bytes_read", PyLong_FromUnsignedLongLong(self->state->bytes_read()));
    set("releases", PyLong_FromUnsignedLongLong(self->state->releases()));
    set("initialization_count", PyLong_FromUnsignedLongLong(self->state->initialization_count()));
    set("close_count", PyLong_FromUnsignedLongLong(self->state->close_count()));
    set("file_open_count", PyLong_FromUnsignedLongLong(self->state->file_open_count()));
    set("driver_open_count", PyLong_FromUnsignedLongLong(self->state->driver_open_count()));
    set("handle_registration_count", PyLong_FromUnsignedLongLong(self->state->handle_registration_count()));
    set("buffer_registration_count", PyLong_FromUnsignedLongLong(self->state->buffer_registration_count()));
    set("metadata_allocation_count", PyLong_FromUnsignedLongLong(self->state->metadata_allocation_count()));
    set("module_load_count", PyLong_FromUnsignedLongLong(self->state->module_load_count()));
    set("batch_setup_count", PyLong_FromUnsignedLongLong(self->state->batch_setup_count()));
    set("host_workspace_allocation_count",
        PyLong_FromUnsignedLongLong(self->state->host_workspace_allocation_count()));
    set("pinned_metadata_allocation_count",
        PyLong_FromUnsignedLongLong(self->state->pinned_metadata_allocation_count()));
    set("pinned_metadata_free_count",
        PyLong_FromUnsignedLongLong(self->state->pinned_metadata_free_count()));
    set("pinned_metadata_allocation_bytes",
        PyLong_FromUnsignedLongLong(self->state->pinned_metadata_allocation_bytes()));
    set("pinned_metadata_free_bytes",
        PyLong_FromUnsignedLongLong(self->state->pinned_metadata_free_bytes()));
    set("output_rebind_count",
        PyLong_FromUnsignedLongLong(self->state->output_rebind_count()));
    set("worker_count", PyLong_FromUnsignedLong(self->state->worker_count()));
    set("pin_workers", PyBool_FromLong(self->state->pin_workers()));
    set("poll_mode", PyBool_FromLong(self->state->poll_mode()));
    return result;
}

PyMethodDef persistent_methods[] = {
    {"read", reinterpret_cast<PyCFunction>(persistent_read), METH_VARARGS, "Read into a torch-owned slot."},
    {"read_ids", reinterpret_cast<PyCFunction>(persistent_read_ids), METH_VARARGS,
     "Plan signed CPU row IDs in C++ and read into a torch-owned slot."},
    {"read_ids_at", reinterpret_cast<PyCFunction>(persistent_read_ids_at), METH_VARARGS,
     "Plan signed CPU row IDs and read at a byte offset in the output slot."},
    {"rebind_outputs", reinterpret_cast<PyCFunction>(persistent_rebind_outputs), METH_VARARGS,
     "Replace output pointers while every reader slot is released."},
    {"release", reinterpret_cast<PyCFunction>(persistent_release), METH_VARARGS, "Release a consumed slot."},
    {"close", reinterpret_cast<PyCFunction>(persistent_close), METH_NOARGS, "Close persistent resources."},
    {"stats", reinterpret_cast<PyCFunction>(persistent_stats), METH_NOARGS, "Return persistent counters."},
    {nullptr, nullptr, 0, nullptr},
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
PyTypeObject PersistentType = {PyVarObject_HEAD_INIT(nullptr, 0)};
#pragma GCC diagnostic pop
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_persistent_native", nullptr, -1, nullptr,
                      nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__persistent_native() {
    PersistentType.tp_name = "ple_gds._persistent_native.PersistentReader";
    PersistentType.tp_basicsize = sizeof(PersistentObject);
    PersistentType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    PersistentType.tp_new = PyType_GenericNew;
    PersistentType.tp_init = reinterpret_cast<initproc>(persistent_init);
    PersistentType.tp_dealloc = reinterpret_cast<destructor>(persistent_dealloc);
    PersistentType.tp_methods = persistent_methods;
    if (PyType_Ready(&PersistentType) < 0) return nullptr;
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
    Py_INCREF(&PersistentType);
    if (PyModule_AddObject(result, "PersistentReader", reinterpret_cast<PyObject*>(&PersistentType)) < 0) {
        Py_DECREF(&PersistentType);
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
