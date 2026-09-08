#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda.h>
#include <cufile.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Range {
    std::uint64_t file_offset;
    std::uint64_t size;
    std::uint64_t staging_offset;
};

struct Result {
    std::uint64_t bytes_read = 0;
    std::uint32_t mismatch_count = 0;
    double io_seconds = 0.0;
    std::string gpu_uuid;
    std::string gpu_bdf;
    bool handle_registered = false;
    bool buffer_registered = false;
    bool compat_mode_allowed = false;
    unsigned int driver_control_flags = 0;
};

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
        throw std::runtime_error("invalid expected GPU BDF: " + value);
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

constexpr const char* kPtx = R"PTX(
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
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
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

.visible .entry compare_bytes(
    .param .u64 p_actual,
    .param .u64 p_expected,
    .param .u64 p_size,
    .param .u64 p_mismatches)
{
    .reg .pred %p<4>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [p_actual];
    ld.param.u64 %rd2, [p_expected];
    ld.param.u64 %rd3, [p_size];
    ld.param.u64 %rd4, [p_mismatches];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    cvt.u64.u32 %rd5, %r4;
    setp.ge.u64 %p1, %rd5, %rd3;
    @%p1 bra COMPARE_DONE;
    add.u64 %rd6, %rd1, %rd5;
    add.u64 %rd7, %rd2, %rd5;
    ld.global.u8 %r5, [%rd6];
    ld.global.u8 %r6, [%rd7];
    setp.eq.u32 %p2, %r5, %r6;
    @%p2 bra COMPARE_DONE;
    atom.global.add.u32 %r7, [%rd4], 1;
COMPARE_DONE:
    ret;
}
)PTX";

struct Resources {
    int fd = -1;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUdeviceptr staging = 0;
    CUdeviceptr output = 0;
    CUdeviceptr expected = 0;
    CUdeviceptr sources = 0;
    CUdeviceptr destinations = 0;
    CUdeviceptr lengths = 0;
    CUdeviceptr mismatches = 0;
    CUfileHandle_t file_handle = nullptr;
    bool driver_open = false;
    bool buffer_registered = false;

    ~Resources() {
        if (buffer_registered) cuFileBufDeregister(reinterpret_cast<void*>(staging));
        if (file_handle) cuFileHandleDeregister(file_handle);
        if (driver_open) cuFileDriverClose();
        if (mismatches) cuMemFree(mismatches);
        if (lengths) cuMemFree(lengths);
        if (destinations) cuMemFree(destinations);
        if (sources) cuMemFree(sources);
        if (expected) cuMemFree(expected);
        if (output) cuMemFree(output);
        if (staging) cuMemFree(staging);
        if (module) cuModuleUnload(module);
        if (fd >= 0) close(fd);
        if (context) cuDevicePrimaryCtxRelease(device);
    }
};

Result run(const std::string& path, const std::vector<Range>& ranges,
           const std::vector<std::uint64_t>& gather_offsets, std::uint64_t row_bytes,
           const unsigned char* expected_host, std::uint64_t expected_size,
           std::uint64_t staging_size, std::uint64_t max_staging_size,
           const std::string& expected_uuid, const std::string& expected_bdf) {
    if (staging_size > max_staging_size) throw std::runtime_error("staging capacity limit exceeded");
    if (expected_size != gather_offsets.size() * row_bytes) {
        throw std::runtime_error("expected payload size disagrees with gather geometry");
    }
    std::uint64_t range_total = 0;
    for (const auto& range : ranges) {
        if ((range.file_offset | range.size | range.staging_offset) & 4095U) {
            throw std::runtime_error("cuFile range is not 4096-byte aligned");
        }
        if (!range.size || range.staging_offset + range.size > staging_size) {
            throw std::runtime_error("cuFile range exceeds staging allocation");
        }
        range_total += range.size;
    }
    for (auto offset : gather_offsets) {
        if (offset + row_bytes > staging_size) throw std::runtime_error("gather span exceeds staging allocation");
    }

    Resources resource;
    Result result;
    check_cuda(cuInit(0), "cuInit");
    int device_count = 0;
    check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count != 1) {
        throw std::runtime_error("exactly one CUDA-visible GPU is required, got " + std::to_string(device_count));
    }
    check_cuda(cuDeviceGet(&resource.device, 0), "cuDeviceGet");
    CUuuid uuid{};
    char bdf[32]{};
    check_cuda(cuDeviceGetUuid(&uuid, resource.device), "cuDeviceGetUuid");
    check_cuda(cuDeviceGetPCIBusId(bdf, sizeof(bdf), resource.device), "cuDeviceGetPCIBusId");
    result.gpu_uuid = format_uuid(uuid);
    result.gpu_bdf = normalize_bdf(bdf);
    if (!expected_uuid.empty() && lower(result.gpu_uuid) != lower(expected_uuid)) {
        throw std::runtime_error("target GPU UUID mismatch: " + result.gpu_uuid + " != " + expected_uuid);
    }
    if (!expected_bdf.empty() && result.gpu_bdf != normalize_bdf(expected_bdf)) {
        throw std::runtime_error("target GPU BDF mismatch: " + result.gpu_bdf + " != " + normalize_bdf(expected_bdf));
    }
    check_cuda(cuDevicePrimaryCtxRetain(&resource.context, resource.device), "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(resource.context), "cuCtxSetCurrent");

    if (expected_size == 0) return result;
    if (staging_size == 0 || ranges.empty()) throw std::runtime_error("non-empty request has no cuFile ranges");
    resource.fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (resource.fd < 0) throw std::runtime_error("open(O_DIRECT) failed: " + std::string(std::strerror(errno)));
    check_cufile(cuFileDriverOpen(), "cuFileDriverOpen");
    resource.driver_open = true;
    CUfileDrvProps_t properties{};
    check_cufile(cuFileDriverGetProperties(&properties), "cuFileDriverGetProperties");
    result.driver_control_flags = properties.nvfs.dcontrolflags;
    result.compat_mode_allowed = (properties.nvfs.dcontrolflags & (1U << CU_FILE_ALLOW_COMPAT_MODE)) != 0;
    if (result.compat_mode_allowed) throw std::runtime_error("cuFile compatibility mode is enabled");
    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = resource.fd;
    check_cufile(cuFileHandleRegister(&resource.file_handle, &descriptor), "cuFileHandleRegister");
    result.handle_registered = true;
    check_cuda(cuModuleLoadData(&resource.module, kPtx), "cuModuleLoadData(gather PTX)");
    check_cuda(cuMemAlloc(&resource.staging, staging_size), "cuMemAlloc(staging)");
    check_cufile(cuFileBufRegister(reinterpret_cast<void*>(resource.staging), staging_size, 0),
                 "cuFileBufRegister");
    resource.buffer_registered = true;
    result.buffer_registered = true;

    const auto io_start = std::chrono::steady_clock::now();
    for (const auto& range : ranges) {
        const ssize_t got = cuFileRead(resource.file_handle, reinterpret_cast<void*>(resource.staging),
                                       range.size, range.file_offset, range.staging_offset);
        if (got < 0) throw std::runtime_error("cuFileRead failed with " + std::to_string(got));
        if (static_cast<std::uint64_t>(got) != range.size) {
            throw std::runtime_error("cuFileRead short read: " + std::to_string(got) +
                                     " != " + std::to_string(range.size));
        }
        result.bytes_read += static_cast<std::uint64_t>(got);
    }
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(after cuFileRead)");
    result.io_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - io_start).count();
    if (result.bytes_read != range_total) throw std::runtime_error("aggregate cuFile byte count mismatch");

    const std::size_t count = gather_offsets.size();
    std::vector<std::uint64_t> destinations(count), lengths(count, row_bytes);
    for (std::size_t index = 0; index < count; ++index) destinations[index] = index * row_bytes;
    check_cuda(cuMemAlloc(&resource.output, expected_size), "cuMemAlloc(output)");
    check_cuda(cuMemAlloc(&resource.expected, expected_size), "cuMemAlloc(expected)");
    check_cuda(cuMemAlloc(&resource.sources, count * sizeof(std::uint64_t)), "cuMemAlloc(sources)");
    check_cuda(cuMemAlloc(&resource.destinations, count * sizeof(std::uint64_t)), "cuMemAlloc(destinations)");
    check_cuda(cuMemAlloc(&resource.lengths, count * sizeof(std::uint64_t)), "cuMemAlloc(lengths)");
    check_cuda(cuMemAlloc(&resource.mismatches, sizeof(std::uint32_t)), "cuMemAlloc(mismatches)");
    check_cuda(cuMemcpyHtoD(resource.expected, expected_host, expected_size), "cuMemcpyHtoD(expected)");
    check_cuda(cuMemcpyHtoD(resource.sources, gather_offsets.data(), count * sizeof(std::uint64_t)),
               "cuMemcpyHtoD(sources)");
    check_cuda(cuMemcpyHtoD(resource.destinations, destinations.data(), count * sizeof(std::uint64_t)),
               "cuMemcpyHtoD(destinations)");
    check_cuda(cuMemcpyHtoD(resource.lengths, lengths.data(), count * sizeof(std::uint64_t)),
               "cuMemcpyHtoD(lengths)");
    check_cuda(cuMemsetD32(resource.mismatches, 0, 1), "cuMemsetD32(mismatches)");
    CUfunction gather = nullptr, compare = nullptr;
    check_cuda(cuModuleGetFunction(&gather, resource.module, "gather_spans"), "cuModuleGetFunction(gather)");
    check_cuda(cuModuleGetFunction(&compare, resource.module, "compare_bytes"), "cuModuleGetFunction(compare)");
    unsigned int count_u32 = static_cast<unsigned int>(count);
    void* gather_args[] = {&resource.staging, &resource.output, &resource.sources,
                           &resource.destinations, &resource.lengths, &count_u32};
    check_cuda(cuLaunchKernel(gather, count_u32, 1, 1, 256, 1, 1, 0, nullptr, gather_args, nullptr),
               "cuLaunchKernel(gather)");
    void* compare_args[] = {&resource.output, &resource.expected, &expected_size, &resource.mismatches};
    const unsigned int blocks = static_cast<unsigned int>((expected_size + 255) / 256);
    check_cuda(cuLaunchKernel(compare, blocks, 1, 1, 256, 1, 1, 0, nullptr, compare_args, nullptr),
               "cuLaunchKernel(compare)");
    check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(after compare)");
    check_cuda(cuMemcpyDtoH(&result.mismatch_count, resource.mismatches, sizeof(result.mismatch_count)),
               "cuMemcpyDtoH(mismatch count)");
    return result;
}

bool parse_u64(PyObject* value, std::uint64_t* output) {
    const unsigned long long parsed = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return false;
    *output = static_cast<std::uint64_t>(parsed);
    return true;
}

PyObject* py_read_and_compare(PyObject*, PyObject* args) {
    const char* path = nullptr;
    PyObject* ranges_object = nullptr;
    PyObject* offsets_object = nullptr;
    unsigned long long row_bytes = 0;
    Py_buffer expected{};
    unsigned long long staging_size = 0, max_staging_size = 0;
    const char* expected_uuid = nullptr;
    const char* expected_bdf = nullptr;
    if (!PyArg_ParseTuple(args, "sOOKy*KKss", &path, &ranges_object, &offsets_object,
                          &row_bytes, &expected, &staging_size, &max_staging_size,
                          &expected_uuid, &expected_bdf)) return nullptr;
    PyObject* ranges_fast = PySequence_Fast(ranges_object, "ranges must be a sequence");
    PyObject* offsets_fast = PySequence_Fast(offsets_object, "gather_offsets must be a sequence");
    if (!ranges_fast || !offsets_fast) {
        Py_XDECREF(ranges_fast);
        Py_XDECREF(offsets_fast);
        PyBuffer_Release(&expected);
        return nullptr;
    }
    std::vector<Range> ranges;
    std::vector<std::uint64_t> offsets;
    bool parsed = true;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(ranges_fast) && parsed; ++i) {
        PyObject* tuple = PySequence_Fast(PySequence_Fast_GET_ITEM(ranges_fast, i),
                                          "each range must be a sequence");
        if (!tuple) { parsed = false; break; }
        if (PySequence_Fast_GET_SIZE(tuple) != 3) {
            PyErr_SetString(PyExc_ValueError, "each range needs file_offset, size, staging_offset");
            parsed = false;
        } else {
            Range range{};
            parsed = parse_u64(PySequence_Fast_GET_ITEM(tuple, 0), &range.file_offset) &&
                     parse_u64(PySequence_Fast_GET_ITEM(tuple, 1), &range.size) &&
                     parse_u64(PySequence_Fast_GET_ITEM(tuple, 2), &range.staging_offset);
            if (parsed) ranges.push_back(range);
        }
        Py_DECREF(tuple);
    }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(offsets_fast) && parsed; ++i) {
        std::uint64_t value = 0;
        parsed = parse_u64(PySequence_Fast_GET_ITEM(offsets_fast, i), &value);
        if (parsed) offsets.push_back(value);
    }
    Py_DECREF(ranges_fast);
    Py_DECREF(offsets_fast);
    if (!parsed) {
        PyBuffer_Release(&expected);
        return nullptr;
    }
    Result result;
    std::string failure;
    PyThreadState* state = PyEval_SaveThread();
    try {
        result = run(path, ranges, offsets, row_bytes,
                     static_cast<const unsigned char*>(expected.buf), expected.len,
                     staging_size, max_staging_size, expected_uuid, expected_bdf);
    } catch (const std::exception& exc) {
        failure = exc.what();
    }
    PyEval_RestoreThread(state);
    PyBuffer_Release(&expected);
    if (!failure.empty()) {
        PyErr_SetString(PyExc_RuntimeError, failure.c_str());
        return nullptr;
    }
    PyObject* dictionary = PyDict_New();
    auto set = [&](const char* key, PyObject* value) {
        PyDict_SetItemString(dictionary, key, value);
        Py_DECREF(value);
    };
    set("bytes_read", PyLong_FromUnsignedLongLong(result.bytes_read));
    set("mismatch_count", PyLong_FromUnsignedLong(result.mismatch_count));
    set("io_seconds", PyFloat_FromDouble(result.io_seconds));
    set("gpu_uuid", PyUnicode_FromString(result.gpu_uuid.c_str()));
    set("gpu_bdf", PyUnicode_FromString(result.gpu_bdf.c_str()));
    set("cufile_handle_registered", PyBool_FromLong(result.handle_registered));
    set("cufile_buffer_registered", PyBool_FromLong(result.buffer_registered));
    set("compat_mode_allowed", PyBool_FromLong(result.compat_mode_allowed));
    set("driver_control_flags", PyLong_FromUnsignedLong(result.driver_control_flags));
    set("cpu_row_payload_returned", PyBool_FromLong(0));
    return dictionary;
}

PyMethodDef methods[] = {
    {"read_and_compare", py_read_and_compare, METH_VARARGS,
     "Read aligned pages with cuFile, gather on GPU, and return only mismatch metadata."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
