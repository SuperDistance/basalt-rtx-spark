import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import struct

win = c.WinDLL("kernel32", use_last_error=True)
driver = c.WinDLL("nvcuda.dll")
PVOID = c.c_void_p
P64 = c.POINTER(c.c_uint64)
MIB = 1 << 20
OPEN_EXISTING = 3
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
FILE_ATTRIBUTE_NORMAL = 0x80
PAGE_READONLY = 2
FILE_MAP_READ = 4
CUDA_DEVICE_MAP = 0x02
CUDA_READ_ONLY = 0x08
CUDA_READ_ONLY_ATTRIBUTE = 113


class PerformanceInfo(c.Structure):
    _fields_ = [("size", w.DWORD)] + [
        (name, c.c_size_t) for name in (
            "commit_total", "commit_limit", "commit_peak", "physical_total", "physical_available",
            "system_cache", "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size",
        )
    ] + [(name, w.DWORD) for name in ("handles", "processes", "threads")]


def signature(module, name, result, *parameters):
    function = getattr(module, name)
    function.restype = result
    function.argtypes = parameters
    return function


open_file = signature(win, "CreateFileW", w.HANDLE, w.LPCWSTR, w.DWORD, w.DWORD,
                      PVOID, w.DWORD, w.DWORD, w.HANDLE)
create_mapping = signature(win, "CreateFileMappingW", w.HANDLE, w.HANDLE, PVOID,
                           w.DWORD, w.DWORD, w.DWORD, w.LPCWSTR)
map_view = signature(win, "MapViewOfFile", PVOID, w.HANDLE, w.DWORD, w.DWORD, w.DWORD, c.c_size_t)
unmap = signature(win, "UnmapViewOfFile", w.BOOL, PVOID)
close = signature(win, "CloseHandle", w.BOOL, w.HANDLE)
performance_info = signature(win, "K32GetPerformanceInfo", w.BOOL, c.POINTER(PerformanceInfo), w.DWORD)

init = signature(driver, "cuInit", c.c_int, c.c_uint)
retain = signature(driver, "cuDevicePrimaryCtxRetain", c.c_int, c.POINTER(PVOID), c.c_int)
set_current = signature(driver, "cuCtxSetCurrent", c.c_int, PVOID)
release_context = signature(driver, "cuDevicePrimaryCtxRelease_v2", c.c_int, c.c_int)
get_attribute = signature(driver, "cuDeviceGetAttribute", c.c_int, c.POINTER(c.c_int), c.c_int, c.c_int)
register = signature(driver, "cuMemHostRegister_v2", c.c_int, PVOID, c.c_size_t, c.c_uint)
unregister = signature(driver, "cuMemHostUnregister", c.c_int, PVOID)
get_pointer = signature(driver, "cuMemHostGetDevicePointer_v2", c.c_int, P64, PVOID, c.c_uint)
read_pointer = signature(driver, "cuMemcpyDtoH_v2", c.c_int, PVOID, c.c_uint64, c.c_size_t)
error_name = signature(driver, "cuGetErrorName", c.c_int, c.c_int, c.POINTER(c.c_char_p))


def cuda_label(status):
    name = c.c_char_p()
    return name.value.decode() if error_name(status, c.byref(name)) == 0 and name.value else str(status)


def require(status):
    if status:
        raise RuntimeError(cuda_label(status))


def section_range(path, name, limit):
    with path.open("rb") as model:
        table = model.read(65536)
    magic, version, count, file_size = struct.unpack_from("<8sIIQ", table)
    if magic != b"BASALTMF" or version != 2 or not 0 < count < 512 or path.stat().st_size != file_size:
        raise ValueError("Expected a complete Basalt v2 model file")
    for index in range(count):
        label, offset, size = struct.unpack_from("<64sQQ", table, 64 + index * 128)
        if label.rstrip(b"\0").decode() == name:
            length = min(size, limit) // (2 * MIB) * (2 * MIB)
            if offset % (2 * MIB) or not length or offset + length > file_size:
                raise ValueError("Section alignment or size is invalid")
            return offset, length
    raise ValueError(f"No section named {name!r}")


def windows_memory():
    info = PerformanceInfo()
    info.size = c.sizeof(info)
    if not performance_info(c.byref(info), info.size):
        raise c.WinError(c.get_last_error())
    return {
        "commit_mib": round(info.commit_total * info.page_size / MIB, 2),
        "physical_available_mib": round(info.physical_available * info.page_size / MIB, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Check CUDA access to a read-only Basalt file section on Windows")
    parser.add_argument("model", type=Path)
    parser.add_argument("--section", default="gguf")
    parser.add_argument("--mib", type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.mib <= 512:
        parser.error("--mib must be between 2 and 512")
    offset, length = section_range(args.model, args.section, args.mib * MIB)
    handle = open_file(str(args.model), GENERIC_READ, FILE_SHARE_READ, None,
                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle in (None, PVOID(-1).value):
        raise c.WinError(c.get_last_error())
    mapping = None
    view = None
    gpu = PVOID()
    registered = False
    result = {"model": args.model.name, "section": args.section, "mapping_mib": length // MIB}
    try:
        mapping = create_mapping(handle, None, PAGE_READONLY, 0, 0, None)
        if not mapping:
            raise c.WinError(c.get_last_error())
        view = map_view(mapping, FILE_MAP_READ, offset >> 32, offset & 0xFFFFFFFF, length)
        if not view:
            raise c.WinError(c.get_last_error())
        require(init(0))
        require(retain(c.byref(gpu), 0))
        require(set_current(gpu))
        feature = c.c_int()
        status = get_attribute(c.byref(feature), CUDA_READ_ONLY_ATTRIBUTE, 0)
        result["read_only_register_attribute"] = feature.value if status == 0 else cuda_label(status)

        result["before_registration"] = windows_memory()
        status = register(view, length, CUDA_DEVICE_MAP | CUDA_READ_ONLY)
        result["after_registration"] = windows_memory()
        result["register_read_only_file_view"] = cuda_label(status)
        if status == 0:
            registered = True
            address = c.c_uint64()
            require(get_pointer(c.byref(address), view, 0))
            matches = []
            for page in (0, length // 2, length - 4096):
                expected = c.string_at(view + page, 4096)
                actual = c.create_string_buffer(len(expected))
                require(read_pointer(actual, address.value + page, len(expected)))
                matches.append(actual.raw == expected)
            result["cuda_alias_data_matches"] = all(matches)
            if not all(matches):
                raise RuntimeError("CUDA alias data differs from the file view")
        print(json.dumps(result, indent=2))
    finally:
        if registered:
            unregister(view)
        if gpu:
            set_current(None)
            release_context(0)
        if view:
            unmap(view)
        if mapping:
            close(mapping)
        close(handle)


if __name__ == "__main__":
    main()
