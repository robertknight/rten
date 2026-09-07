//! 回归测试：手写 protobuf 解析器的 length-delimited 字段不得按线缆声明长度预分配。
//!
//! `ValueReader::read_bytes(len)` 的 `len` 来自不可信的 ONNX 模型文件（protobuf 线缆）。
//! 旧实现 `vec![0; len]` 会在读取真实字节前先申请 `len` 字节，恶意模型把某字段
//! `len` 声明成数 GiB 即可触发内存耗尽型 DoS（CWE-770）。修复后改为按实际可读
//! 字节增量读取，峰值分配量被真实数据量限制。
//!
//! 本测试用进程级计数分配器记录「单次最大分配请求」：
//!   - 修复版：峰值远小于阈值（仅分配真实存在的少量字节）
//!   - 漏洞版（`git stash` 回退 src 后）：`vec![0; 256 MiB]` 单次请求远超阈值 → 断言失败

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rten_onnx::protobuf::{Fields, ValueReader};

/// 进程级分配器：转发给系统分配器，并记录见过的最大单次请求尺寸。
struct CountingAllocator {
    max_request: AtomicUsize,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

impl CountingAllocator {
    fn record(&self, size: usize) {
        let mut cur = self.max_request.load(Ordering::Relaxed);
        while size > cur {
            match self.max_request.compare_exchange_weak(cur, size, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator {
    max_request: AtomicUsize::new(0),
};

#[test]
fn untrusted_length_delimited_does_not_preallocate() {
    // 构造一个 protobuf 消息：字段 1、wire type 2（length-delimited），
    // 声明长度 = 256 MiB（varint 0x80 0x80 0x80 0x80 0x10），但仅 4 字节真实载荷。
    // 模拟恶意 ONNX 模型把 TensorProto.raw_data / String / 内嵌消息的声明长度夸大。
    let mut bytes: Vec<u8> = vec![0x0A]; // tag: field 1, wire type 2
    bytes.extend_from_slice(&[0x80, 0x80, 0x80, 0x80, 0x10]); // len = 4 GiB (4294967296)
    bytes.extend_from_slice(&[1, 2, 3, 4]); // 4 字节真实载荷

    let mut reader = ValueReader::from_buf(bytes);
    let mut fields = Fields::new(&mut reader, None);
    let mut field = fields.next().unwrap().unwrap();
    let result = field.read_bytes();

    // 截断的字段必须优雅报错，而不是静默成功或崩溃。
    assert!(result.is_err(), "截断的 length-delimited 字段应返回错误");

    // 峰值单次分配必须远小于声明长度（4 GiB）。修复版只分配真实存在的 4 字节，
    // 漏洞版会 `vec![0; 4 GiB]` 一次性请求 4294967296 字节。
    let max_request = ALLOC.max_request.load(Ordering::Relaxed);
    assert!(
        max_request < 64 * 1024 * 1024,
        "峰值单次分配 {max_request} 字节 >= 64 MiB 阈值，说明仍按不可信声明长度预分配（CWE-770）"
    );
}
