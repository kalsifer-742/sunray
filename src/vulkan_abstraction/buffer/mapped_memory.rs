use std::ffi;

//it is up to the owner to drop this object before the memory is no longer valid
pub struct RawMappedMemory {
    p: *mut ffi::c_void,
    byte_size: usize,
}
// unsafe impl Send for RawMappedMemory {}
// unsafe impl Sync for RawMappedMemory {}
impl RawMappedMemory {
    pub unsafe fn new(p: *mut ffi::c_void, byte_size: usize) -> Self {
        Self { p, byte_size }
    }

    pub fn borrow<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.p as *mut T, self.byte_size / size_of::<T>())
        }
    }
}