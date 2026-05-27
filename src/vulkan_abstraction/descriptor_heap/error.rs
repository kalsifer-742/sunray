use crate::error::ErrorSource;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum HeapError {
    OutOfMemory,
}

impl Display for HeapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Heap error {self:?} ")
    }
}

impl Error for HeapError {}

impl From<HeapError> for ErrorSource {
    fn from(val: HeapError) -> Self {
        ErrorSource::DescriptorHeap(val)
    }
}
