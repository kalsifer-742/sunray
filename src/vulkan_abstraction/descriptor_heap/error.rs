use std::error::Error;
use std::fmt::{Display, Formatter};
use crate::error::{ErrorSource, SrError, SrResult};

#[derive(Debug )]
pub enum HeapError{
    OutOfMemory,
}

impl Display for HeapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Heap error {self:?} ")
    }
}

impl Error for HeapError{

}


impl Into<ErrorSource> for HeapError {
    fn into(self) -> ErrorSource {
        ErrorSource::DescriptorHeap(self)
    }
}
