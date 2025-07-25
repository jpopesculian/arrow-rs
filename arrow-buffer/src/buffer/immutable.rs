// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::alloc::Layout;
use std::fmt::Debug;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::alloc::{Allocation, Deallocation};
use crate::util::bit_chunk_iterator::{BitChunks, UnalignedBitChunk};
use crate::BufferBuilder;
use crate::{bit_util, bytes::Bytes, native::ArrowNativeType};

#[cfg(feature = "pool")]
use crate::pool::MemoryPool;

use super::ops::bitwise_unary_op_helper;
use super::{MutableBuffer, ScalarBuffer};

/// A contiguous memory region that can be shared with other buffers and across
/// thread boundaries that stores Arrow data.
///
/// `Buffer`s can be sliced and cloned without copying the underlying data and can
/// be created from memory allocated by non-Rust sources such as C/C++.
///
/// # Example: Create a `Buffer` from a `Vec` (without copying)
/// ```
/// # use arrow_buffer::Buffer;
/// let vec: Vec<u32> = vec![1, 2, 3];
/// let buffer = Buffer::from(vec);
/// ```
///
/// # Example: Convert a `Buffer` to a `Vec` (without copying)
///
/// Use [`Self::into_vec`] to convert a `Buffer` back into a `Vec` if there are
/// no other references and the types are aligned correctly.
/// ```
/// # use arrow_buffer::Buffer;
/// # let vec: Vec<u32> = vec![1, 2, 3];
/// # let buffer = Buffer::from(vec);
/// // convert the buffer back into a Vec of u32
/// // note this will fail if the buffer is shared or not aligned correctly
/// let vec: Vec<u32> = buffer.into_vec().unwrap();
/// ```
///
/// # Example: Create a `Buffer` from a [`bytes::Bytes`] (without copying)
///
/// [`bytes::Bytes`] is a common type in the Rust ecosystem for shared memory
/// regions. You can create a buffer from a `Bytes` instance using the `From`
/// implementation, also without copying.
///
/// ```
/// # use arrow_buffer::Buffer;
/// let bytes = bytes::Bytes::from("hello");
/// let buffer = Buffer::from(bytes);
///```
#[derive(Clone, Debug)]
pub struct Buffer {
    /// the internal byte buffer.
    data: Arc<Bytes>,

    /// Pointer into `data` valid
    ///
    /// We store a pointer instead of an offset to avoid pointer arithmetic
    /// which causes LLVM to fail to vectorise code correctly
    ptr: *const u8,

    /// Byte length of the buffer.
    ///
    /// Must be less than or equal to `data.len()`
    length: usize,
}

impl Default for Buffer {
    #[inline]
    fn default() -> Self {
        MutableBuffer::default().into()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl Eq for Buffer {}

unsafe impl Send for Buffer where Bytes: Send {}
unsafe impl Sync for Buffer where Bytes: Sync {}

impl Buffer {
    /// Create a new Buffer from a (internal) `Bytes`
    ///
    /// NOTE despite the same name, `Bytes` is an internal struct in arrow-rs
    /// and is different than [`bytes::Bytes`].
    ///
    /// See examples on [`Buffer`] for ways to create a buffer from a [`bytes::Bytes`].
    #[deprecated(since = "54.1.0", note = "Use Buffer::from instead")]
    pub fn from_bytes(bytes: Bytes) -> Self {
        Self::from(bytes)
    }

    /// Returns the offset, in bytes, of `Self::ptr` to `Self::data`
    ///
    /// self.ptr and self.data can be different after slicing or advancing the buffer.
    pub fn ptr_offset(&self) -> usize {
        // Safety: `ptr` is always in bounds of `data`.
        unsafe { self.ptr.offset_from(self.data.ptr().as_ptr()) as usize }
    }

    /// Returns the pointer to the start of the buffer without the offset.
    pub fn data_ptr(&self) -> NonNull<u8> {
        self.data.ptr()
    }

    /// Returns the number of strong references to the buffer.
    ///
    /// This method is safe but if the buffer is shared across multiple threads
    /// the underlying value could change between calling this method and using
    /// the result.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Create a [`Buffer`] from the provided [`Vec`] without copying
    #[inline]
    pub fn from_vec<T: ArrowNativeType>(vec: Vec<T>) -> Self {
        MutableBuffer::from(vec).into()
    }

    /// Initializes a [Buffer] from a slice of items.
    pub fn from_slice_ref<U: ArrowNativeType, T: AsRef<[U]>>(items: T) -> Self {
        let slice = items.as_ref();
        let capacity = std::mem::size_of_val(slice);
        let mut buffer = MutableBuffer::with_capacity(capacity);
        buffer.extend_from_slice(slice);
        buffer.into()
    }

    /// Creates a buffer from an existing memory region.
    ///
    /// Ownership of the memory is tracked via reference counting
    /// and the memory will be freed using the `drop` method of
    /// [crate::alloc::Allocation] when the reference count reaches zero.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to raw parts
    /// * `len` - Length of raw parts in **bytes**
    /// * `owner` - A [crate::alloc::Allocation] which is responsible for freeing that data
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len` bytes
    pub unsafe fn from_custom_allocation(
        ptr: NonNull<u8>,
        len: usize,
        owner: Arc<dyn Allocation>,
    ) -> Self {
        Buffer::build_with_arguments(ptr, len, Deallocation::Custom(owner, len))
    }

    /// Auxiliary method to create a new Buffer
    unsafe fn build_with_arguments(
        ptr: NonNull<u8>,
        len: usize,
        deallocation: Deallocation,
    ) -> Self {
        let bytes = Bytes::new(ptr, len, deallocation);
        let ptr = bytes.as_ptr();
        Buffer {
            ptr,
            data: Arc::new(bytes),
            length: len,
        }
    }

    /// Returns the number of bytes in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns the capacity of this buffer.
    /// For externally owned buffers, this returns zero
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Tries to shrink the capacity of the buffer as much as possible, freeing unused memory.
    ///
    /// If the buffer is shared, this is a no-op.
    ///
    /// If the memory was allocated with a custom allocator, this is a no-op.
    ///
    /// If the capacity is already less than or equal to the desired capacity, this is a no-op.
    ///
    /// The memory region will be reallocated using `std::alloc::realloc`.
    pub fn shrink_to_fit(&mut self) {
        let offset = self.ptr_offset();
        let is_empty = self.is_empty();
        let desired_capacity = if is_empty {
            0
        } else {
            // For realloc to work, we cannot free the elements before the offset
            offset + self.len()
        };
        if desired_capacity < self.capacity() {
            if let Some(bytes) = Arc::get_mut(&mut self.data) {
                if bytes.try_realloc(desired_capacity).is_ok() {
                    // Realloc complete - update our pointer into `bytes`:
                    self.ptr = if is_empty {
                        bytes.as_ptr()
                    } else {
                        // SAFETY: we kept all elements leading up to the offset
                        unsafe { bytes.as_ptr().add(offset) }
                    }
                } else {
                    // Failure to reallocate is fine; we just failed to free up memory.
                }
            }
        }
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the byte slice stored in this buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }

    pub(crate) fn deallocation(&self) -> &Deallocation {
        self.data.deallocation()
    }

    /// Returns a new [Buffer] that is a slice of this buffer starting at `offset`.
    ///
    /// This function is `O(1)` and does not copy any data, allowing the
    /// same memory region to be shared between buffers.
    ///
    /// # Panics
    ///
    /// Panics iff `offset` is larger than `len`.
    pub fn slice(&self, offset: usize) -> Self {
        let mut s = self.clone();
        s.advance(offset);
        s
    }

    /// Increases the offset of this buffer by `offset`
    ///
    /// # Panics
    ///
    /// Panics iff `offset` is larger than `len`.
    #[inline]
    pub fn advance(&mut self, offset: usize) {
        assert!(
            offset <= self.length,
            "the offset of the new Buffer cannot exceed the existing length: offset={} length={}",
            offset,
            self.length
        );
        self.length -= offset;
        // Safety:
        // This cannot overflow as
        // `self.offset + self.length < self.data.len()`
        // `offset < self.length`
        self.ptr = unsafe { self.ptr.add(offset) };
    }

    /// Returns a new [Buffer] that is a slice of this buffer starting at `offset`,
    /// with `length` bytes.
    ///
    /// This function is `O(1)` and does not copy any data, allowing the same
    /// memory region to be shared between buffers.
    ///
    /// # Panics
    /// Panics iff `(offset + length)` is larger than the existing length.
    pub fn slice_with_length(&self, offset: usize, length: usize) -> Self {
        assert!(
            offset.saturating_add(length) <= self.length,
            "the offset of the new Buffer cannot exceed the existing length: slice offset={offset} length={length} selflen={}",
            self.length
        );
        // Safety:
        // offset + length <= self.length
        let ptr = unsafe { self.ptr.add(offset) };
        Self {
            data: self.data.clone(),
            ptr,
            length,
        }
    }

    /// Returns a pointer to the start of this buffer.
    ///
    /// Note that this should be used cautiously, and the returned pointer should not be
    /// stored anywhere, to avoid dangling pointers.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// View buffer as a slice of a specific type.
    ///
    /// # Panics
    ///
    /// This function panics if the underlying buffer is not aligned
    /// correctly for type `T`.
    pub fn typed_data<T: ArrowNativeType>(&self) -> &[T] {
        // SAFETY
        // ArrowNativeType is trivially transmutable, is sealed to prevent potentially incorrect
        // implementation outside this crate, and this method checks alignment
        let (prefix, offsets, suffix) = unsafe { self.as_slice().align_to::<T>() };
        assert!(prefix.is_empty() && suffix.is_empty());
        offsets
    }

    /// Returns a slice of this buffer starting at a certain bit offset.
    /// If the offset is byte-aligned the returned buffer is a shallow clone,
    /// otherwise a new buffer is allocated and filled with a copy of the bits in the range.
    pub fn bit_slice(&self, offset: usize, len: usize) -> Self {
        if offset % 8 == 0 {
            return self.slice_with_length(offset / 8, bit_util::ceil(len, 8));
        }

        bitwise_unary_op_helper(self, offset, len, |a| a)
    }

    /// Returns a `BitChunks` instance which can be used to iterate over this buffers bits
    /// in larger chunks and starting at arbitrary bit offsets.
    /// Note that both `offset` and `length` are measured in bits.
    pub fn bit_chunks(&self, offset: usize, len: usize) -> BitChunks {
        BitChunks::new(self.as_slice(), offset, len)
    }

    /// Returns the number of 1-bits in this buffer, starting from `offset` with `length` bits
    /// inspected. Note that both `offset` and `length` are measured in bits.
    pub fn count_set_bits_offset(&self, offset: usize, len: usize) -> usize {
        UnalignedBitChunk::new(self.as_slice(), offset, len).count_ones()
    }

    /// Returns `MutableBuffer` for mutating the buffer if this buffer is not shared.
    /// Returns `Err` if this is shared or its allocation is from an external source or
    /// it is not allocated with alignment [`ALIGNMENT`]
    ///
    /// [`ALIGNMENT`]: crate::alloc::ALIGNMENT
    pub fn into_mutable(self) -> Result<MutableBuffer, Self> {
        let ptr = self.ptr;
        let length = self.length;
        Arc::try_unwrap(self.data)
            .and_then(|bytes| {
                // The pointer of underlying buffer should not be offset.
                assert_eq!(ptr, bytes.ptr().as_ptr());
                MutableBuffer::from_bytes(bytes).map_err(Arc::new)
            })
            .map_err(|bytes| Buffer {
                data: bytes,
                ptr,
                length,
            })
    }

    /// Converts self into a `Vec`, if possible.
    ///
    /// This can be used to reuse / mutate the underlying data.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if
    /// 1. this buffer does not have the same [`Layout`] as the destination Vec
    /// 2. contains a non-zero offset
    /// 3. The buffer is shared
    pub fn into_vec<T: ArrowNativeType>(self) -> Result<Vec<T>, Self> {
        let layout = match self.data.deallocation() {
            Deallocation::Standard(l) => l,
            _ => return Err(self), // Custom allocation
        };

        if self.ptr != self.data.as_ptr() {
            return Err(self); // Data is offset
        }

        let v_capacity = layout.size() / std::mem::size_of::<T>();
        match Layout::array::<T>(v_capacity) {
            Ok(expected) if layout == &expected => {}
            _ => return Err(self), // Incorrect layout
        }

        let length = self.length;
        let ptr = self.ptr;
        let v_len = self.length / std::mem::size_of::<T>();

        Arc::try_unwrap(self.data)
            .map(|bytes| unsafe {
                let ptr = bytes.ptr().as_ptr() as _;
                std::mem::forget(bytes);
                // Safety
                // Verified that bytes layout matches that of Vec
                Vec::from_raw_parts(ptr, v_len, v_capacity)
            })
            .map_err(|bytes| Buffer {
                data: bytes,
                ptr,
                length,
            })
    }

    /// Returns true if this [`Buffer`] is equal to `other`, using pointer comparisons
    /// to determine buffer equality. This is cheaper than `PartialEq::eq` but may
    /// return false when the arrays are logically equal
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.length == other.length
    }

    /// Register this [`Buffer`] with the provided [`MemoryPool`]
    ///
    /// This claims the memory used by this buffer in the pool, allowing for
    /// accurate accounting of memory usage. Any prior reservation will be
    /// released so this works well when the buffer is being shared among
    /// multiple arrays.
    #[cfg(feature = "pool")]
    pub fn claim(&self, pool: &dyn MemoryPool) {
        self.data.claim(pool)
    }
}

/// Note that here we deliberately do not implement
/// `impl<T: AsRef<[u8]>> From<T> for Buffer`
/// As it would accept `Buffer::from(vec![...])` that would cause an unexpected copy.
/// Instead, we ask user to be explicit when copying is occurring, e.g., `Buffer::from(vec![...].to_byte_slice())`.
/// For zero-copy conversion, user should use `Buffer::from_vec(vec![...])`.
///
/// Since we removed impl for `AsRef<u8>`, we added the following three specific implementations to reduce API breakage.
/// See <https://github.com/apache/arrow-rs/issues/6033> for more discussion on this.
impl From<&[u8]> for Buffer {
    fn from(p: &[u8]) -> Self {
        Self::from_slice_ref(p)
    }
}

impl<const N: usize> From<[u8; N]> for Buffer {
    fn from(p: [u8; N]) -> Self {
        Self::from_slice_ref(p)
    }
}

impl<const N: usize> From<&[u8; N]> for Buffer {
    fn from(p: &[u8; N]) -> Self {
        Self::from_slice_ref(p)
    }
}

impl<T: ArrowNativeType> From<Vec<T>> for Buffer {
    fn from(value: Vec<T>) -> Self {
        Self::from_vec(value)
    }
}

impl<T: ArrowNativeType> From<ScalarBuffer<T>> for Buffer {
    fn from(value: ScalarBuffer<T>) -> Self {
        value.into_inner()
    }
}

/// Convert from internal `Bytes` (not [`bytes::Bytes`]) to `Buffer`
impl From<Bytes> for Buffer {
    #[inline]
    fn from(bytes: Bytes) -> Self {
        let length = bytes.len();
        let ptr = bytes.as_ptr();
        Self {
            data: Arc::new(bytes),
            ptr,
            length,
        }
    }
}

/// Convert from [`bytes::Bytes`], not internal `Bytes` to `Buffer`
impl From<bytes::Bytes> for Buffer {
    fn from(bytes: bytes::Bytes) -> Self {
        let bytes: Bytes = bytes.into();
        Self::from(bytes)
    }
}

/// Create a `Buffer` instance by storing the boolean values into the buffer
impl FromIterator<bool> for Buffer {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        MutableBuffer::from_iter(iter).into()
    }
}

impl std::ops::Deref for Buffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl From<MutableBuffer> for Buffer {
    #[inline]
    fn from(buffer: MutableBuffer) -> Self {
        buffer.into_buffer()
    }
}

impl<T: ArrowNativeType> From<BufferBuilder<T>> for Buffer {
    fn from(mut value: BufferBuilder<T>) -> Self {
        value.finish()
    }
}

impl Buffer {
    /// Creates a [`Buffer`] from an [`Iterator`] with a trusted (upper) length.
    ///
    /// Prefer this to `collect` whenever possible, as it is ~60% faster.
    ///
    /// # Example
    /// ```
    /// # use arrow_buffer::buffer::Buffer;
    /// let v = vec![1u32];
    /// let iter = v.iter().map(|x| x * 2);
    /// let buffer = unsafe { Buffer::from_trusted_len_iter(iter) };
    /// assert_eq!(buffer.len(), 4) // u32 has 4 bytes
    /// ```
    /// # Safety
    /// This method assumes that the iterator's size is correct and is undefined behavior
    /// to use it on an iterator that reports an incorrect length.
    // This implementation is required for two reasons:
    // 1. there is no trait `TrustedLen` in stable rust and therefore
    //    we can't specialize `extend` for `TrustedLen` like `Vec` does.
    // 2. `from_trusted_len_iter` is faster.
    #[inline]
    pub unsafe fn from_trusted_len_iter<T: ArrowNativeType, I: Iterator<Item = T>>(
        iterator: I,
    ) -> Self {
        MutableBuffer::from_trusted_len_iter(iterator).into()
    }

    /// Creates a [`Buffer`] from an [`Iterator`] with a trusted (upper) length or errors
    /// if any of the items of the iterator is an error.
    /// Prefer this to `collect` whenever possible, as it is ~60% faster.
    /// # Safety
    /// This method assumes that the iterator's size is correct and is undefined behavior
    /// to use it on an iterator that reports an incorrect length.
    #[inline]
    pub unsafe fn try_from_trusted_len_iter<
        E,
        T: ArrowNativeType,
        I: Iterator<Item = Result<T, E>>,
    >(
        iterator: I,
    ) -> Result<Self, E> {
        Ok(MutableBuffer::try_from_trusted_len_iter(iterator)?.into())
    }
}

impl<T: ArrowNativeType> FromIterator<T> for Buffer {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec = Vec::from_iter(iter);
        Buffer::from_vec(vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::i256;
    use std::panic::{RefUnwindSafe, UnwindSafe};
    use std::thread;

    use super::*;

    #[test]
    fn test_buffer_data_equality() {
        let buf1 = Buffer::from(&[0, 1, 2, 3, 4]);
        let buf2 = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(buf1, buf2);

        // slice with same offset and same length should still preserve equality
        let buf3 = buf1.slice(2);
        assert_ne!(buf1, buf3);
        let buf4 = buf2.slice_with_length(2, 3);
        assert_eq!(buf3, buf4);

        // Different capacities should still preserve equality
        let mut buf2 = MutableBuffer::new(65);
        buf2.extend_from_slice(&[0u8, 1, 2, 3, 4]);

        let buf2 = buf2.into();
        assert_eq!(buf1, buf2);

        // unequal because of different elements
        let buf2 = Buffer::from(&[0, 0, 2, 3, 4]);
        assert_ne!(buf1, buf2);

        // unequal because of different length
        let buf2 = Buffer::from(&[0, 1, 2, 3]);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_from_raw_parts() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(5, buf.len());
        assert!(!buf.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.as_slice());
    }

    #[test]
    fn test_from_vec() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        assert_eq!(5, buf.len());
        assert!(!buf.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf.as_slice());
    }

    #[test]
    fn test_copy() {
        let buf = Buffer::from(&[0, 1, 2, 3, 4]);
        let buf2 = buf;
        assert_eq!(5, buf2.len());
        assert_eq!(64, buf2.capacity());
        assert!(!buf2.as_ptr().is_null());
        assert_eq!([0, 1, 2, 3, 4], buf2.as_slice());
    }

    #[test]
    fn test_slice() {
        let buf = Buffer::from(&[2, 4, 6, 8, 10]);
        let buf2 = buf.slice(2);

        assert_eq!([6, 8, 10], buf2.as_slice());
        assert_eq!(3, buf2.len());
        assert_eq!(unsafe { buf.as_ptr().offset(2) }, buf2.as_ptr());

        let buf3 = buf2.slice_with_length(1, 2);
        assert_eq!([8, 10], buf3.as_slice());
        assert_eq!(2, buf3.len());
        assert_eq!(unsafe { buf.as_ptr().offset(3) }, buf3.as_ptr());

        let buf4 = buf.slice(5);
        let empty_slice: [u8; 0] = [];
        assert_eq!(empty_slice, buf4.as_slice());
        assert_eq!(0, buf4.len());
        assert!(buf4.is_empty());
        assert_eq!(buf2.slice_with_length(2, 1).as_slice(), &[10]);
    }

    #[test]
    fn test_shrink_to_fit() {
        let original = Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(original.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(original.capacity(), 64);

        let slice = original.slice_with_length(2, 3);
        drop(original); // Make sure the buffer isn't shared (or shrink_to_fit won't work)
        assert_eq!(slice.as_slice(), &[2, 3, 4]);
        assert_eq!(slice.capacity(), 64);

        let mut shrunk = slice;
        shrunk.shrink_to_fit();
        assert_eq!(shrunk.as_slice(), &[2, 3, 4]);
        assert_eq!(shrunk.capacity(), 5); // shrink_to_fit is allowed to keep the elements before the offset

        // Test that we can handle empty slices:
        let empty_slice = shrunk.slice_with_length(1, 0);
        drop(shrunk); // Make sure the buffer isn't shared (or shrink_to_fit won't work)
        assert_eq!(empty_slice.as_slice(), &[]);
        assert_eq!(empty_slice.capacity(), 5);

        let mut shrunk_empty = empty_slice;
        shrunk_empty.shrink_to_fit();
        assert_eq!(shrunk_empty.as_slice(), &[]);
        assert_eq!(shrunk_empty.capacity(), 0);
    }

    #[test]
    #[should_panic(expected = "the offset of the new Buffer cannot exceed the existing length")]
    fn test_slice_offset_out_of_bound() {
        let buf = Buffer::from(&[2, 4, 6, 8, 10]);
        buf.slice(6);
    }

    #[test]
    fn test_access_concurrently() {
        let buffer = Buffer::from([1, 2, 3, 4, 5]);
        let buffer2 = buffer.clone();
        assert_eq!([1, 2, 3, 4, 5], buffer.as_slice());

        let buffer_copy = thread::spawn(move || {
            // access buffer in another thread.
            buffer
        })
        .join();

        assert!(buffer_copy.is_ok());
        assert_eq!(buffer2, buffer_copy.ok().unwrap());
    }

    macro_rules! check_as_typed_data {
        ($input: expr, $native_t: ty) => {{
            let buffer = Buffer::from_slice_ref($input);
            let slice: &[$native_t] = buffer.typed_data::<$native_t>();
            assert_eq!($input, slice);
        }};
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_as_typed_data() {
        check_as_typed_data!(&[1i8, 3i8, 6i8], i8);
        check_as_typed_data!(&[1u8, 3u8, 6u8], u8);
        check_as_typed_data!(&[1i16, 3i16, 6i16], i16);
        check_as_typed_data!(&[1i32, 3i32, 6i32], i32);
        check_as_typed_data!(&[1i64, 3i64, 6i64], i64);
        check_as_typed_data!(&[1u16, 3u16, 6u16], u16);
        check_as_typed_data!(&[1u32, 3u32, 6u32], u32);
        check_as_typed_data!(&[1u64, 3u64, 6u64], u64);
        check_as_typed_data!(&[1f32, 3f32, 6f32], f32);
        check_as_typed_data!(&[1f64, 3f64, 6f64], f64);
    }

    #[test]
    fn test_count_bits() {
        assert_eq!(0, Buffer::from(&[0b00000000]).count_set_bits_offset(0, 8));
        assert_eq!(8, Buffer::from(&[0b11111111]).count_set_bits_offset(0, 8));
        assert_eq!(3, Buffer::from(&[0b00001101]).count_set_bits_offset(0, 8));
        assert_eq!(
            6,
            Buffer::from(&[0b01001001, 0b01010010]).count_set_bits_offset(0, 16)
        );
        assert_eq!(
            16,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(0, 16)
        );
    }

    #[test]
    fn test_count_bits_slice() {
        assert_eq!(
            0,
            Buffer::from(&[0b11111111, 0b00000000])
                .slice(1)
                .count_set_bits_offset(0, 8)
        );
        assert_eq!(
            8,
            Buffer::from(&[0b11111111, 0b11111111])
                .slice_with_length(1, 1)
                .count_set_bits_offset(0, 8)
        );
        assert_eq!(
            3,
            Buffer::from(&[0b11111111, 0b11111111, 0b00001101])
                .slice(2)
                .count_set_bits_offset(0, 8)
        );
        assert_eq!(
            6,
            Buffer::from(&[0b11111111, 0b01001001, 0b01010010])
                .slice_with_length(1, 2)
                .count_set_bits_offset(0, 16)
        );
        assert_eq!(
            16,
            Buffer::from(&[0b11111111, 0b11111111, 0b11111111, 0b11111111])
                .slice(2)
                .count_set_bits_offset(0, 16)
        );
    }

    #[test]
    fn test_count_bits_offset_slice() {
        assert_eq!(8, Buffer::from(&[0b11111111]).count_set_bits_offset(0, 8));
        assert_eq!(3, Buffer::from(&[0b11111111]).count_set_bits_offset(0, 3));
        assert_eq!(5, Buffer::from(&[0b11111111]).count_set_bits_offset(3, 5));
        assert_eq!(1, Buffer::from(&[0b11111111]).count_set_bits_offset(3, 1));
        assert_eq!(0, Buffer::from(&[0b11111111]).count_set_bits_offset(8, 0));
        assert_eq!(2, Buffer::from(&[0b01010101]).count_set_bits_offset(0, 3));
        assert_eq!(
            16,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(0, 16)
        );
        assert_eq!(
            10,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(0, 10)
        );
        assert_eq!(
            10,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(3, 10)
        );
        assert_eq!(
            8,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(8, 8)
        );
        assert_eq!(
            5,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(11, 5)
        );
        assert_eq!(
            0,
            Buffer::from(&[0b11111111, 0b11111111]).count_set_bits_offset(16, 0)
        );
        assert_eq!(
            2,
            Buffer::from(&[0b01101101, 0b10101010]).count_set_bits_offset(7, 5)
        );
        assert_eq!(
            4,
            Buffer::from(&[0b01101101, 0b10101010]).count_set_bits_offset(7, 9)
        );
    }

    #[test]
    fn test_unwind_safe() {
        fn assert_unwind_safe<T: RefUnwindSafe + UnwindSafe>() {}
        assert_unwind_safe::<Buffer>()
    }

    #[test]
    fn test_from_foreign_vec() {
        let mut vector = vec![1_i32, 2, 3, 4, 5];
        let buffer = unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(vector.as_mut_ptr() as *mut u8),
                vector.len() * std::mem::size_of::<i32>(),
                Arc::new(vector),
            )
        };

        let slice = buffer.typed_data::<i32>();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        let buffer = buffer.slice(std::mem::size_of::<i32>());

        let slice = buffer.typed_data::<i32>();
        assert_eq!(slice, &[2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "the offset of the new Buffer cannot exceed the existing length")]
    fn slice_overflow() {
        let buffer = Buffer::from(MutableBuffer::from_len_zeroed(12));
        buffer.slice_with_length(2, usize::MAX);
    }

    #[test]
    fn test_vec_interop() {
        // Test empty vec
        let a: Vec<i128> = Vec::new();
        let b = Buffer::from_vec(a);
        b.into_vec::<i128>().unwrap();

        // Test vec with capacity
        let a: Vec<i128> = Vec::with_capacity(20);
        let b = Buffer::from_vec(a);
        let back = b.into_vec::<i128>().unwrap();
        assert_eq!(back.len(), 0);
        assert_eq!(back.capacity(), 20);

        // Test vec with values
        let mut a: Vec<i128> = Vec::with_capacity(3);
        a.extend_from_slice(&[1, 2, 3]);
        let b = Buffer::from_vec(a);
        let back = b.into_vec::<i128>().unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.capacity(), 3);

        // Test vec with values and spare capacity
        let mut a: Vec<i128> = Vec::with_capacity(20);
        a.extend_from_slice(&[1, 4, 7, 8, 9, 3, 6]);
        let b = Buffer::from_vec(a);
        let back = b.into_vec::<i128>().unwrap();
        assert_eq!(back.len(), 7);
        assert_eq!(back.capacity(), 20);

        // Test incorrect alignment
        let a: Vec<i128> = Vec::new();
        let b = Buffer::from_vec(a);
        let b = b.into_vec::<i32>().unwrap_err();
        b.into_vec::<i8>().unwrap_err();

        // Test convert between types with same alignment
        // This is an implementation quirk, but isn't harmful
        // as ArrowNativeType are trivially transmutable
        let a: Vec<i64> = vec![1, 2, 3, 4];
        let b = Buffer::from_vec(a);
        let back = b.into_vec::<u64>().unwrap();
        assert_eq!(back.len(), 4);
        assert_eq!(back.capacity(), 4);

        // i256 has the same layout as i128 so this is valid
        let mut b: Vec<i128> = Vec::with_capacity(4);
        b.extend_from_slice(&[1, 2, 3, 4]);
        let b = Buffer::from_vec(b);
        let back = b.into_vec::<i256>().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.capacity(), 2);

        // Invalid layout
        let b: Vec<i128> = vec![1, 2, 3];
        let b = Buffer::from_vec(b);
        b.into_vec::<i256>().unwrap_err();

        // Invalid layout
        let mut b: Vec<i128> = Vec::with_capacity(5);
        b.extend_from_slice(&[1, 2, 3, 4]);
        let b = Buffer::from_vec(b);
        b.into_vec::<i256>().unwrap_err();

        // Truncates length
        // This is an implementation quirk, but isn't harmful
        let mut b: Vec<i128> = Vec::with_capacity(4);
        b.extend_from_slice(&[1, 2, 3]);
        let b = Buffer::from_vec(b);
        let back = b.into_vec::<i256>().unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back.capacity(), 2);

        // Cannot use aligned allocation
        let b = Buffer::from(MutableBuffer::new(10));
        let b = b.into_vec::<u8>().unwrap_err();
        b.into_vec::<u64>().unwrap_err();

        // Test slicing
        let mut a: Vec<i128> = Vec::with_capacity(20);
        a.extend_from_slice(&[1, 4, 7, 8, 9, 3, 6]);
        let b = Buffer::from_vec(a);
        let slice = b.slice_with_length(0, 64);

        // Shared reference fails
        let slice = slice.into_vec::<i128>().unwrap_err();
        drop(b);

        // Succeeds as no outstanding shared reference
        let back = slice.into_vec::<i128>().unwrap();
        assert_eq!(&back, &[1, 4, 7, 8]);
        assert_eq!(back.capacity(), 20);

        // Slicing by non-multiple length truncates
        let mut a: Vec<i128> = Vec::with_capacity(8);
        a.extend_from_slice(&[1, 4, 7, 3]);

        let b = Buffer::from_vec(a);
        let slice = b.slice_with_length(0, 34);
        drop(b);

        let back = slice.into_vec::<i128>().unwrap();
        assert_eq!(&back, &[1, 4]);
        assert_eq!(back.capacity(), 8);

        // Offset prevents conversion
        let a: Vec<u32> = vec![1, 3, 4, 6];
        let b = Buffer::from_vec(a).slice(2);
        b.into_vec::<u32>().unwrap_err();

        let b = MutableBuffer::new(16).into_buffer();
        let b = b.into_vec::<u8>().unwrap_err(); // Invalid layout
        let b = b.into_vec::<u32>().unwrap_err(); // Invalid layout
        b.into_mutable().unwrap();

        let b = Buffer::from_vec(vec![1_u32, 3, 5]);
        let b = b.into_mutable().unwrap();
        let b = Buffer::from(b);
        let b = b.into_vec::<u32>().unwrap();
        assert_eq!(b, &[1, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "capacity overflow")]
    fn test_from_iter_overflow() {
        let iter_len = usize::MAX / std::mem::size_of::<u64>() + 1;
        let _ = Buffer::from_iter(std::iter::repeat_n(0_u64, iter_len));
    }

    #[test]
    fn bit_slice_length_preserved() {
        // Create a boring buffer
        let buf = Buffer::from_iter(std::iter::repeat_n(true, 64));

        let assert_preserved = |offset: usize, len: usize| {
            let new_buf = buf.bit_slice(offset, len);
            assert_eq!(new_buf.len(), bit_util::ceil(len, 8));

            // if the offset is not byte-aligned, we have to create a deep copy to a new buffer
            // (since the `offset` value inside a Buffer is byte-granular, not bit-granular), so
            // checking the offset should always return 0 if so. If the offset IS byte-aligned, we
            // want to make sure it doesn't unnecessarily create a deep copy.
            if offset % 8 == 0 {
                assert_eq!(new_buf.ptr_offset(), offset / 8);
            } else {
                assert_eq!(new_buf.ptr_offset(), 0);
            }
        };

        // go through every available value for offset
        for o in 0..=64 {
            // and go through every length that could accompany that offset - we can't have a
            // situation where offset + len > 64, because that would go past the end of the buffer,
            // so we use the map to ensure it's in range.
            for l in (o..=64).map(|l| l - o) {
                // and we just want to make sure every one of these keeps its offset and length
                // when neeeded
                assert_preserved(o, l);
            }
        }
    }

    #[test]
    fn test_strong_count() {
        let buffer = Buffer::from_iter(std::iter::repeat_n(0_u8, 100));
        assert_eq!(buffer.strong_count(), 1);

        let buffer2 = buffer.clone();
        assert_eq!(buffer.strong_count(), 2);

        let buffer3 = buffer2.clone();
        assert_eq!(buffer.strong_count(), 3);

        drop(buffer);
        assert_eq!(buffer2.strong_count(), 2);
        assert_eq!(buffer3.strong_count(), 2);

        // Strong count does not increase on move
        let capture = move || {
            assert_eq!(buffer3.strong_count(), 2);
        };

        capture();
        assert_eq!(buffer2.strong_count(), 2);

        drop(capture);
        assert_eq!(buffer2.strong_count(), 1);
    }
}
