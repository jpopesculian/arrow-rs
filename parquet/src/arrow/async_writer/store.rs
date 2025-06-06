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

use bytes::Bytes;
use futures::future::BoxFuture;
use std::fmt;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::arrow::async_writer::AsyncFileWriter;
use crate::errors::{ParquetError, Result};
use futures::prelude::*;
use object_store::buffered::BufWriter;
use object_store::path::Path;
use object_store::ObjectStore;
use pin_project_lite::pin_project;
use tokio::io::AsyncWriteExt;

enum ParquetObjectWriterState {
    Ready(Box<BufWriter>),
    Writing(BoxFuture<'static, (BufWriter, Result<()>)>),
    ShuttingDown(BoxFuture<'static, (BufWriter, Result<()>)>),
    Invalid,
}

impl ParquetObjectWriterState {
    fn new(w: BufWriter) -> Self {
        Self::Ready(Box::new(w))
    }
}

impl fmt::Debug for ParquetObjectWriterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ParquetObjectWriterState::*;
        match self {
            Ready(w) => w.fmt(f),
            Writing(_) => write!(f, "Writing"),
            ShuttingDown(_) => write!(f, "ShuttingDown"),
            Invalid => write!(f, "Invalid"),
        }
    }
}

pin_project! {
/// [`ParquetObjectWriter`] for writing to parquet to [`ObjectStore`]
///
/// ```
/// # use arrow_array::{ArrayRef, Int64Array, RecordBatch};
/// # use object_store::memory::InMemory;
/// # use object_store::path::Path;
/// # use object_store::ObjectStore;
/// # use std::sync::Arc;
///
/// # use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
/// # use parquet::arrow::async_writer::ParquetObjectWriter;
/// # use parquet::arrow::AsyncArrowWriter;
///
/// # #[tokio::main(flavor="current_thread")]
/// # async fn main() {
///     let store = Arc::new(InMemory::new());
///
///     let col = Arc::new(Int64Array::from_iter_values([1, 2, 3])) as ArrayRef;
///     let to_write = RecordBatch::try_from_iter([("col", col)]).unwrap();
///
///     let object_store_writer = ParquetObjectWriter::new(store.clone(), Path::from("test"));
///     let mut writer =
///         AsyncArrowWriter::try_new(object_store_writer, to_write.schema(), None).unwrap();
///     writer.write(&to_write).await.unwrap();
///     writer.close().await.unwrap();
///
///     let buffer = store
///         .get(&Path::from("test"))
///         .await
///         .unwrap()
///         .bytes()
///         .await
///         .unwrap();
///     let mut reader = ParquetRecordBatchReaderBuilder::try_new(buffer)
///         .unwrap()
///         .build()
///         .unwrap();
///     let read = reader.next().unwrap().unwrap();
///
///     assert_eq!(to_write, read);
/// # }
/// ```
#[derive(Debug)]
pub struct ParquetObjectWriter {
    state: ParquetObjectWriterState,
}
}

impl ParquetObjectWriter {
    /// Create a new [`ParquetObjectWriter`] that writes to the specified path in the given store.
    ///
    /// To configure the writer behavior, please build [`BufWriter`] and then use [`Self::from_buf_writer`]
    pub fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        Self::from_buf_writer(BufWriter::new(store, path))
    }

    /// Construct a new ParquetObjectWriter via a existing BufWriter.
    pub fn from_buf_writer(w: BufWriter) -> Self {
        Self {
            state: ParquetObjectWriterState::new(w),
        }
    }

    /// Take ownership of the underlying BufWriter. This will mut the writer in an invalid state.
    ///
    /// This will throw an error if the current state is invalid or unavailable
    pub fn take(&mut self) -> Result<BufWriter> {
        use ParquetObjectWriterState::*;
        match mem::replace(&mut self.state, ParquetObjectWriterState::Invalid) {
            Ready(w) => Ok(*w),
            Writing(fut) => {
                self.state = Writing(fut);
                Err(ParquetError::General(
                    "Cannot consume writer while it is writing".into(),
                ))
            }
            ShuttingDown(fut) => {
                self.state = ShuttingDown(fut);
                Err(ParquetError::General(
                    "Cannot consume writer while it is shutting down".into(),
                ))
            }
            Invalid => Err(ParquetError::General("Writer in invalid state".into())),
        }
    }

    /// Consume the writer and return the underlying BufWriter.
    ///
    /// This will throw an error if the current state is invalid or unavailable
    pub fn into_inner(mut self) -> Result<BufWriter> {
        self.take()
    }
}

impl AsyncFileWriter for ParquetObjectWriter {
    fn poll_write(self: Pin<&mut Self>, cx: &mut Context<'_>, bs: Bytes) -> Poll<Result<()>> {
        use ParquetObjectWriterState::*;
        let this = self.project();
        let mut w = match mem::replace(this.state, ParquetObjectWriterState::Invalid) {
            Ready(w) => *w,
            Writing(mut fut) => match fut.as_mut().poll(cx) {
                Poll::Ready((w, res)) => {
                    *this.state = Ready(Box::new(w));
                    return Poll::Ready(res);
                }
                Poll::Pending => {
                    *this.state = Writing(fut);
                    return Poll::Pending;
                }
            },
            ShuttingDown(mut fut) => match fut.as_mut().poll(cx) {
                Poll::Ready((w, res)) => match res {
                    Ok(()) => w,
                    Err(err) => {
                        return Poll::Ready(Err(err));
                    }
                },
                Poll::Pending => {
                    *this.state = ShuttingDown(fut);
                    return Poll::Pending;
                }
            },
            Invalid => {
                return Poll::Ready(Err(ParquetError::General("Writer in invalid state".into())));
            }
        };
        let mut fut = async move {
            let res = w
                .put(bs)
                .await
                .map_err(|err| ParquetError::External(Box::new(err)));
            (w, res)
        }
        .boxed();
        match fut.as_mut().poll(cx) {
            Poll::Ready((w, res)) => {
                *this.state = Ready(Box::new(w));
                Poll::Ready(res)
            }
            Poll::Pending => {
                *this.state = Writing(fut);
                Poll::Pending
            }
        }
    }

    fn poll_complete(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        use ParquetObjectWriterState::*;
        let this = self.project();
        let mut w = match mem::replace(this.state, ParquetObjectWriterState::Invalid) {
            Ready(w) => *w,
            Writing(mut fut) => match fut.as_mut().poll(cx) {
                Poll::Ready((w, res)) => match res {
                    Ok(()) => w,
                    Err(err) => {
                        return Poll::Ready(Err(err));
                    }
                },
                Poll::Pending => {
                    *this.state = Writing(fut);
                    return Poll::Pending;
                }
            },
            ShuttingDown(mut fut) => match fut.as_mut().poll(cx) {
                Poll::Ready((w, res)) => {
                    *this.state = Ready(Box::new(w));
                    return Poll::Ready(res);
                }
                Poll::Pending => {
                    *this.state = ShuttingDown(fut);
                    return Poll::Pending;
                }
            },
            Invalid => {
                return Poll::Ready(Err(ParquetError::General("Writer in invalid state".into())));
            }
        };
        let mut fut = async move {
            let res = w
                .shutdown()
                .await
                .map_err(|err| ParquetError::External(Box::new(err)));
            (w, res)
        }
        .boxed();
        match fut.as_mut().poll(cx) {
            Poll::Ready((w, res)) => {
                *this.state = Ready(Box::new(w));
                Poll::Ready(res)
            }
            Poll::Pending => {
                *this.state = ShuttingDown(fut);
                Poll::Pending
            }
        }
    }
}
impl From<BufWriter> for ParquetObjectWriter {
    fn from(w: BufWriter) -> Self {
        Self::from_buf_writer(w)
    }
}
#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use object_store::memory::InMemory;
    use std::sync::Arc;

    use super::*;
    use crate::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use crate::arrow::AsyncArrowWriter;

    #[tokio::test]
    async fn test_async_writer() {
        let store = Arc::new(InMemory::new());

        let col = Arc::new(Int64Array::from_iter_values([1, 2, 3])) as ArrayRef;
        let to_write = RecordBatch::try_from_iter([("col", col)]).unwrap();

        let object_store_writer = ParquetObjectWriter::new(store.clone(), Path::from("test"));
        let mut writer =
            AsyncArrowWriter::try_new(object_store_writer, to_write.schema(), None).unwrap();
        writer.write(&to_write).await.unwrap();
        writer.close().await.unwrap();

        let buffer = store
            .get(&Path::from("test"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(buffer)
            .unwrap()
            .build()
            .unwrap();
        let read = reader.next().unwrap().unwrap();

        assert_eq!(to_write, read);
    }
}
