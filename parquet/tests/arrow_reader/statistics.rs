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

//! This file contains an end to end test of extracting statistics from parquet files.
//! It writes data into a parquet file, reads statistics and verifies they are correct

use std::default::Default;
use std::fs::File;
use std::sync::Arc;

use super::make_test_file_rg;
use super::{struct_array, Scenario};
use arrow::compute::kernels::cast_utils::Parser;
use arrow::datatypes::{
    i256, Date32Type, Date64Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use arrow_array::{
    make_array, new_null_array, Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray,
    Date32Array, Date64Array, Decimal128Array, Decimal256Array, FixedSizeBinaryArray, Float16Array,
    Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray,
    LargeStringArray, RecordBatch, StringArray, StringViewArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use half::f16;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{
    ArrowReaderBuilder, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::statistics::{Statistics, ValueStatistics};
use parquet::schema::types::{SchemaDescPtr, SchemaDescriptor};

#[derive(Debug, Default, Clone)]
struct Int64Case {
    /// Number of nulls in the column
    null_values: usize,
    /// Non null values in the range `[no_null_values_start,
    /// no_null_values_end]`, one value for each row
    no_null_values_start: i64,
    no_null_values_end: i64,
    /// Number of rows per row group
    row_per_group: usize,
    /// if specified, overrides default statistics settings
    enable_stats: Option<EnabledStatistics>,
    /// If specified, the number of values in each page
    data_page_row_count_limit: Option<usize>,
}

impl Int64Case {
    /// Return a record batch with i64 with Null values
    /// The first no_null_values_end - no_null_values_start values
    /// are non-null with the specified range, the rest are null
    fn make_int64_batches_with_null(&self) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("i64", DataType::Int64, true)]));

        let v64: Vec<i64> =
            (self.no_null_values_start as _..self.no_null_values_end as _).collect();

        RecordBatch::try_new(
            schema,
            vec![make_array(
                Int64Array::from_iter(
                    v64.into_iter()
                        .map(Some)
                        .chain(std::iter::repeat_n(None, self.null_values)),
                )
                .to_data(),
            )],
        )
        .unwrap()
    }

    // Create a parquet file with the specified settings
    pub fn build(&self) -> ParquetRecordBatchReaderBuilder<File> {
        let batches = vec![self.make_int64_batches_with_null()];
        build_parquet_file(
            self.row_per_group,
            self.enable_stats,
            self.data_page_row_count_limit,
            batches,
        )
    }
}

fn build_parquet_file(
    row_per_group: usize,
    enable_stats: Option<EnabledStatistics>,
    data_page_row_count_limit: Option<usize>,
    batches: Vec<RecordBatch>,
) -> ParquetRecordBatchReaderBuilder<File> {
    let mut output_file = tempfile::Builder::new()
        .prefix("parquert_statistics_test")
        .suffix(".parquet")
        .tempfile()
        .expect("tempfile creation");

    let mut builder = WriterProperties::builder().set_max_row_group_size(row_per_group);
    if let Some(enable_stats) = enable_stats {
        builder = builder.set_statistics_enabled(enable_stats);
    }
    if let Some(data_page_row_count_limit) = data_page_row_count_limit {
        builder = builder.set_data_page_row_count_limit(data_page_row_count_limit);
    }
    let props = builder.build();

    let schema = batches[0].schema();

    let mut writer = ArrowWriter::try_new(&mut output_file, schema, Some(props)).unwrap();

    // if we have a datapage limit send the batches in one at a time to give
    // the writer a chance to be split into multiple pages
    if data_page_row_count_limit.is_some() {
        for batch in &batches {
            for i in 0..batch.num_rows() {
                writer.write(&batch.slice(i, 1)).expect("writing batch");
            }
        }
    } else {
        for batch in &batches {
            writer.write(batch).expect("writing batch");
        }
    }

    let _file_meta = writer.close().unwrap();

    let file = output_file.reopen().unwrap();
    let options = ArrowReaderOptions::new().with_page_index(true);
    ArrowReaderBuilder::try_new_with_options(file, options).unwrap()
}

/// Defines what data to create in a parquet file
#[derive(Debug, Clone, Copy)]
struct TestReader {
    /// What data to create in the parquet file
    scenario: Scenario,
    /// Number of rows per row group
    row_per_group: usize,
}

impl TestReader {
    /// Create a parquet file with the specified data, and return a
    /// ParquetRecordBatchReaderBuilder opened to that file.
    async fn build(self) -> ParquetRecordBatchReaderBuilder<File> {
        let TestReader {
            scenario,
            row_per_group,
        } = self;
        let file = make_test_file_rg(scenario, row_per_group).await;

        // open the file & get the reader
        let file = file.reopen().unwrap();
        let options = ArrowReaderOptions::new().with_page_index(true);
        ArrowReaderBuilder::try_new_with_options(file, options).unwrap()
    }
}

/// Which statistics should we check?
#[derive(Clone, Debug, Copy)]
enum Check {
    /// Extract and check row group statistics
    RowGroup,
    /// Extract and check data page statistics
    DataPage,
    /// Extract and check both row group and data page statistics.
    ///
    /// Note if a row group contains a single data page,
    /// the statistics for row groups and data pages are the same.
    Both,
}

impl Check {
    fn row_group(&self) -> bool {
        match self {
            Self::RowGroup | Self::Both => true,
            Self::DataPage => false,
        }
    }

    fn data_page(&self) -> bool {
        match self {
            Self::DataPage | Self::Both => true,
            Self::RowGroup => false,
        }
    }
}

/// Defines a test case for statistics extraction
struct Test<'a> {
    /// The parquet file reader
    reader: &'a ParquetRecordBatchReaderBuilder<File>,
    expected_min: ArrayRef,
    expected_max: ArrayRef,
    expected_null_counts: UInt64Array,
    expected_row_counts: Option<UInt64Array>,
    expected_max_value_exact: BooleanArray,
    expected_min_value_exact: BooleanArray,
    /// Which column to extract statistics from
    column_name: &'static str,
    /// What statistics should be checked?
    check: Check,
}

impl Test<'_> {
    fn run(self) {
        let converter = StatisticsConverter::try_new(
            self.column_name,
            self.reader.schema(),
            self.reader.parquet_schema(),
        )
        .unwrap();

        self.run_checks(converter);
    }

    fn run_with_schema(self, schema: &Schema) {
        let converter =
            StatisticsConverter::try_new(self.column_name, schema, self.reader.parquet_schema())
                .unwrap();

        self.run_checks(converter);
    }

    fn run_checks(self, converter: StatisticsConverter) {
        let Self {
            reader,
            expected_min,
            expected_max,
            expected_null_counts,
            expected_row_counts,
            expected_max_value_exact: expected_max_exact,
            expected_min_value_exact: expected_min_exact,
            column_name,
            check,
        } = self;

        let row_groups = reader.metadata().row_groups();

        if check.data_page() {
            let column_page_index = reader
                .metadata()
                .column_index()
                .expect("File should have column page indices");

            let column_offset_index = reader
                .metadata()
                .offset_index()
                .expect("File should have column offset indices");

            let row_group_indices: Vec<_> = (0..row_groups.len()).collect();

            let min = converter
                .data_page_mins(column_page_index, column_offset_index, &row_group_indices)
                .unwrap();
            assert_eq!(
                &min, &expected_min,
                "{column_name}: Mismatch with expected data page minimums"
            );

            let max = converter
                .data_page_maxes(column_page_index, column_offset_index, &row_group_indices)
                .unwrap();
            assert_eq!(
                &max, &expected_max,
                "{column_name}: Mismatch with expected data page maximum"
            );

            let null_counts = converter
                .data_page_null_counts(column_page_index, column_offset_index, &row_group_indices)
                .unwrap();

            assert_eq!(
                &null_counts, &expected_null_counts,
                "{column_name}: Mismatch with expected data page null counts. \
                Actual: {null_counts:?}. Expected: {expected_null_counts:?}"
            );

            let row_counts = converter
                .data_page_row_counts(column_offset_index, row_groups, &row_group_indices)
                .unwrap();
            assert_eq!(
                row_counts, expected_row_counts,
                "{column_name}: Mismatch with expected row counts. \
                Actual: {row_counts:?}. Expected: {expected_row_counts:?}"
            );
        }

        if check.row_group() {
            let min = converter.row_group_mins(row_groups).unwrap();
            assert_eq!(
                &min, &expected_min,
                "{column_name}: Mismatch with expected minimums"
            );

            let max = converter.row_group_maxes(row_groups).unwrap();
            assert_eq!(
                &max, &expected_max,
                "{column_name}: Mismatch with expected maximum"
            );

            let null_counts = converter.row_group_null_counts(row_groups).unwrap();
            assert_eq!(
                &null_counts, &expected_null_counts,
                "{column_name}: Mismatch with expected null counts. \
                Actual: {null_counts:?}. Expected: {expected_null_counts:?}"
            );

            let row_counts = converter
                .row_group_row_counts(reader.metadata().row_groups().iter())
                .unwrap();
            assert_eq!(
                row_counts, expected_row_counts,
                "{column_name}: Mismatch with expected row counts. \
                Actual: {row_counts:?}. Expected: {expected_row_counts:?}"
            );

            let is_max_value_exact = converter
                .row_group_is_max_value_exact(reader.metadata().row_groups().iter())
                .unwrap();
            assert_eq!(
                is_max_value_exact, expected_max_exact,
                "{column_name}: Mismatch with expected max value exactness. \
                Actual: {is_max_value_exact:?}. Expected: {expected_max_exact:?}"
            );

            let is_min_value_exact = converter
                .row_group_is_min_value_exact(reader.metadata().row_groups().iter())
                .unwrap();
            assert_eq!(
                is_min_value_exact, expected_min_exact,
                "{column_name}: Mismatch with expected min value exactness. \
                Actual: {is_min_value_exact:?}. Expected: {expected_min_exact:?}"
            );
        }
    }

    /// Run the test and expect a column not found error
    fn run_col_not_found(self) {
        let Self {
            reader,
            expected_min: _,
            expected_max: _,
            expected_null_counts: _,
            expected_row_counts: _,
            column_name,
            ..
        } = self;

        let converter =
            StatisticsConverter::try_new(column_name, reader.schema(), reader.parquet_schema());

        assert!(converter.is_err());
    }
}

// TESTS
//
// Remaining cases
//   f64::NAN

#[tokio::test]
async fn test_max_and_min_value_truncated() {
    let reader = TestReader {
        scenario: Scenario::TruncatedUTF8,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // min is truncated to
        // 1. `"a".repeat(64)`, original value is `"a".repeat(64) + "1"`
        // 2. `"e".repeat(64)`, original value is `"e".repeat(64) + "5"`
        // 3. "j", as expected with no truncation
        expected_min: Arc::new(StringArray::from(vec![
            &("a".repeat(64)),
            &("e".repeat(64)),
            "j",
        ])),
        // max is truncated to
        // 1. `"d".repeat(63) + "e"`, original value is `"d".repeat(64) + "4"`
        // 2. `"i".repeat(63) + "j"`, original value is `"i".repeat(64) + "6"`
        // 3. `"n".repeat(63) + "o"`, original value is `"n".repeat(64) + "14"`
        expected_max: Arc::new(StringArray::from(vec![
            "d".repeat(63) + "e",
            "i".repeat(63) + "j",
            "n".repeat(63) + "o",
        ])),
        // no nulls
        expected_null_counts: UInt64Array::from(vec![1, 0, 0]),
        // 3 rows
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // all max values are truncated
        expected_max_value_exact: BooleanArray::from(vec![false, false, false]),
        // min values are truncated in the first two row groups
        expected_min_value_exact: BooleanArray::from(vec![false, false, true]),
        column_name: "utf8",
        check: Check::Both,
    }
    .run()
}

#[tokio::test]
async fn test_one_row_group_without_null() {
    let reader = Int64Case {
        null_values: 0,
        no_null_values_start: 4,
        no_null_values_end: 7,
        row_per_group: 20,
        ..Default::default()
    }
    .build();

    Test {
        reader: &reader,
        // min is 4
        expected_min: Arc::new(Int64Array::from(vec![4])),
        // max is 6
        expected_max: Arc::new(Int64Array::from(vec![6])),
        // no nulls
        expected_null_counts: UInt64Array::from(vec![0]),
        // 3 rows
        expected_row_counts: Some(UInt64Array::from(vec![3])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true]),
        expected_min_value_exact: BooleanArray::from(vec![true]),
        column_name: "i64",
        check: Check::Both,
    }
    .run()
}

#[tokio::test]
async fn test_one_row_group_with_null_and_negative() {
    let reader = Int64Case {
        null_values: 2,
        no_null_values_start: -1,
        no_null_values_end: 5,
        row_per_group: 20,
        ..Default::default()
    }
    .build();

    Test {
        reader: &reader,
        // min is -1
        expected_min: Arc::new(Int64Array::from(vec![-1])),
        // max is 4
        expected_max: Arc::new(Int64Array::from(vec![4])),
        // 2 nulls
        expected_null_counts: UInt64Array::from(vec![2]),
        // 8 rows
        expected_row_counts: Some(UInt64Array::from(vec![8])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true]),
        expected_min_value_exact: BooleanArray::from(vec![true]),
        column_name: "i64",
        check: Check::Both,
    }
    .run()
}

#[tokio::test]
async fn test_two_row_group_with_null() {
    let reader = Int64Case {
        null_values: 2,
        no_null_values_start: 4,
        no_null_values_end: 17,
        row_per_group: 10,
        ..Default::default()
    }
    .build();

    Test {
        reader: &reader,
        // mins are [4, 14]
        expected_min: Arc::new(Int64Array::from(vec![4, 14])),
        // maxes are [13, 16]
        expected_max: Arc::new(Int64Array::from(vec![13, 16])),
        // nulls are [0, 2]
        expected_null_counts: UInt64Array::from(vec![0, 2]),
        // row counts are [10, 5]
        expected_row_counts: Some(UInt64Array::from(vec![10, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "i64",
        check: Check::Both,
    }
    .run()
}

#[tokio::test]
async fn test_two_row_groups_with_all_nulls_in_one() {
    let reader = Int64Case {
        null_values: 4,
        no_null_values_start: -2,
        no_null_values_end: 2,
        row_per_group: 5,
        ..Default::default()
    }
    .build();

    Test {
        reader: &reader,
        // mins are [-2, null]
        expected_min: Arc::new(Int64Array::from(vec![Some(-2), None])),
        // maxes are [1, null]
        expected_max: Arc::new(Int64Array::from(vec![Some(1), None])),
        // nulls are [1, 3]
        expected_null_counts: UInt64Array::from(vec![1, 3]),
        // row counts are [5, 3]
        expected_row_counts: Some(UInt64Array::from(vec![5, 3])),
        expected_max_value_exact: BooleanArray::from(vec![true, false]),
        expected_min_value_exact: BooleanArray::from(vec![true, false]),
        column_name: "i64",
        check: Check::Both,
    }
    .run()
}

#[tokio::test]
async fn test_multiple_data_pages_nulls_and_negatives() {
    let reader = Int64Case {
        null_values: 3,
        no_null_values_start: -1,
        no_null_values_end: 10,
        row_per_group: 20,
        // limit page row count to 4
        data_page_row_count_limit: Some(4),
        enable_stats: Some(EnabledStatistics::Page),
    }
    .build();

    // Data layout looks like this:
    //
    // page 0: [-1, 0, 1, 2]
    // page 1: [3, 4, 5, 6]
    // page 2: [7, 8, 9, null]
    // page 3: [null, null]
    Test {
        reader: &reader,
        expected_min: Arc::new(Int64Array::from(vec![Some(-1), Some(3), Some(7), None])),
        expected_max: Arc::new(Int64Array::from(vec![Some(2), Some(6), Some(9), None])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 1, 2]),
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 2])),
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, false]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, false]),
        column_name: "i64",
        check: Check::DataPage,
    }
    .run()
}

#[tokio::test]
async fn test_data_page_stats_with_all_null_page() {
    for data_type in &[
        DataType::Boolean,
        DataType::UInt64,
        DataType::UInt32,
        DataType::UInt16,
        DataType::UInt8,
        DataType::Int64,
        DataType::Int32,
        DataType::Int16,
        DataType::Int8,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Second, None),
        DataType::Timestamp(TimeUnit::Millisecond, None),
        DataType::Timestamp(TimeUnit::Microsecond, None),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        DataType::Binary,
        DataType::LargeBinary,
        DataType::FixedSizeBinary(3),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        DataType::Decimal128(8, 2),  // as INT32
        DataType::Decimal128(10, 2), // as INT64
        DataType::Decimal128(20, 2), // as FIXED_LEN_BYTE_ARRAY
        DataType::Decimal256(8, 2),  // as INT32
        DataType::Decimal256(10, 2), // as INT64
        DataType::Decimal256(20, 2), // as FIXED_LEN_BYTE_ARRAY
    ] {
        let batch = RecordBatch::try_from_iter(vec![("col", new_null_array(data_type, 4))])
            .expect("record batch creation");

        let reader = build_parquet_file(4, Some(EnabledStatistics::Page), Some(4), vec![batch]);

        let expected_data_type = match data_type {
            DataType::Dictionary(_, value_type) => value_type.as_ref(),
            _ => data_type,
        };

        // There is one data page with 4 nulls
        // The statistics should be present but null
        Test {
            reader: &reader,
            expected_min: new_null_array(expected_data_type, 1),
            expected_max: new_null_array(expected_data_type, 1),
            expected_null_counts: UInt64Array::from(vec![4]),
            expected_row_counts: Some(UInt64Array::from(vec![4])),
            expected_max_value_exact: BooleanArray::from(vec![false]),
            expected_min_value_exact: BooleanArray::from(vec![false]),
            column_name: "col",
            check: Check::DataPage,
        }
        .run()
    }
}

/////////////// MORE GENERAL TESTS //////////////////////
// . Many columns in a file
// . Differnet data types
// . Different row group sizes

// Four different integer types
#[tokio::test]
async fn test_int_64() {
    // This creates a parquet files of 4 columns named "i8", "i16", "i32", "i64"
    let reader = TestReader {
        scenario: Scenario::Int,
        row_per_group: 5,
    }
    .build()
    .await;

    // since each row has only one data page, the statistics are the same
    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Int64Array::from(vec![-5, -4, 0, 5])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Int64Array::from(vec![-1, 0, 4, 9])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "i64",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_int_32() {
    // This creates a parquet files of 4 columns named "i8", "i16", "i32", "i64"
    let reader = TestReader {
        scenario: Scenario::Int,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Int32Array::from(vec![-5, -4, 0, 5])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Int32Array::from(vec![-1, 0, 4, 9])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "i32",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_int_16() {
    // This creates a parquet files of 4 columns named "i8", "i16", "i32", "i64"
    let reader = TestReader {
        scenario: Scenario::Int,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Int16Array::from(vec![-5, -4, 0, 5])), // panic here because the actual data is Int32Array
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Int16Array::from(vec![-1, 0, 4, 9])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "i16",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_int_8() {
    // This creates a parquet files of 4 columns named "i8", "i16", "i32", "i64"
    let reader = TestReader {
        scenario: Scenario::Int,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Int8Array::from(vec![-5, -4, 0, 5])), // panic here because the actual data is Int32Array
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Int8Array::from(vec![-1, 0, 4, 9])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "i8",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_float_16() {
    // This creates a parquet files of 1 column named f
    let reader = TestReader {
        scenario: Scenario::Float16,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Float16Array::from(vec![
            f16::from_f32(-5.),
            f16::from_f32(-4.),
            f16::from_f32(-0.),
            f16::from_f32(5.),
        ])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Float16Array::from(vec![
            f16::from_f32(-1.),
            f16::from_f32(0.),
            f16::from_f32(4.),
            f16::from_f32(9.),
        ])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "f",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_float_32() {
    // This creates a parquet files of 1 column named f
    let reader = TestReader {
        scenario: Scenario::Float32,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Float32Array::from(vec![-5., -4., -0., 5.0])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Float32Array::from(vec![-1., 0., 4., 9.])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "f",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_float_64() {
    // This creates a parquet files of 1 column named f
    let reader = TestReader {
        scenario: Scenario::Float64,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Float64Array::from(vec![-5., -4., -0., 5.0])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Float64Array::from(vec![-1., 0., 4., 9.])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "f",
        check: Check::Both,
    }
    .run();
}

// timestamp
#[tokio::test]
async fn test_timestamp() {
    // This creates a parquet files of 9 columns named "nanos", "nanos_timezoned", "micros", "micros_timezoned", "millis", "millis_timezoned", "seconds", "seconds_timezoned", "names"
    // "nanos" --> TimestampNanosecondArray
    // "nanos_timezoned" --> TimestampNanosecondArray
    // "micros" --> TimestampMicrosecondArray
    // "micros_timezoned" --> TimestampMicrosecondArray
    // "millis" --> TimestampMillisecondArray
    // "millis_timezoned" --> TimestampMillisecondArray
    // "seconds" --> TimestampSecondArray
    // "seconds_timezoned" --> TimestampSecondArray
    // "names" --> StringArray
    //
    // The file is created by 4 record batches, each has 5 rows.
    // Since the row group size is set to 5, those 4 batches will go into 4 row groups
    // This creates a parquet files of 4 columns named "nanos", "nanos_timezoned", "micros", "micros_timezoned", "millis", "millis_timezoned", "seconds", "seconds_timezoned"
    let reader = TestReader {
        scenario: Scenario::Timestamps,
        row_per_group: 5,
    }
    .build()
    .await;

    let tz = "Pacific/Efate";

    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampNanosecondArray::from(vec![
            TimestampNanosecondType::parse("2020-01-01T01:01:01"),
            TimestampNanosecondType::parse("2020-01-01T01:01:11"),
            TimestampNanosecondType::parse("2020-01-01T01:11:01"),
            TimestampNanosecondType::parse("2020-01-11T01:01:01"),
        ])),
        expected_max: Arc::new(TimestampNanosecondArray::from(vec![
            TimestampNanosecondType::parse("2020-01-02T01:01:01"),
            TimestampNanosecondType::parse("2020-01-02T01:01:11"),
            TimestampNanosecondType::parse("2020-01-02T01:11:01"),
            TimestampNanosecondType::parse("2020-01-12T01:01:01"),
        ])),
        // nulls are [1, 1, 1, 1]
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "nanos",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampNanosecondArray::from(vec![
                TimestampNanosecondType::parse("2020-01-01T01:01:01"),
                TimestampNanosecondType::parse("2020-01-01T01:01:11"),
                TimestampNanosecondType::parse("2020-01-01T01:11:01"),
                TimestampNanosecondType::parse("2020-01-11T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampNanosecondArray::from(vec![
                TimestampNanosecondType::parse("2020-01-02T01:01:01"),
                TimestampNanosecondType::parse("2020-01-02T01:01:11"),
                TimestampNanosecondType::parse("2020-01-02T01:11:01"),
                TimestampNanosecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 1, 1, 1]
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "nanos_timezoned",
        check: Check::Both,
    }
    .run();

    // micros
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampMicrosecondArray::from(vec![
            TimestampMicrosecondType::parse("2020-01-01T01:01:01"),
            TimestampMicrosecondType::parse("2020-01-01T01:01:11"),
            TimestampMicrosecondType::parse("2020-01-01T01:11:01"),
            TimestampMicrosecondType::parse("2020-01-11T01:01:01"),
        ])),
        expected_max: Arc::new(TimestampMicrosecondArray::from(vec![
            TimestampMicrosecondType::parse("2020-01-02T01:01:01"),
            TimestampMicrosecondType::parse("2020-01-02T01:01:11"),
            TimestampMicrosecondType::parse("2020-01-02T01:11:01"),
            TimestampMicrosecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "micros",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampMicrosecondArray::from(vec![
                TimestampMicrosecondType::parse("2020-01-01T01:01:01"),
                TimestampMicrosecondType::parse("2020-01-01T01:01:11"),
                TimestampMicrosecondType::parse("2020-01-01T01:11:01"),
                TimestampMicrosecondType::parse("2020-01-11T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampMicrosecondArray::from(vec![
                TimestampMicrosecondType::parse("2020-01-02T01:01:01"),
                TimestampMicrosecondType::parse("2020-01-02T01:01:11"),
                TimestampMicrosecondType::parse("2020-01-02T01:11:01"),
                TimestampMicrosecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 1, 1, 1]
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "micros_timezoned",
        check: Check::Both,
    }
    .run();

    // millis
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampMillisecondArray::from(vec![
            TimestampMillisecondType::parse("2020-01-01T01:01:01"),
            TimestampMillisecondType::parse("2020-01-01T01:01:11"),
            TimestampMillisecondType::parse("2020-01-01T01:11:01"),
            TimestampMillisecondType::parse("2020-01-11T01:01:01"),
        ])),
        expected_max: Arc::new(TimestampMillisecondArray::from(vec![
            TimestampMillisecondType::parse("2020-01-02T01:01:01"),
            TimestampMillisecondType::parse("2020-01-02T01:01:11"),
            TimestampMillisecondType::parse("2020-01-02T01:11:01"),
            TimestampMillisecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "millis",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampMillisecondArray::from(vec![
                TimestampMillisecondType::parse("2020-01-01T01:01:01"),
                TimestampMillisecondType::parse("2020-01-01T01:01:11"),
                TimestampMillisecondType::parse("2020-01-01T01:11:01"),
                TimestampMillisecondType::parse("2020-01-11T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampMillisecondArray::from(vec![
                TimestampMillisecondType::parse("2020-01-02T01:01:01"),
                TimestampMillisecondType::parse("2020-01-02T01:01:11"),
                TimestampMillisecondType::parse("2020-01-02T01:11:01"),
                TimestampMillisecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 1, 1, 1]
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),

        column_name: "millis_timezoned",
        check: Check::Both,
    }
    .run();

    // seconds
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampSecondArray::from(vec![
            TimestampSecondType::parse("2020-01-01T01:01:01"),
            TimestampSecondType::parse("2020-01-01T01:01:11"),
            TimestampSecondType::parse("2020-01-01T01:11:01"),
            TimestampSecondType::parse("2020-01-11T01:01:01"),
        ])),
        expected_max: Arc::new(TimestampSecondArray::from(vec![
            TimestampSecondType::parse("2020-01-02T01:01:01"),
            TimestampSecondType::parse("2020-01-02T01:01:11"),
            TimestampSecondType::parse("2020-01-02T01:11:01"),
            TimestampSecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),

        column_name: "seconds",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampSecondArray::from(vec![
                TimestampSecondType::parse("2020-01-01T01:01:01"),
                TimestampSecondType::parse("2020-01-01T01:01:11"),
                TimestampSecondType::parse("2020-01-01T01:11:01"),
                TimestampSecondType::parse("2020-01-11T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampSecondArray::from(vec![
                TimestampSecondType::parse("2020-01-02T01:01:01"),
                TimestampSecondType::parse("2020-01-02T01:01:11"),
                TimestampSecondType::parse("2020-01-02T01:11:01"),
                TimestampSecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 1, 1, 1]
        expected_null_counts: UInt64Array::from(vec![1, 1, 1, 1]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),

        column_name: "seconds_timezoned",
        check: Check::Both,
    }
    .run();
}

// timestamp with different row group sizes
#[tokio::test]
async fn test_timestamp_diff_rg_sizes() {
    // This creates a parquet files of 9 columns named "nanos", "nanos_timezoned", "micros", "micros_timezoned", "millis", "millis_timezoned", "seconds", "seconds_timezoned", "names"
    // "nanos" --> TimestampNanosecondArray
    // "nanos_timezoned" --> TimestampNanosecondArray
    // "micros" --> TimestampMicrosecondArray
    // "micros_timezoned" --> TimestampMicrosecondArray
    // "millis" --> TimestampMillisecondArray
    // "millis_timezoned" --> TimestampMillisecondArray
    // "seconds" --> TimestampSecondArray
    // "seconds_timezoned" --> TimestampSecondArray
    // "names" --> StringArray
    //
    // The file is created by 4 record batches (each has a null row), each has 5 rows but then will be split into 3 row groups with size 8, 8, 4
    let reader = TestReader {
        scenario: Scenario::Timestamps,
        row_per_group: 8, // note that the row group size is 8
    }
    .build()
    .await;

    let tz = "Pacific/Efate";

    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampNanosecondArray::from(vec![
            TimestampNanosecondType::parse("2020-01-01T01:01:01"),
            TimestampNanosecondType::parse("2020-01-01T01:11:01"),
            TimestampNanosecondType::parse("2020-01-11T01:02:01"),
        ])),
        expected_max: Arc::new(TimestampNanosecondArray::from(vec![
            TimestampNanosecondType::parse("2020-01-02T01:01:01"),
            TimestampNanosecondType::parse("2020-01-11T01:01:01"),
            TimestampNanosecondType::parse("2020-01-12T01:01:01"),
        ])),
        // nulls are [1, 2, 1]
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        // row counts are [8, 8, 4]
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "nanos",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampNanosecondArray::from(vec![
                TimestampNanosecondType::parse("2020-01-01T01:01:01"),
                TimestampNanosecondType::parse("2020-01-01T01:11:01"),
                TimestampNanosecondType::parse("2020-01-11T01:02:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampNanosecondArray::from(vec![
                TimestampNanosecondType::parse("2020-01-02T01:01:01"),
                TimestampNanosecondType::parse("2020-01-11T01:01:01"),
                TimestampNanosecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 2, 1]
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        // row counts are [8, 8, 4]
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "nanos_timezoned",
        check: Check::Both,
    }
    .run();

    // micros
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampMicrosecondArray::from(vec![
            TimestampMicrosecondType::parse("2020-01-01T01:01:01"),
            TimestampMicrosecondType::parse("2020-01-01T01:11:01"),
            TimestampMicrosecondType::parse("2020-01-11T01:02:01"),
        ])),
        expected_max: Arc::new(TimestampMicrosecondArray::from(vec![
            TimestampMicrosecondType::parse("2020-01-02T01:01:01"),
            TimestampMicrosecondType::parse("2020-01-11T01:01:01"),
            TimestampMicrosecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "micros",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampMicrosecondArray::from(vec![
                TimestampMicrosecondType::parse("2020-01-01T01:01:01"),
                TimestampMicrosecondType::parse("2020-01-01T01:11:01"),
                TimestampMicrosecondType::parse("2020-01-11T01:02:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampMicrosecondArray::from(vec![
                TimestampMicrosecondType::parse("2020-01-02T01:01:01"),
                TimestampMicrosecondType::parse("2020-01-11T01:01:01"),
                TimestampMicrosecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 2, 1]
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        // row counts are [8, 8, 4]
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "micros_timezoned",
        check: Check::Both,
    }
    .run();

    // millis
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampMillisecondArray::from(vec![
            TimestampMillisecondType::parse("2020-01-01T01:01:01"),
            TimestampMillisecondType::parse("2020-01-01T01:11:01"),
            TimestampMillisecondType::parse("2020-01-11T01:02:01"),
        ])),
        expected_max: Arc::new(TimestampMillisecondArray::from(vec![
            TimestampMillisecondType::parse("2020-01-02T01:01:01"),
            TimestampMillisecondType::parse("2020-01-11T01:01:01"),
            TimestampMillisecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "millis",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampMillisecondArray::from(vec![
                TimestampMillisecondType::parse("2020-01-01T01:01:01"),
                TimestampMillisecondType::parse("2020-01-01T01:11:01"),
                TimestampMillisecondType::parse("2020-01-11T01:02:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampMillisecondArray::from(vec![
                TimestampMillisecondType::parse("2020-01-02T01:01:01"),
                TimestampMillisecondType::parse("2020-01-11T01:01:01"),
                TimestampMillisecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 2, 1]
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        // row counts are [8, 8, 4]
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "millis_timezoned",
        check: Check::Both,
    }
    .run();

    // seconds
    Test {
        reader: &reader,
        expected_min: Arc::new(TimestampSecondArray::from(vec![
            TimestampSecondType::parse("2020-01-01T01:01:01"),
            TimestampSecondType::parse("2020-01-01T01:11:01"),
            TimestampSecondType::parse("2020-01-11T01:02:01"),
        ])),
        expected_max: Arc::new(TimestampSecondArray::from(vec![
            TimestampSecondType::parse("2020-01-02T01:01:01"),
            TimestampSecondType::parse("2020-01-11T01:01:01"),
            TimestampSecondType::parse("2020-01-12T01:01:01"),
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "seconds",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(
            TimestampSecondArray::from(vec![
                TimestampSecondType::parse("2020-01-01T01:01:01"),
                TimestampSecondType::parse("2020-01-01T01:11:01"),
                TimestampSecondType::parse("2020-01-11T01:02:01"),
            ])
            .with_timezone(tz),
        ),
        expected_max: Arc::new(
            TimestampSecondArray::from(vec![
                TimestampSecondType::parse("2020-01-02T01:01:01"),
                TimestampSecondType::parse("2020-01-11T01:01:01"),
                TimestampSecondType::parse("2020-01-12T01:01:01"),
            ])
            .with_timezone(tz),
        ),
        // nulls are [1, 2, 1]
        expected_null_counts: UInt64Array::from(vec![1, 2, 1]),
        // row counts are [8, 8, 4]
        expected_row_counts: Some(UInt64Array::from(vec![8, 8, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "seconds_timezoned",
        check: Check::Both,
    }
    .run();
}

// date with different row group sizes
#[tokio::test]
async fn test_dates_32_diff_rg_sizes() {
    // This creates a parquet files of 3 columns named "date32", "date64", "names"
    // "date32" --> Date32Array
    // "date64" --> Date64Array
    // "names" --> StringArray
    //
    // The file is created by 4 record batches (each has a null row), each has 5 rows but then will be split into 2 row groups with size 13, 7
    let reader = TestReader {
        scenario: Scenario::Dates,
        row_per_group: 13,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [2020-01-01, 2020-10-30]
        expected_min: Arc::new(Date32Array::from(vec![
            Date32Type::parse("2020-01-01"),
            Date32Type::parse("2020-10-30"),
        ])),
        // maxes are [2020-10-29, 2029-11-12]
        expected_max: Arc::new(Date32Array::from(vec![
            Date32Type::parse("2020-10-29"),
            Date32Type::parse("2029-11-12"),
        ])),
        // nulls are [2, 2]
        expected_null_counts: UInt64Array::from(vec![2, 2]),
        // row counts are [13, 7]
        expected_row_counts: Some(UInt64Array::from(vec![13, 7])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "date32",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_time32_second_diff_rg_sizes() {
    let reader = TestReader {
        scenario: Scenario::Time32Second,
        row_per_group: 4,
    }
    .build()
    .await;

    // Test for Time32Second column
    Test {
        reader: &reader,
        // Assuming specific minimum and maximum values for demonstration
        expected_min: Arc::new(Time32SecondArray::from(vec![18506, 18510, 18514, 18518])),
        expected_max: Arc::new(Time32SecondArray::from(vec![18509, 18513, 18517, 18521])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]), // Assuming 1 null per row group for simplicity
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "second",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_time32_millisecond_diff_rg_sizes() {
    let reader = TestReader {
        scenario: Scenario::Time32Millisecond,
        row_per_group: 4,
    }
    .build()
    .await;

    // Test for Time32Millisecond column
    Test {
        reader: &reader,
        // Assuming specific minimum and maximum values for demonstration
        expected_min: Arc::new(Time32MillisecondArray::from(vec![
            3600000, 3600004, 3600008, 3600012,
        ])),
        expected_max: Arc::new(Time32MillisecondArray::from(vec![
            3600003, 3600007, 3600011, 3600015,
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]), // Assuming 1 null per row group for simplicity
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "millisecond",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_time64_microsecond_diff_rg_sizes() {
    let reader = TestReader {
        scenario: Scenario::Time64Microsecond,
        row_per_group: 4,
    }
    .build()
    .await;

    // Test for Time64MicroSecond column
    Test {
        reader: &reader,
        // Assuming specific minimum and maximum values for demonstration
        expected_min: Arc::new(Time64MicrosecondArray::from(vec![
            1234567890123,
            1234567890127,
            1234567890131,
            1234567890135,
        ])),
        expected_max: Arc::new(Time64MicrosecondArray::from(vec![
            1234567890126,
            1234567890130,
            1234567890134,
            1234567890138,
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]), // Assuming 1 null per row group for simplicity
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "microsecond",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_time64_nanosecond_diff_rg_sizes() {
    let reader = TestReader {
        scenario: Scenario::Time64Nanosecond,
        row_per_group: 4,
    }
    .build()
    .await;

    // Test for Time32Second column
    Test {
        reader: &reader,
        // Assuming specific minimum and maximum values for demonstration
        expected_min: Arc::new(Time64NanosecondArray::from(vec![
            987654321012345,
            987654321012349,
            987654321012353,
            987654321012357,
        ])),
        expected_max: Arc::new(Time64NanosecondArray::from(vec![
            987654321012348,
            987654321012352,
            987654321012356,
            987654321012360,
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]), // Assuming 1 null per row group for simplicity
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),
        column_name: "nanosecond",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_dates_64_diff_rg_sizes() {
    // The file is created by 4 record batches (each has a null row), each has 5 rows but then will be split into 2 row groups with size 13, 7
    let reader = TestReader {
        scenario: Scenario::Dates,
        row_per_group: 13,
    }
    .build()
    .await;
    Test {
        reader: &reader,
        expected_min: Arc::new(Date64Array::from(vec![
            Date64Type::parse("2020-01-01"),
            Date64Type::parse("2020-10-30"),
        ])),
        expected_max: Arc::new(Date64Array::from(vec![
            Date64Type::parse("2020-10-29"),
            Date64Type::parse("2029-11-12"),
        ])),
        expected_null_counts: UInt64Array::from(vec![2, 2]),
        expected_row_counts: Some(UInt64Array::from(vec![13, 7])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "date64",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_uint() {
    // This creates a parquet files of 4 columns named "u8", "u16", "u32", "u64"
    // "u8" --> UInt8Array
    // "u16" --> UInt16Array
    // "u32" --> UInt32Array
    // "u64" --> UInt64Array

    // The file is created by 4 record batches (each has a null row), each has 5 rows but then will be split into 5 row groups with size 4
    let reader = TestReader {
        scenario: Scenario::UInt,
        row_per_group: 4,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt8Array::from(vec![0, 1, 4, 7, 251])),
        expected_max: Arc::new(UInt8Array::from(vec![3, 4, 6, 250, 254])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        column_name: "u8",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt16Array::from(vec![0, 1, 4, 7, 251])),
        expected_max: Arc::new(UInt16Array::from(vec![3, 4, 6, 250, 254])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        column_name: "u16",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt32Array::from(vec![0, 1, 4, 7, 251])),
        expected_max: Arc::new(UInt32Array::from(vec![3, 4, 6, 250, 254])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        column_name: "u32",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt64Array::from(vec![0, 1, 4, 7, 251])),
        expected_max: Arc::new(UInt64Array::from(vec![3, 4, 6, 250, 254])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![4, 4, 4, 4, 4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true, true]),
        column_name: "u64",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_int32_range() {
    // This creates a parquet file of 1 column "i"
    // file has 2 record batches, each has 2 rows. They will be saved into one row group
    let reader = TestReader {
        scenario: Scenario::Int32Range,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(Int32Array::from(vec![0])),
        expected_max: Arc::new(Int32Array::from(vec![300000])),
        expected_null_counts: UInt64Array::from(vec![0]),
        expected_row_counts: Some(UInt64Array::from(vec![4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true]),
        expected_min_value_exact: BooleanArray::from(vec![true]),
        column_name: "i",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_uint32_range() {
    // This creates a parquet file of 1 column "u"
    // file has 2 record batches, each has 2 rows. They will be saved into one row group
    let reader = TestReader {
        scenario: Scenario::UInt32Range,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt32Array::from(vec![0])),
        expected_max: Arc::new(UInt32Array::from(vec![300000])),
        expected_null_counts: UInt64Array::from(vec![0]),
        expected_row_counts: Some(UInt64Array::from(vec![4])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true]),
        expected_min_value_exact: BooleanArray::from(vec![true]),
        column_name: "u",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_numeric_limits_unsigned() {
    // file has 7 rows, 2 row groups: one with 5 rows, one with 2 rows.
    let reader = TestReader {
        scenario: Scenario::NumericLimits,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt8Array::from(vec![u8::MIN, 100])),
        expected_max: Arc::new(UInt8Array::from(vec![100, u8::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "u8",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt16Array::from(vec![u16::MIN, 100])),
        expected_max: Arc::new(UInt16Array::from(vec![100, u16::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "u16",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt32Array::from(vec![u32::MIN, 100])),
        expected_max: Arc::new(UInt32Array::from(vec![100, u32::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "u32",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(UInt64Array::from(vec![u64::MIN, 100])),
        expected_max: Arc::new(UInt64Array::from(vec![100, u64::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "u64",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_numeric_limits_signed() {
    // file has 7 rows, 2 row groups: one with 5 rows, one with 2 rows.
    let reader = TestReader {
        scenario: Scenario::NumericLimits,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(Int8Array::from(vec![i8::MIN, -100])),
        expected_max: Arc::new(Int8Array::from(vec![100, i8::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "i8",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Int16Array::from(vec![i16::MIN, -100])),
        expected_max: Arc::new(Int16Array::from(vec![100, i16::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "i16",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Int32Array::from(vec![i32::MIN, -100])),
        expected_max: Arc::new(Int32Array::from(vec![100, i32::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "i32",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Int64Array::from(vec![i64::MIN, -100])),
        expected_max: Arc::new(Int64Array::from(vec![100, i64::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "i64",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_numeric_limits_float() {
    // file has 7 rows, 2 row groups: one with 5 rows, one with 2 rows.
    let reader = TestReader {
        scenario: Scenario::NumericLimits,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(Float32Array::from(vec![f32::MIN, -100.0])),
        expected_max: Arc::new(Float32Array::from(vec![100.0, f32::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "f32",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Float64Array::from(vec![f64::MIN, -100.0])),
        expected_max: Arc::new(Float64Array::from(vec![100.0, f64::MAX])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "f64",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Float32Array::from(vec![-1.0, -100.0])),
        expected_max: Arc::new(Float32Array::from(vec![100.0, -100.0])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "f32_nan",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Float64Array::from(vec![-1.0, -100.0])),
        expected_max: Arc::new(Float64Array::from(vec![100.0, -100.0])),
        expected_null_counts: UInt64Array::from(vec![0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "f64_nan",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_float64() {
    // This creates a parquet file of 1 column "f"
    // file has 4 record batches, each has 5 rows. They will be saved into 4 row groups
    let reader = TestReader {
        scenario: Scenario::Float64,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(Float64Array::from(vec![-5.0, -4.0, -0.0, 5.0])),
        expected_max: Arc::new(Float64Array::from(vec![-1.0, 0.0, 4.0, 9.0])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),

        column_name: "f",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_float16() {
    // This creates a parquet file of 1 column "f"
    // file has 4 record batches, each has 5 rows. They will be saved into 4 row groups
    let reader = TestReader {
        scenario: Scenario::Float16,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(Float16Array::from(
            vec![-5.0, -4.0, -0.0, 5.0]
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )),
        expected_max: Arc::new(Float16Array::from(
            vec![-1.0, 0.0, 4.0, 9.0]
                .into_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true, true]),

        column_name: "f",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_decimal() {
    // This creates a parquet file of 1 column "decimal_col" with decimal data type and precicion 9, scale 2
    // file has 3 record batches, each has 5 rows. They will be saved into 3 row groups
    let reader = TestReader {
        scenario: Scenario::Decimal,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(
            Decimal128Array::from(vec![100, -500, 2000])
                .with_precision_and_scale(9, 2)
                .unwrap(),
        ),
        expected_max: Arc::new(
            Decimal128Array::from(vec![600, 600, 6000])
                .with_precision_and_scale(9, 2)
                .unwrap(),
        ),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "decimal_col",
        check: Check::Both,
    }
    .run();
}
#[tokio::test]
async fn test_decimal_256() {
    // This creates a parquet file of 1 column "decimal256_col" with decimal data type and precicion 9, scale 2
    // file has 3 record batches, each has 5 rows. They will be saved into 3 row groups
    let reader = TestReader {
        scenario: Scenario::Decimal256,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(
            Decimal256Array::from(vec![i256::from(100), i256::from(-500), i256::from(2000)])
                .with_precision_and_scale(9, 2)
                .unwrap(),
        ),
        expected_max: Arc::new(
            Decimal256Array::from(vec![i256::from(600), i256::from(600), i256::from(6000)])
                .with_precision_and_scale(9, 2)
                .unwrap(),
        ),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "decimal256_col",
        check: Check::Both,
    }
    .run();
}
#[tokio::test]
async fn test_dictionary() {
    let reader = TestReader {
        scenario: Scenario::Dictionary,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec!["abc", "aaa"])),
        expected_max: Arc::new(StringArray::from(vec!["def", "fffff"])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "string_dict_i8",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec!["abc", "aaa"])),
        expected_max: Arc::new(StringArray::from(vec!["def", "fffff"])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "string_dict_i32",
        check: Check::Both,
    }
    .run();

    Test {
        reader: &reader,
        expected_min: Arc::new(Int64Array::from(vec![-100, 0])),
        expected_max: Arc::new(Int64Array::from(vec![0, 100])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 2])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "int_dict_i8",
        check: Check::Both,
    }
    .run();
}

#[tokio::test]
async fn test_byte() {
    // This creates a parquet file of 5 columns
    // "name"
    // "service_string"
    // "service_binary"
    // "service_fixedsize"
    // "service_large_binary"

    // file has 3 record batches, each has 5 rows. They will be saved into 3 row groups
    let reader = TestReader {
        scenario: Scenario::ByteArray,
        row_per_group: 5,
    }
    .build()
    .await;

    // column "name"
    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec![
            "all frontends",
            "mixed",
            "all backends",
        ])),
        expected_max: Arc::new(StringArray::from(vec![
            "all frontends",
            "mixed",
            "all backends",
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "name",
        check: Check::Both,
    }
    .run();

    // column "service_string"
    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec![
            "frontend five",
            "backend one",
            "backend eight",
        ])),
        expected_max: Arc::new(StringArray::from(vec![
            "frontend two",
            "frontend six",
            "backend six",
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "service_string",
        check: Check::Both,
    }
    .run();

    // column "service_binary"

    let expected_service_binary_min_values: Vec<&[u8]> =
        vec![b"frontend five", b"backend one", b"backend eight"];

    let expected_service_binary_max_values: Vec<&[u8]> =
        vec![b"frontend two", b"frontend six", b"backend six"];

    Test {
        reader: &reader,
        expected_min: Arc::new(BinaryArray::from(expected_service_binary_min_values)),
        expected_max: Arc::new(BinaryArray::from(expected_service_binary_max_values)),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "service_binary",
        check: Check::Both,
    }
    .run();

    // column "service_fixedsize"
    // b"fe1", b"be1", b"be4"
    let min_input = vec![vec![102, 101, 49], vec![98, 101, 49], vec![98, 101, 52]];
    // b"fe5", b"fe6", b"be8"
    let max_input = vec![vec![102, 101, 55], vec![102, 101, 54], vec![98, 101, 56]];

    Test {
        reader: &reader,
        expected_min: Arc::new(FixedSizeBinaryArray::try_from_iter(min_input.into_iter()).unwrap()),
        expected_max: Arc::new(FixedSizeBinaryArray::try_from_iter(max_input.into_iter()).unwrap()),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "service_fixedsize",
        check: Check::Both,
    }
    .run();

    let expected_service_large_binary_min_values: Vec<&[u8]> =
        vec![b"frontend five", b"backend one", b"backend eight"];

    let expected_service_large_binary_max_values: Vec<&[u8]> =
        vec![b"frontend two", b"frontend six", b"backend six"];

    Test {
        reader: &reader,
        expected_min: Arc::new(LargeBinaryArray::from(
            expected_service_large_binary_min_values,
        )),
        expected_max: Arc::new(LargeBinaryArray::from(
            expected_service_large_binary_max_values,
        )),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "service_large_binary",
        check: Check::Both,
    }
    .run();
}

// PeriodsInColumnNames
#[tokio::test]
async fn test_period_in_column_names() {
    // This creates a parquet file of 2 columns "name" and "service.name"
    // file has 3 record batches, each has 5 rows. They will be saved into 3 row groups
    let reader = TestReader {
        scenario: Scenario::PeriodsInColumnNames,
        row_per_group: 5,
    }
    .build()
    .await;

    // column "name"
    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec![
            "HTTP GET / DISPATCH",
            "HTTP PUT / DISPATCH",
            "HTTP GET / DISPATCH",
        ])),
        expected_max: Arc::new(StringArray::from(vec![
            "HTTP GET / DISPATCH",
            "HTTP PUT / DISPATCH",
            "HTTP GET / DISPATCH",
        ])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "name",
        check: Check::Both,
    }
    .run();

    // column "service.name"
    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec!["frontend", "backend", "backend"])),
        expected_max: Arc::new(StringArray::from(vec!["frontend", "frontend", "backend"])),
        expected_null_counts: UInt64Array::from(vec![0, 0, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "service.name",
        check: Check::Both,
    }
    .run();
}

// Boolean
#[tokio::test]
async fn test_boolean() {
    // This creates a parquet files of 1 column named "bool"
    // The file is created by 2 record batches each has 5 rows --> 2 row groups
    let reader = TestReader {
        scenario: Scenario::Boolean,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        expected_min: Arc::new(BooleanArray::from(vec![false, false])),
        expected_max: Arc::new(BooleanArray::from(vec![true, false])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "bool",
        check: Check::Both,
    }
    .run();
}

// struct array
// BUG
// https://github.com/apache/datafusion/issues/10609
// Note that: since I have not worked on struct array before, there may be a bug in the test code rather than the real bug in the code
#[ignore]
#[tokio::test]
async fn test_struct() {
    // This creates a parquet files of 1 column named "struct"
    // The file is created by 1 record batch with 3 rows in the struct array
    let reader = TestReader {
        scenario: Scenario::StructArray,
        row_per_group: 5,
    }
    .build()
    .await;
    Test {
        reader: &reader,
        expected_min: Arc::new(struct_array(vec![(Some(1), Some(6.0), Some(12.0))])),
        expected_max: Arc::new(struct_array(vec![(Some(2), Some(8.5), Some(14.0))])),
        expected_null_counts: UInt64Array::from(vec![0]),
        expected_row_counts: Some(UInt64Array::from(vec![3])),
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "struct",
        check: Check::RowGroup,
    }
    .run();
}

// UTF8
#[tokio::test]
async fn test_utf8() {
    let reader = TestReader {
        scenario: Scenario::UTF8,
        row_per_group: 5,
    }
    .build()
    .await;

    // test for utf8
    Test {
        reader: &reader,
        expected_min: Arc::new(StringArray::from(vec!["a", "e"])),
        expected_max: Arc::new(StringArray::from(vec!["d", "i"])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "utf8",
        check: Check::Both,
    }
    .run();

    // test for large_utf8
    Test {
        reader: &reader,
        expected_min: Arc::new(LargeStringArray::from(vec!["a", "e"])),
        expected_max: Arc::new(LargeStringArray::from(vec!["d", "i"])),
        expected_null_counts: UInt64Array::from(vec![1, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "large_utf8",
        check: Check::Both,
    }
    .run();
}

// UTF8View
#[tokio::test]
async fn test_utf8_view() {
    let reader = TestReader {
        scenario: Scenario::UTF8View,
        row_per_group: 5,
    }
    .build()
    .await;

    // test for utf8_view
    Test {
        reader: &reader,
        expected_min: Arc::new(StringViewArray::from(vec!["a", "a", "e_longerthan12"])),
        expected_max: Arc::new(StringViewArray::from(vec![
            "d",
            "e_longerthan12",
            "i_longerthan12",
        ])),
        expected_null_counts: UInt64Array::from(vec![1, 3, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "utf8_view",
        check: Check::Both,
    }
    .run()
}

// BinaryView
#[tokio::test]
async fn test_binary_view() {
    let reader = TestReader {
        scenario: Scenario::BinaryView,
        row_per_group: 5,
    }
    .build()
    .await;

    let expected_min: Vec<Option<&[u8]>> = vec![Some(b"a"), Some(b"a"), Some(b"e_longerthan12")];
    let expected_max: Vec<Option<&[u8]>> =
        vec![Some(b"d"), Some(b"e_longerthan12"), Some(b"i_longerthan12")];

    // test for utf8_view
    Test {
        reader: &reader,
        expected_min: Arc::new(BinaryViewArray::from(expected_min)),
        expected_max: Arc::new(BinaryViewArray::from(expected_max)),
        expected_null_counts: UInt64Array::from(vec![1, 3, 0]),
        expected_row_counts: Some(UInt64Array::from(vec![5, 5, 5])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true, true]),
        column_name: "binary_view",
        check: Check::Both,
    }
    .run()
}

////// Files with missing statistics ///////

#[tokio::test]
async fn test_missing_statistics() {
    let reader = Int64Case {
        null_values: 0,
        no_null_values_start: 4,
        no_null_values_end: 7,
        row_per_group: 5,
        enable_stats: Some(EnabledStatistics::None),
        ..Default::default()
    }
    .build();

    Test {
        reader: &reader,
        expected_min: Arc::new(Int64Array::from(vec![None])),
        expected_max: Arc::new(Int64Array::from(vec![None])),
        expected_null_counts: UInt64Array::from(vec![None]),
        expected_row_counts: Some(UInt64Array::from(vec![3])), // still has row count statistics
        expected_max_value_exact: BooleanArray::from(vec![None]),
        expected_min_value_exact: BooleanArray::from(vec![None]),
        column_name: "i64",
        check: Check::Both,
    }
    .run();
}

#[test]
fn missing_null_counts_as_zero() {
    let min = None;
    let max = None;
    let distinct_count = None;
    let null_count = None; // NB: no null count
    let is_min_max_deprecated = false;
    let stats = Statistics::Boolean(ValueStatistics::new(
        min,
        max,
        distinct_count,
        null_count,
        is_min_max_deprecated,
    ));
    let (arrow_schema, parquet_schema) = bool_arrow_and_parquet_schema();

    let column_chunk = ColumnChunkMetaData::builder(parquet_schema.column(0))
        .set_statistics(stats)
        .build()
        .unwrap();
    let metadata = RowGroupMetaData::builder(parquet_schema.clone())
        .set_column_metadata(vec![column_chunk])
        .build()
        .unwrap();

    let converter = StatisticsConverter::try_new("b", &arrow_schema, &parquet_schema).unwrap();

    // by default null count should be 0
    assert_eq!(
        converter.row_group_null_counts([&metadata]).unwrap(),
        UInt64Array::from_iter(vec![Some(0)])
    );

    // if we disable missing null counts as zero flag null count will be None
    let converter = converter.with_missing_null_counts_as_zero(false);
    assert_eq!(
        converter.row_group_null_counts([&metadata]).unwrap(),
        UInt64Array::from_iter(vec![None])
    );
}

/// return an Arrow schema and corresponding Parquet SchemaDescriptor for
/// a schema with a single boolean column "b"
fn bool_arrow_and_parquet_schema() -> (SchemaRef, SchemaDescPtr) {
    let arrow_schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, true)]));
    use parquet::schema::types::Type as ParquetType;
    let parquet_schema = ParquetType::group_type_builder("schema")
        .with_fields(vec![Arc::new(
            ParquetType::primitive_type_builder("a", parquet::basic::Type::INT32)
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap();

    let parquet_schema = Arc::new(SchemaDescriptor::new(Arc::new(parquet_schema)));
    (arrow_schema, parquet_schema)
}

/////// NEGATIVE TESTS ///////
// column not found
#[tokio::test]
async fn test_column_not_found() {
    let reader = TestReader {
        scenario: Scenario::Dates,
        row_per_group: 5,
    }
    .build()
    .await;
    Test {
        reader: &reader,
        expected_min: Arc::new(Int64Array::from(vec![18262, 18565])),
        expected_max: Arc::new(Int64Array::from(vec![18564, 21865])),
        expected_null_counts: UInt64Array::from(vec![2, 2]),
        expected_row_counts: Some(UInt64Array::from(vec![13, 7])),
        // stats are exact
        expected_max_value_exact: BooleanArray::from(vec![true, true]),
        expected_min_value_exact: BooleanArray::from(vec![true, true]),
        column_name: "not_a_column",
        check: Check::Both,
    }
    .run_col_not_found();
}

#[tokio::test]
async fn test_column_non_existent() {
    // Create a schema with an additional column
    // that will not have a matching parquet index
    let schema = Arc::new(Schema::new(vec![
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("i32", DataType::Int32, true),
        Field::new("i64", DataType::Int64, true),
        Field::new("i_do_not_exist", DataType::Int64, true),
    ]));

    let reader = TestReader {
        scenario: Scenario::Int,
        row_per_group: 5,
    }
    .build()
    .await;

    Test {
        reader: &reader,
        // mins are [-5, -4, 0, 5]
        expected_min: Arc::new(Int64Array::from(vec![None, None, None, None])),
        // maxes are [-1, 0, 4, 9]
        expected_max: Arc::new(Int64Array::from(vec![None, None, None, None])),
        // nulls are [0, 0, 0, 0]
        expected_null_counts: UInt64Array::from(vec![None, None, None, None]),
        // row counts are [5, 5, 5, 5]
        expected_row_counts: None,
        expected_max_value_exact: BooleanArray::from(vec![None, None, None, None]),
        expected_min_value_exact: BooleanArray::from(vec![None, None, None, None]),
        column_name: "i_do_not_exist",
        check: Check::Both,
    }
    .run_with_schema(&schema);
}

/// The following tests were initially in `arrow-rs/parquet/src/arrow/arrow_reader/statistics.rs`.
/// Part of them was moved here to avoid confusion and duplication,
/// including edge conditions and data file validations for only `row_group` statistics.
#[cfg(test)]
mod test {
    use super::*;
    use arrow::util::test_util::parquet_test_data;
    use arrow_array::{
        new_empty_array, ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, RecordBatch, StringArray,
        TimestampNanosecondArray,
    };
    use arrow_schema::{DataType, SchemaRef, TimeUnit};
    use bytes::Bytes;
    use parquet::arrow::parquet_column;
    use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData};
    use std::path::PathBuf;
    use std::sync::Arc;
    // TODO error cases (with parquet statistics that are mismatched in expected type)

    #[test]
    fn roundtrip_empty() {
        let all_types = vec![
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Date32,
            DataType::Date64,
            // Skip types that don't support statistics
            // DataType::Time32(Second),
            // DataType::Time64(Second),
            // DataType::Duration(Second),
            // DataType::Interval(IntervalUnit),
            DataType::Binary,
            DataType::FixedSizeBinary(0),
            DataType::LargeBinary,
            DataType::BinaryView,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            // DataType::List(FieldRef),
            // DataType::ListView(FieldRef),
            // DataType::FixedSizeList(FieldRef, i32),
            // DataType::LargeList(FieldRef),
            // DataType::LargeListView(FieldRef),
            // DataType::Struct(Fields),
            // DataType::Union(UnionFields, UnionMode),
            // DataType::Dictionary(Box<DataType>, Box<DataType>),
            // DataType::Decimal128(u8, i8),
            // DataType::Decimal256(u8, i8),
            // DataType::Map(FieldRef, bool),
            // DataType::RunEndEncoded(FieldRef, FieldRef),
        ];
        for data_type in all_types {
            let empty_array = new_empty_array(&data_type);
            Test {
                input: empty_array.clone(),
                expected_min: empty_array.clone(),
                expected_max: empty_array,
            }
            .run();
        }
    }

    #[test]
    fn nan_in_stats() {
        // /parquet-testing/data/nan_in_stats.parquet
        // row_groups: 1
        // "x": Double({min: Some(1.0), max: Some(NaN), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})

        TestFile::new("nan_in_stats.parquet")
            .with_column(ExpectedColumn {
                name: "x",
                expected_min: Arc::new(Float64Array::from(vec![Some(1.0)])),
                expected_max: Arc::new(Float64Array::from(vec![Some(f64::NAN)])),
            })
            .run();
    }

    #[test]
    fn alltypes_plain() {
        // /parquet-testing/data/datapage_v1-snappy-compressed-checksum.parquet
        // row_groups: 1
        // (has no statistics)
        TestFile::new("alltypes_plain.parquet")
            // No column statistics should be read as NULL, but with the right type
            .with_column(ExpectedColumn {
                name: "id",
                expected_min: i32_array([None]),
                expected_max: i32_array([None]),
            })
            .with_column(ExpectedColumn {
                name: "bool_col",
                expected_min: bool_array([None]),
                expected_max: bool_array([None]),
            })
            .run();
    }

    #[test]
    fn alltypes_tiny_pages() {
        // /parquet-testing/data/alltypes_tiny_pages.parquet
        // row_groups: 1
        // "id": Int32({min: Some(0), max: Some(7299), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "bool_col": Boolean({min: Some(false), max: Some(true), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "tinyint_col": Int32({min: Some(0), max: Some(9), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "smallint_col": Int32({min: Some(0), max: Some(9), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "int_col": Int32({min: Some(0), max: Some(9), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "bigint_col": Int64({min: Some(0), max: Some(90), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "float_col": Float({min: Some(0.0), max: Some(9.9), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "double_col": Double({min: Some(0.0), max: Some(90.89999999999999), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "date_string_col": ByteArray({min: Some(ByteArray { data: "01/01/09" }), max: Some(ByteArray { data: "12/31/10" }), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "string_col": ByteArray({min: Some(ByteArray { data: "0" }), max: Some(ByteArray { data: "9" }), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "timestamp_col": Int96({min: None, max: None, distinct_count: None, null_count: 0, min_max_deprecated: true, min_max_backwards_compatible: true})
        // "year": Int32({min: Some(2009), max: Some(2010), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        // "month": Int32({min: Some(1), max: Some(12), distinct_count: None, null_count: 0, min_max_deprecated: false, min_max_backwards_compatible: false})
        TestFile::new("alltypes_tiny_pages.parquet")
            .with_column(ExpectedColumn {
                name: "id",
                expected_min: i32_array([Some(0)]),
                expected_max: i32_array([Some(7299)]),
            })
            .with_column(ExpectedColumn {
                name: "bool_col",
                expected_min: bool_array([Some(false)]),
                expected_max: bool_array([Some(true)]),
            })
            .with_column(ExpectedColumn {
                name: "tinyint_col",
                expected_min: i8_array([Some(0)]),
                expected_max: i8_array([Some(9)]),
            })
            .with_column(ExpectedColumn {
                name: "smallint_col",
                expected_min: i16_array([Some(0)]),
                expected_max: i16_array([Some(9)]),
            })
            .with_column(ExpectedColumn {
                name: "int_col",
                expected_min: i32_array([Some(0)]),
                expected_max: i32_array([Some(9)]),
            })
            .with_column(ExpectedColumn {
                name: "bigint_col",
                expected_min: i64_array([Some(0)]),
                expected_max: i64_array([Some(90)]),
            })
            .with_column(ExpectedColumn {
                name: "float_col",
                expected_min: f32_array([Some(0.0)]),
                expected_max: f32_array([Some(9.9)]),
            })
            .with_column(ExpectedColumn {
                name: "double_col",
                expected_min: f64_array([Some(0.0)]),
                expected_max: f64_array([Some(90.89999999999999)]),
            })
            .with_column(ExpectedColumn {
                name: "date_string_col",
                expected_min: utf8_array([Some("01/01/09")]),
                expected_max: utf8_array([Some("12/31/10")]),
            })
            .with_column(ExpectedColumn {
                name: "string_col",
                expected_min: utf8_array([Some("0")]),
                expected_max: utf8_array([Some("9")]),
            })
            // File has no min/max for timestamp_col
            .with_column(ExpectedColumn {
                name: "timestamp_col",
                expected_min: timestamp_nanoseconds_array([None], None),
                expected_max: timestamp_nanoseconds_array([None], None),
            })
            .with_column(ExpectedColumn {
                name: "year",
                expected_min: i32_array([Some(2009)]),
                expected_max: i32_array([Some(2010)]),
            })
            .with_column(ExpectedColumn {
                name: "month",
                expected_min: i32_array([Some(1)]),
                expected_max: i32_array([Some(12)]),
            })
            .run();
    }

    #[test]
    fn fixed_length_decimal_legacy() {
        // /parquet-testing/data/fixed_length_decimal_legacy.parquet
        // row_groups: 1
        // "value": FixedLenByteArray({min: Some(FixedLenByteArray(ByteArray { data: Some(ByteBufferPtr { data: b"\0\0\0\0\0\xc8" }) })), max: Some(FixedLenByteArray(ByteArray { data: "\0\0\0\0\t`" })), distinct_count: None, null_count: 0, min_max_deprecated: true, min_max_backwards_compatible: true})

        TestFile::new("fixed_length_decimal_legacy.parquet")
            .with_column(ExpectedColumn {
                name: "value",
                expected_min: Arc::new(
                    Decimal128Array::from(vec![Some(200)])
                        .with_precision_and_scale(13, 2)
                        .unwrap(),
                ),
                expected_max: Arc::new(
                    Decimal128Array::from(vec![Some(2400)])
                        .with_precision_and_scale(13, 2)
                        .unwrap(),
                ),
            })
            .run();
    }

    const ROWS_PER_ROW_GROUP: usize = 3;

    /// Writes the input batch into a parquet file, with every every three rows as
    /// their own row group, and compares the min/maxes to the expected values
    struct Test {
        input: ArrayRef,
        expected_min: ArrayRef,
        expected_max: ArrayRef,
    }

    impl Test {
        fn run(self) {
            let Self {
                input,
                expected_min,
                expected_max,
            } = self;

            let input_batch = RecordBatch::try_from_iter([("c1", input)]).unwrap();

            let schema = input_batch.schema();

            let metadata = parquet_metadata(schema.clone(), input_batch);
            let parquet_schema = metadata.file_metadata().schema_descr();

            let row_groups = metadata.row_groups();

            for field in schema.fields() {
                if field.data_type().is_nested() {
                    let lookup = parquet_column(parquet_schema, &schema, field.name());
                    assert_eq!(lookup, None);
                    continue;
                }

                let converter =
                    StatisticsConverter::try_new(field.name(), &schema, parquet_schema).unwrap();

                assert_eq!(converter.arrow_field(), field.as_ref());

                let mins = converter.row_group_mins(row_groups.iter()).unwrap();
                assert_eq!(
                    &mins,
                    &expected_min,
                    "Min. Statistics\n\n{}\n\n",
                    DisplayStats(row_groups)
                );

                let maxes = converter.row_group_maxes(row_groups.iter()).unwrap();
                assert_eq!(
                    &maxes,
                    &expected_max,
                    "Max. Statistics\n\n{}\n\n",
                    DisplayStats(row_groups)
                );
            }
        }
    }

    /// Write the specified batches out as parquet and return the metadata
    fn parquet_metadata(schema: SchemaRef, batch: RecordBatch) -> Arc<ParquetMetaData> {
        let props = WriterProperties::builder()
            .set_statistics_enabled(EnabledStatistics::Chunk)
            .set_max_row_group_size(ROWS_PER_ROW_GROUP)
            .build();

        let mut buffer = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::try_new(Bytes::from(buffer)).unwrap();
        reader.metadata().clone()
    }

    /// Formats the statistics nicely for display
    struct DisplayStats<'a>(&'a [RowGroupMetaData]);
    impl std::fmt::Display for DisplayStats<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let row_groups = self.0;
            writeln!(f, "  row_groups: {}", row_groups.len())?;
            for rg in row_groups {
                for col in rg.columns() {
                    if let Some(statistics) = col.statistics() {
                        writeln!(f, "   {}: {:?}", col.column_path(), statistics)?;
                    }
                }
            }
            Ok(())
        }
    }

    struct ExpectedColumn {
        name: &'static str,
        expected_min: ArrayRef,
        expected_max: ArrayRef,
    }

    /// Reads statistics out of the specified, and compares them to the expected values
    struct TestFile {
        file_name: &'static str,
        expected_columns: Vec<ExpectedColumn>,
    }

    impl TestFile {
        fn new(file_name: &'static str) -> Self {
            Self {
                file_name,
                expected_columns: Vec::new(),
            }
        }

        fn with_column(mut self, column: ExpectedColumn) -> Self {
            self.expected_columns.push(column);
            self
        }

        /// Reads the specified parquet file and validates that the expected min/max
        /// values for the specified columns are as expected.
        fn run(self) {
            let path = PathBuf::from(parquet_test_data()).join(self.file_name);
            let file = File::open(path).unwrap();
            let reader = ArrowReaderBuilder::try_new(file).unwrap();
            let arrow_schema = reader.schema();
            let metadata = reader.metadata();
            let row_groups = metadata.row_groups();
            let parquet_schema = metadata.file_metadata().schema_descr();

            for expected_column in self.expected_columns {
                let ExpectedColumn {
                    name,
                    expected_min,
                    expected_max,
                } = expected_column;

                let converter =
                    StatisticsConverter::try_new(name, arrow_schema, parquet_schema).unwrap();

                // test accessors on the converter
                let parquet_column_index =
                    parquet_column(parquet_schema, arrow_schema, name).map(|(idx, _field)| idx);
                assert_eq!(converter.parquet_column_index(), parquet_column_index);
                assert_eq!(converter.arrow_field().name(), name);

                let actual_min = converter.row_group_mins(row_groups.iter()).unwrap();
                assert_eq!(&expected_min, &actual_min, "column {name}");

                let actual_max = converter.row_group_maxes(row_groups.iter()).unwrap();
                assert_eq!(&expected_max, &actual_max, "column {name}");
            }
        }
    }

    fn bool_array(input: impl IntoIterator<Item = Option<bool>>) -> ArrayRef {
        let array: BooleanArray = input.into_iter().collect();
        Arc::new(array)
    }

    fn i8_array(input: impl IntoIterator<Item = Option<i8>>) -> ArrayRef {
        let array: Int8Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn i16_array(input: impl IntoIterator<Item = Option<i16>>) -> ArrayRef {
        let array: Int16Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn i32_array(input: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
        let array: Int32Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn i64_array(input: impl IntoIterator<Item = Option<i64>>) -> ArrayRef {
        let array: Int64Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn f32_array(input: impl IntoIterator<Item = Option<f32>>) -> ArrayRef {
        let array: Float32Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn f64_array(input: impl IntoIterator<Item = Option<f64>>) -> ArrayRef {
        let array: Float64Array = input.into_iter().collect();
        Arc::new(array)
    }

    fn timestamp_nanoseconds_array(
        input: impl IntoIterator<Item = Option<i64>>,
        timzezone: Option<&str>,
    ) -> ArrayRef {
        let array: TimestampNanosecondArray = input.into_iter().collect();
        match timzezone {
            Some(tz) => Arc::new(array.with_timezone(tz)),
            None => Arc::new(array),
        }
    }

    fn utf8_array<'a>(input: impl IntoIterator<Item = Option<&'a str>>) -> ArrayRef {
        let array: StringArray = input
            .into_iter()
            .map(|s| s.map(|s| s.to_string()))
            .collect();
        Arc::new(array)
    }
}
