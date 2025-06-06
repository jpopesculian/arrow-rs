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

//! Conversion methods for dates and times.

use crate::timezone::Tz;
use crate::ArrowPrimitiveType;
use arrow_schema::{DataType, TimeUnit};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc};

/// Number of seconds in a day
pub const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
pub const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
pub const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
pub const NANOSECONDS: i64 = 1_000_000_000;

/// Number of milliseconds in a day
pub const MILLISECONDS_IN_DAY: i64 = SECONDS_IN_DAY * MILLISECONDS;
/// Number of microseconds in a day
pub const MICROSECONDS_IN_DAY: i64 = SECONDS_IN_DAY * MICROSECONDS;
/// Number of nanoseconds in a day
pub const NANOSECONDS_IN_DAY: i64 = SECONDS_IN_DAY * NANOSECONDS;

/// Constant from chrono crate
///
/// Number of days between Januari 1, 1970 and December 31, 1 BCE which we define to be day 0.
/// 4 full leap year cycles until December 31, 1600     4 * 146097 = 584388
/// 1 day until January 1, 1601                                           1
/// 369 years until Januari 1, 1970                      369 * 365 = 134685
/// of which floor(369 / 4) are leap years          floor(369 / 4) =     92
/// except for 1700, 1800 and 1900                                       -3 +
///                                                                  --------
///                                                                  719163
pub const UNIX_EPOCH_DAY: i64 = 719_163;

/// converts a `i32` representing a `date32` to [`NaiveDateTime`]
#[inline]
pub fn date32_to_datetime(v: i32) -> Option<NaiveDateTime> {
    Some(DateTime::from_timestamp(v as i64 * SECONDS_IN_DAY, 0)?.naive_utc())
}

/// converts a `i64` representing a `date64` to [`NaiveDateTime`]
#[inline]
pub fn date64_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, milli_sec) = split_second(v, MILLISECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from milliseconds
        sec,
        // discard extracted seconds and convert milliseconds to nanoseconds
        milli_sec * MICROSECONDS as u32,
    )?;
    Some(datetime.naive_utc())
}

/// converts a `i32` representing a `time32(s)` to [`NaiveDateTime`]
#[inline]
pub fn time32s_to_time(v: i32) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(v as u32, 0)
}

/// converts a `i32` representing a `time32(ms)` to [`NaiveDateTime`]
#[inline]
pub fn time32ms_to_time(v: i32) -> Option<NaiveTime> {
    let v = v as i64;
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from milliseconds
        (v / MILLISECONDS) as u32,
        // discard extracted seconds and convert milliseconds to
        // nanoseconds
        (v % MILLISECONDS * MICROSECONDS) as u32,
    )
}

/// converts a `i64` representing a `time64(us)` to [`NaiveDateTime`]
#[inline]
pub fn time64us_to_time(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from microseconds
        (v / MICROSECONDS) as u32,
        // discard extracted seconds and convert microseconds to
        // nanoseconds
        (v % MICROSECONDS * MILLISECONDS) as u32,
    )
}

/// converts a `i64` representing a `time64(ns)` to [`NaiveDateTime`]
#[inline]
pub fn time64ns_to_time(v: i64) -> Option<NaiveTime> {
    NaiveTime::from_num_seconds_from_midnight_opt(
        // extract seconds from nanoseconds
        (v / NANOSECONDS) as u32,
        // discard extracted seconds
        (v % NANOSECONDS) as u32,
    )
}

/// converts [`NaiveTime`] to a `i32` representing a `time32(s)`
#[inline]
pub fn time_to_time32s(v: NaiveTime) -> i32 {
    v.num_seconds_from_midnight() as i32
}

/// converts [`NaiveTime`] to a `i32` representing a `time32(ms)`
#[inline]
pub fn time_to_time32ms(v: NaiveTime) -> i32 {
    (v.num_seconds_from_midnight() as i64 * MILLISECONDS
        + v.nanosecond() as i64 * MILLISECONDS / NANOSECONDS) as i32
}

/// converts [`NaiveTime`] to a `i64` representing a `time64(us)`
#[inline]
pub fn time_to_time64us(v: NaiveTime) -> i64 {
    v.num_seconds_from_midnight() as i64 * MICROSECONDS
        + v.nanosecond() as i64 * MICROSECONDS / NANOSECONDS
}

/// converts [`NaiveTime`] to a `i64` representing a `time64(ns)`
#[inline]
pub fn time_to_time64ns(v: NaiveTime) -> i64 {
    v.num_seconds_from_midnight() as i64 * NANOSECONDS + v.nanosecond() as i64
}

/// converts a `i64` representing a `timestamp(s)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_s_to_datetime(v: i64) -> Option<NaiveDateTime> {
    Some(DateTime::from_timestamp(v, 0)?.naive_utc())
}

/// Similar to timestamp_s_to_datetime but only compute `date`
#[inline]
pub fn timestamp_s_to_date(secs: i64) -> Option<NaiveDateTime> {
    let days = secs.div_euclid(86_400) + UNIX_EPOCH_DAY;
    if days < i32::MIN as i64 || days > i32::MAX as i64 {
        return None;
    }
    let date = NaiveDate::from_num_days_from_ce_opt(days as i32)?;
    Some(date.and_time(NaiveTime::default()).and_utc().naive_utc())
}

/// Similar to timestamp_s_to_datetime but only compute `time`
#[inline]
pub fn timestamp_s_to_time(secs: i64) -> Option<NaiveDateTime> {
    let secs = secs.rem_euclid(86_400);
    let time = NaiveTime::from_num_seconds_from_midnight_opt(secs as u32, 0)?;
    Some(
        DateTime::<Utc>::from_naive_utc_and_offset(
            NaiveDateTime::new(NaiveDate::default(), time),
            Utc,
        )
        .naive_utc(),
    )
}

/// converts a `i64` representing a `timestamp(ms)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ms_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, milli_sec) = split_second(v, MILLISECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from milliseconds
        sec,
        // discard extracted seconds and convert milliseconds to nanoseconds
        milli_sec * MICROSECONDS as u32,
    )?;
    Some(datetime.naive_utc())
}

/// converts a `i64` representing a `timestamp(us)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_us_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, micro_sec) = split_second(v, MICROSECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from microseconds
        sec,
        // discard extracted seconds and convert microseconds to nanoseconds
        micro_sec * MILLISECONDS as u32,
    )?;
    Some(datetime.naive_utc())
}

/// converts a `i64` representing a `timestamp(ns)` to [`NaiveDateTime`]
#[inline]
pub fn timestamp_ns_to_datetime(v: i64) -> Option<NaiveDateTime> {
    let (sec, nano_sec) = split_second(v, NANOSECONDS);

    let datetime = DateTime::from_timestamp(
        // extract seconds from nanoseconds
        sec, // discard extracted seconds
        nano_sec,
    )?;
    Some(datetime.naive_utc())
}

#[inline]
pub(crate) fn split_second(v: i64, base: i64) -> (i64, u32) {
    (v.div_euclid(base), v.rem_euclid(base) as u32)
}

/// converts a `i64` representing a `duration(s)` to [`Duration`]
#[inline]
#[deprecated(since = "55.2.0", note = "Use `try_duration_s_to_duration` instead")]
pub fn duration_s_to_duration(v: i64) -> Duration {
    Duration::try_seconds(v).unwrap()
}

/// converts a `i64` representing a `duration(s)` to [`Option<Duration>`]
#[inline]
pub fn try_duration_s_to_duration(v: i64) -> Option<Duration> {
    Duration::try_seconds(v)
}

/// converts a `i64` representing a `duration(ms)` to [`Duration`]
#[inline]
#[deprecated(since = "55.2.0", note = "Use `try_duration_ms_to_duration` instead")]
pub fn duration_ms_to_duration(v: i64) -> Duration {
    Duration::try_seconds(v).unwrap()
}

/// converts a `i64` representing a `duration(ms)` to [`Option<Duration>`]
#[inline]
pub fn try_duration_ms_to_duration(v: i64) -> Option<Duration> {
    Duration::try_milliseconds(v)
}

/// converts a `i64` representing a `duration(us)` to [`Duration`]
#[inline]
pub fn duration_us_to_duration(v: i64) -> Duration {
    Duration::microseconds(v)
}

/// converts a `i64` representing a `duration(ns)` to [`Duration`]
#[inline]
pub fn duration_ns_to_duration(v: i64) -> Duration {
    Duration::nanoseconds(v)
}

/// Converts an [`ArrowPrimitiveType`] to [`NaiveDateTime`]
pub fn as_datetime<T: ArrowPrimitiveType>(v: i64) -> Option<NaiveDateTime> {
    match T::DATA_TYPE {
        DataType::Date32 => date32_to_datetime(v as i32),
        DataType::Date64 => date64_to_datetime(v),
        DataType::Time32(_) | DataType::Time64(_) => None,
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => timestamp_s_to_datetime(v),
            TimeUnit::Millisecond => timestamp_ms_to_datetime(v),
            TimeUnit::Microsecond => timestamp_us_to_datetime(v),
            TimeUnit::Nanosecond => timestamp_ns_to_datetime(v),
        },
        // interval is not yet fully documented [ARROW-3097]
        DataType::Interval(_) => None,
        _ => None,
    }
}

/// Converts an [`ArrowPrimitiveType`] to [`DateTime<Tz>`]
pub fn as_datetime_with_timezone<T: ArrowPrimitiveType>(v: i64, tz: Tz) -> Option<DateTime<Tz>> {
    let naive = as_datetime::<T>(v)?;
    Some(Utc.from_utc_datetime(&naive).with_timezone(&tz))
}

/// Converts an [`ArrowPrimitiveType`] to [`NaiveDate`]
pub fn as_date<T: ArrowPrimitiveType>(v: i64) -> Option<NaiveDate> {
    as_datetime::<T>(v).map(|datetime| datetime.date())
}

/// Converts an [`ArrowPrimitiveType`] to [`NaiveTime`]
pub fn as_time<T: ArrowPrimitiveType>(v: i64) -> Option<NaiveTime> {
    match T::DATA_TYPE {
        DataType::Time32(unit) => {
            // safe to immediately cast to u32 as `self.value(i)` is positive i32
            let v = v as u32;
            match unit {
                TimeUnit::Second => time32s_to_time(v as i32),
                TimeUnit::Millisecond => time32ms_to_time(v as i32),
                _ => None,
            }
        }
        DataType::Time64(unit) => match unit {
            TimeUnit::Microsecond => time64us_to_time(v),
            TimeUnit::Nanosecond => time64ns_to_time(v),
            _ => None,
        },
        DataType::Timestamp(_, _) => as_datetime::<T>(v).map(|datetime| datetime.time()),
        DataType::Date32 | DataType::Date64 => NaiveTime::from_hms_opt(0, 0, 0),
        DataType::Interval(_) => None,
        _ => None,
    }
}

/// Converts an [`ArrowPrimitiveType`] to [`Duration`]
pub fn as_duration<T: ArrowPrimitiveType>(v: i64) -> Option<Duration> {
    match T::DATA_TYPE {
        DataType::Duration(unit) => match unit {
            TimeUnit::Second => try_duration_s_to_duration(v),
            TimeUnit::Millisecond => try_duration_ms_to_duration(v),
            TimeUnit::Microsecond => Some(duration_us_to_duration(v)),
            TimeUnit::Nanosecond => Some(duration_ns_to_duration(v)),
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::temporal_conversions::{
        date64_to_datetime, split_second, timestamp_ms_to_datetime, timestamp_ns_to_datetime,
        timestamp_s_to_date, timestamp_s_to_datetime, timestamp_s_to_time,
        timestamp_us_to_datetime, NANOSECONDS,
    };
    use chrono::DateTime;

    #[test]
    fn test_timestamp_func() {
        let timestamp = 1234;
        let datetime = timestamp_s_to_datetime(timestamp).unwrap();
        let expected_date = datetime.date();
        let expected_time = datetime.time();

        assert_eq!(
            timestamp_s_to_date(timestamp).unwrap().date(),
            expected_date
        );
        assert_eq!(
            timestamp_s_to_time(timestamp).unwrap().time(),
            expected_time
        );
    }

    #[test]
    fn negative_input_timestamp_ns_to_datetime() {
        assert_eq!(
            timestamp_ns_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_999_999).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_ns_to_datetime(-1_000_000_001),
            DateTime::from_timestamp(-2, 999_999_999).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn negative_input_timestamp_us_to_datetime() {
        assert_eq!(
            timestamp_us_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_999_000).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_us_to_datetime(-1_000_001),
            DateTime::from_timestamp(-2, 999_999_000).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn negative_input_timestamp_ms_to_datetime() {
        assert_eq!(
            timestamp_ms_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_000_000).map(|x| x.naive_utc())
        );

        assert_eq!(
            timestamp_ms_to_datetime(-1_001),
            DateTime::from_timestamp(-2, 999_000_000).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn negative_input_date64_to_datetime() {
        assert_eq!(
            date64_to_datetime(-1),
            DateTime::from_timestamp(-1, 999_000_000).map(|x| x.naive_utc())
        );

        assert_eq!(
            date64_to_datetime(-1_001),
            DateTime::from_timestamp(-2, 999_000_000).map(|x| x.naive_utc())
        );
    }

    #[test]
    fn test_split_seconds() {
        let (sec, nano_sec) = split_second(100, NANOSECONDS);
        assert_eq!(sec, 0);
        assert_eq!(nano_sec, 100);

        let (sec, nano_sec) = split_second(123_000_000_456, NANOSECONDS);
        assert_eq!(sec, 123);
        assert_eq!(nano_sec, 456);

        let (sec, nano_sec) = split_second(-1, NANOSECONDS);
        assert_eq!(sec, -1);
        assert_eq!(nano_sec, 999_999_999);

        let (sec, nano_sec) = split_second(-123_000_000_001, NANOSECONDS);
        assert_eq!(sec, -124);
        assert_eq!(nano_sec, 999_999_999);
    }
}
