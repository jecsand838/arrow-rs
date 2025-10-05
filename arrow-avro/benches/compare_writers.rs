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

//! Pairwise **RecordBatch to Avro (OCF)** writer benchmarks.
//!
//! Pairs (per `rows` input):
//!   • `arrow/c`   vs `apache/c`
//!   • `arrow/h`    vs `apache/h`
//!
//! Modes:
//! - **cold** (c)
//!   - arrow: uses the Arrow schema directly (per-iteration Arrow to Avro conversion).
//!   - apache: parses Avro JSON and appends row `Value`s; 1 OCF block per batch.
//! - **hot** (h)
//!   - arrow: Arrow schema contains `avro.schema` JSON; builder parses JSON each iter.
//!   - apache: parses JSON and uses `append_ser`; 1 OCF block per batch.
//!
//! Other features:
//! - Optional allocation counts via `allocation-counter` behind `--features alloc_counts`.
//! - Uses `iter_with_large_drop` so large `Vec<u8>` drops happen **outside** timing.
//! - Emits pprof flamegraphs with `--profile-time N` via a Criterion profiler hook.
//!
//! ### macOS / signal-safety notes (pprof)
//! `pprof` samples with SIGPROF; prefer frame-pointer unwinder and blocklist
//! known libraries for safer stacks. We default to 99 Hz (env override:
//! `AVRO_BENCH_PPROF_FREQ`) and blocklist `"libc", "libgcc", "libunwind", "pthread", "vdso"`.

#[cfg(feature = "alloc_counts")]
use allocation_counter as alloc_counter;

use apache_avro::types::Value as ApacheValue;
use apache_avro::{Schema as ApacheSchema, Writer as ApacheWriter};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_avro::schema::{AvroSchema, SCHEMA_METADATA_KEY};
use arrow_avro::writer::{format::AvroOcfFormat, WriterBuilder};
use arrow_schema::{DataType, Field, Schema};
use criterion::profiler::Profiler;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use pprof::ProfilerGuardBuilder;
use serde::Serialize;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Where we *also* ensure a flamegraph copy is written, even in `--profile-time`
/// (which disables Criterion's normal on-disk outputs).
const CRATE_TARGET_CRITERION_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/criterion");

/// Default profiler sampling frequency (Hz). Tunable via `AVRO_BENCH_PPROF_FREQ`.
fn default_profiler_frequency() -> i32 {
    std::env::var("AVRO_BENCH_PPROF_FREQ")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|f| f.clamp(1, 1000))
        .unwrap_or(99)
}

/// Minimal Criterion profiler using `pprof` to emit a flamegraph
/// per benchmark when `--profile-time` is supplied.
///
/// Tip: for best stacks on some platforms, compile with frame pointers:
/// `RUSTFLAGS="-C force-frame-pointers=yes"` (and rebuild std with `-Z build-std`).
struct FlamegraphProfiler {
    frequency: i32,
    guard: Option<pprof::ProfilerGuard<'static>>,
}

impl FlamegraphProfiler {
    fn new(frequency: i32) -> Self {
        Self {
            frequency,
            guard: None,
        }
    }
}

impl FlamegraphProfiler {
    fn write_flamegraph<P: AsRef<Path>>(report: &pprof::Report, path: P) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {:?}: {e}", parent))?;
        }
        let mut file =
            std::fs::File::create(path).map_err(|e| format!("create {:?}: {e}", path))?;
        report
            .flamegraph(&mut file)
            .map_err(|e| format!("flamegraph {:?}: {e}", path))?;
        eprintln!("pprof: flamegraph saved to {}", path.display());
        Ok(())
    }
}

impl Profiler for FlamegraphProfiler {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &std::path::Path) {
        self.guard = ProfilerGuardBuilder::default()
            .frequency(self.frequency)
            .blocklist(&["libc", "libgcc", "libunwind", "pthread", "vdso"])
            .build()
            .ok();
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        if let Some(guard) = self.guard.take() {
            if let Ok(report) = guard.report().build() {
                let sanitized = benchmark_id.replace('/', "_").replace('\\', "_");
                let in_run = benchmark_dir
                    .join("profile")
                    .join(format!("{sanitized}_flamegraph.svg"));
                let fallback = PathBuf::from(CRATE_TARGET_CRITERION_DIR)
                    .join("profile")
                    .join(format!("{sanitized}_flamegraph.svg"));
                let _ = Self::write_flamegraph(&report, &in_run)
                    .map_err(|e| eprintln!("pprof: failed writing to {:?}: {e}", in_run));
                let _ = Self::write_flamegraph(&report, &fallback)
                    .map_err(|e| eprintln!("pprof: failed writing to {:?}: {e}", fallback));
            } else {
                eprintln!("pprof: failed to build report (no flamegraph generated)");
            }
        }
    }
}

/// Dataset sizes (rows).
const ROW_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
        Field::new("active", DataType::Boolean, false),
    ])
}

fn make_batch(rows: usize) -> RecordBatch {
    let ids: Vec<i64> = (0..rows as i64).collect();
    let names: Vec<String> = (0..rows).map(|i| format!("user_{i}")).collect();
    let scores: Vec<f64> = (0..rows).map(|i| (i as f64) * 0.01).collect();
    let actives: Vec<bool> = (0..rows).map(|i| i % 2 == 0).collect();
    let schema = Arc::new(build_schema());
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(ids)) as ArrayRef,
        Arc::new(StringArray::from(names)) as ArrayRef,
        Arc::new(Float64Array::from(scores)) as ArrayRef,
        Arc::new(BooleanArray::from(actives)) as ArrayRef,
    ];
    RecordBatch::try_new(schema, arrays).expect("failed to build batch")
}

/// Clone the Arrow schema and embed the Avro JSON (`avro.schema`) so the writer
/// can skip Arrow to Avro conversion in "hot-json" mode (but still parse JSON).
fn schema_with_embedded_avro_json(orig: &Schema) -> Schema {
    let avro_json = AvroSchema::try_from(orig)
        .expect("arrow->avro schema to json")
        .json_string;
    let mut md = orig.metadata().clone();
    md.insert(SCHEMA_METADATA_KEY.to_string(), avro_json);
    Schema::new_with_metadata(orig.fields().clone(), md)
}

#[derive(Serialize)]
struct RowSer<'a> {
    id: i64,
    name: &'a str,
    score: f64,
    active: bool,
}

/// Small helper to downcast the four columns once.
fn cols(batch: &RecordBatch) -> (&Int64Array, &StringArray, &Float64Array, &BooleanArray) {
    let id = batch
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let name = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let score = batch
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let active = batch
        .column_by_name("active")
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    (id, name, score, active)
}

/// arrow-avro: build writer via WriterBuilder (lets us set capacity), then write a **single OCF block** for the batch.
fn encode_arrow_ocf_with_schema(
    batch: &RecordBatch,
    schema: &Schema,
    capacity_hint: usize,
) -> Vec<u8> {
    let sink = Vec::<u8>::with_capacity(capacity_hint);
    let mut writer = WriterBuilder::new(schema.clone())
        .with_capacity(capacity_hint)
        .build::<_, AvroOcfFormat>(sink)
        .expect("build arrow-avro writer");
    writer.write(batch).expect("write batch");
    writer.finish().expect("finish writer");
    writer.into_inner()
}

/// arrow-avro (cold): uses the batch's Arrow schema directly (so it will do Arrow to Avro conversion in the timed loop).
fn encode_arrow_ocf_cold(batch: &RecordBatch, capacity_hint: usize) -> Vec<u8> {
    let schema_owned = batch.schema().as_ref().clone();
    encode_arrow_ocf_with_schema(batch, &schema_owned, capacity_hint)
}

/// apache-avro (cold, Value path): parse Avro JSON and append row Values; block size == num rows.
fn encode_apache_ocf_value_cold(
    batch: &RecordBatch,
    avro_json: &str,
    capacity_hint: usize,
) -> Vec<u8> {
    // Parse Avro JSON inside the timed loop (pairwise with arrow cold's schema setup)
    let schema = ApacheSchema::parse_str(avro_json).expect("parse avro schema");
    let n = batch.num_rows();
    let mut writer = ApacheWriter::builder()
        .schema(&schema)
        .writer(Vec::<u8>::with_capacity(capacity_hint))
        .codec(apache_avro::Codec::Null)
        .block_size(n) // one block == one batch
        .build();
    let (id, name, score, active) = cols(batch);
    for i in 0..n {
        // Value path requires owned `String` for `Value::String`.
        let record = ApacheValue::Record(vec![
            ("id".to_string(), ApacheValue::Long(id.value(i))),
            (
                "name".to_string(),
                ApacheValue::String(name.value(i).to_string()),
            ),
            ("score".to_string(), ApacheValue::Double(score.value(i))),
            ("active".to_string(), ApacheValue::Boolean(active.value(i))),
        ]);
        writer.append(record).expect("append record");
    }
    writer.into_inner().expect("writer into_inner")
}

/// apache-avro (hot, serde path): **parse Avro JSON inside the timed loop** and use `append_ser`.
fn encode_apache_ocf_serde_hot_json(
    batch: &RecordBatch,
    avro_json: &str,
    capacity_hint: usize,
) -> Vec<u8> {
    let schema = ApacheSchema::parse_str(avro_json).expect("parse avro schema");
    let n = batch.num_rows();
    let mut writer = ApacheWriter::builder()
        .schema(&schema)
        .writer(Vec::<u8>::with_capacity(capacity_hint))
        .codec(apache_avro::Codec::Null)
        .block_size(n) // one block == one batch
        .build();
    let (id, name, score, active) = cols(batch);
    for i in 0..n {
        let row = RowSer {
            id: id.value(i),
            name: name.value(i), // borrow &str to avoid extra allocation
            score: score.value(i),
            active: active.value(i),
        };
        writer.append_ser(row).expect("append row (serde)");
    }
    writer.into_inner().expect("writer into_inner")
}

/// Feature-gated allocation measurement helper.
/// Returns `(value, (count_total, bytes_total, count_max, bytes_max))`.
#[cfg(feature = "alloc_counts")]
fn measure<T, F: FnOnce() -> T>(f: F) -> (T, (u64, u64, u64, u64)) {
    let mut out = None;
    let info = alloc_counter::measure(|| {
        out = Some(f());
    });
    (
        out.expect("closure ran"),
        (
            info.count_total,
            info.bytes_total,
            info.count_max,
            info.bytes_max,
        ),
    )
}

#[cfg(not(feature = "alloc_counts"))]
fn measure<T, F: FnOnce() -> T>(f: F) -> (T, (u64, u64, u64, u64)) {
    (f(), (0, 0, 0, 0))
}

/// Abbreviate row counts for compact Criterion labels (avoid the 60px clamp truncation).
fn fmt_rows_short(rows: usize) -> String {
    if rows >= 1_000_000 {
        format!("{}M", rows / 1_000_000)
    } else if rows >= 1_000 {
        format!("{}k", rows / 1_000)
    } else {
        rows.to_string()
    }
}

/// Centralized Criterion tuning
fn tune_group(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rows: usize,
    large_secs: u64,
    small_secs: u64,
    small_sample: usize,
) {
    if rows >= 100_000 {
        g.sample_size(18);
        g.sampling_mode(SamplingMode::Flat);
        g.measurement_time(Duration::from_secs(large_secs));
        g.warm_up_time(Duration::from_secs(1));
    } else {
        g.sample_size(small_sample);
        g.measurement_time(Duration::from_secs(small_secs));
        g.warm_up_time(Duration::from_secs(1));
    }
}

fn bench_pairwise(c: &mut Criterion) {
    for &rows in ROW_SIZES {
        let batch = make_batch(rows);
        let avro_json = AvroSchema::try_from(batch.schema().as_ref())
            .expect("arrow->avro schema")
            .json_string;
        let hot_arrow_schema = schema_with_embedded_avro_json(batch.schema().as_ref());
        // Pre-compute output sizes for throughput normalization
        // (These small warmups are not included in timed loops)
        let arrow_len_cold = encode_arrow_ocf_cold(&batch, 1024).len() as u64;
        let apache_len_cold = encode_apache_ocf_value_cold(&batch, &avro_json, 1024).len() as u64;
        let arrow_len_hot_json =
            encode_arrow_ocf_with_schema(&batch, &hot_arrow_schema, 1024).len() as u64;
        let apache_len_hot_json =
            encode_apache_ocf_serde_hot_json(&batch, &avro_json, 1024).len() as u64;
        // Abbreviated label for the BenchmarkId parameter (e.g., "10k", "100k", "1M")
        let rows_label = fmt_rows_short(rows);
        // Single "W" group (mirrors readers' single "R" group)
        let mut g = c.benchmark_group("W".to_string());
        tune_group(&mut g, rows, 12, 10, 30);
        // Pair 1: cold (set throughput for this pair)
        let cold_bytes = arrow_len_cold.max(apache_len_cold);
        g.throughput(Throughput::Bytes(cold_bytes));
        g.bench_with_input(
            BenchmarkId::new("arrow-avro/c", rows_label.clone()),
            &rows,
            |b, &_r| {
                b.iter_with_large_drop(|| {
                    let (bytes, mem) =
                        measure(|| encode_arrow_ocf_cold(&batch, arrow_len_cold as usize));
                    black_box(mem);
                    bytes
                })
            },
        );
        g.bench_with_input(
            BenchmarkId::new("apache-avro/c", rows_label.clone()),
            &rows,
            |b, &_r| {
                b.iter_with_large_drop(|| {
                    let (bytes, mem) = measure(|| {
                        encode_apache_ocf_value_cold(&batch, &avro_json, apache_len_cold as usize)
                    });
                    black_box(mem);
                    bytes
                })
            },
        );
        // Pair 2: hot (set throughput for this pair)
        let hot_bytes = arrow_len_hot_json.max(apache_len_hot_json);
        g.throughput(Throughput::Bytes(hot_bytes));
        g.bench_with_input(
            BenchmarkId::new("arrow-avro/h", rows_label.clone()),
            &rows,
            |b, &_r| {
                b.iter_with_large_drop(|| {
                    // Builder will read `avro.schema` from metadata and parse JSON in-loop
                    let (bytes, mem) = measure(|| {
                        encode_arrow_ocf_with_schema(
                            &batch,
                            &hot_arrow_schema,
                            arrow_len_hot_json as usize,
                        )
                    });
                    black_box(mem);
                    bytes
                })
            },
        );
        g.bench_with_input(
            BenchmarkId::new("apache-avro/h", rows_label.clone()),
            &rows,
            |b, &_r| {
                b.iter_with_large_drop(|| {
                    // Parse Avro JSON inside the timed loop (pairwise with arrow hot)
                    let (bytes, mem) = measure(|| {
                        encode_apache_ocf_serde_hot_json(
                            &batch,
                            &avro_json,
                            apache_len_hot_json as usize,
                        )
                    });
                    black_box(mem);
                    bytes
                })
            },
        );
        g.finish();
    }
}

fn benches(c: &mut Criterion) {
    bench_pairwise(c);
}

criterion_group! {
    name = compare_writers;
    config = Criterion::default()
        .configure_from_args()
        .with_profiler(FlamegraphProfiler::new(default_profiler_frequency()));
    targets = benches
}

criterion_main!(compare_writers);
