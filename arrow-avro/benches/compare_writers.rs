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

//! Pairwise **RecordBatch → Avro (OCF)** writer benchmarks.
//!
//! Mirrors the style of `benches/compare_readers.rs`:
//! - Compares `arrow-avro` vs `apache-avro` using the same `RecordBatch`
//! - Captures allocation counts/bytes via `allocation-counter`
//! - Emits **flamegraphs** when run with `--profile-time N` using a custom
//!   Criterion profiler that integrates `pprof` directly.
//!
//! Run:
//! ```bash
//! cargo bench -p arrow-avro --bench compare_writers -- --profile-time 10
//! ```
//!
//! Dev-deps you may need in this crate's Cargo.toml (examples shown):
//! ```toml
//! [dev-dependencies]
//! criterion = { version = "0.7", features = ["html_reports"] }
//! pprof = { version = "0.15", features = ["flamegraph"] }   # <- no `criterion` feature here
//! allocation-counter = "0.8"
//! apache-avro = "0.20"
//! ```

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use allocation_counter as alloc_counter;

use apache_avro::types::Value as ApacheValue;
use apache_avro::{Schema as ApacheSchema, Writer as ApacheWriter};

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use arrow_avro::schema::AvroSchema;
use arrow_avro::writer::AvroWriter;

use criterion::profiler::Profiler;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};

// pprof (direct, no criterion integration to avoid version mismatch)
use pprof::ProfilerGuardBuilder;

/// Where we *also* ensure a flamegraph copy is written, even in `--profile-time`
/// (which disables Criterion's normal on-disk outputs).
const CRATE_TARGET_CRITERION_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/criterion");

// -------------------------------
// Custom Criterion 0.7 pprof hook
// -------------------------------

/// Minimal Criterion profiler using `pprof` to emit a flamegraph
/// per benchmark when `--profile-time` is supplied.
///
/// Tip: for best stacks on some platforms, compile with frame pointers:
/// `RUSTFLAGS="-C force-frame-pointers=yes"`.
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
        // Recommended blocklist from pprof docs to avoid signal-handling deadlocks
        self.guard = ProfilerGuardBuilder::default()
            .frequency(self.frequency) // samples/sec
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .ok();
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        if let Some(guard) = self.guard.take() {
            if let Ok(report) = guard.report().build() {
                let sanitized = benchmark_id.replace('/', "_").replace('\\', "_");
                // 1) Write under the Criterion-provided dir for this benchmark
                let in_run = benchmark_dir
                    .join("profile")
                    .join(format!("{sanitized}_flamegraph.svg"));
                // 2) Also write a deterministic copy under the crate's target/criterion
                let fallback = PathBuf::from(CRATE_TARGET_CRITERION_DIR)
                    .join("profile")
                    .join(format!("{sanitized}_flamegraph.svg"));

                // Try both; print errors but don't fail the run
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

// -------------------------------
// Benchmark knobs
// -------------------------------

/// Dataset sizes (rows).
const ROW_SIZES: &[usize] = &[10_000, 100_000];

// -------------------------------
// Data generation / helpers
// -------------------------------

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

/// Encode a single `RecordBatch` into Avro **OCF** (uncompressed) using `arrow-avro`.
fn encode_with_arrow_avro_ocf(batch: &RecordBatch) -> Vec<u8> {
    let schema_owned: Schema = (*batch.schema()).clone();
    let mut writer = AvroWriter::new(Vec::<u8>::with_capacity(1024), schema_owned)
        .expect("create arrow-avro AvroWriter");
    writer.write(batch).expect("write batch");
    writer.finish().expect("finish writer");
    writer.into_inner()
}

/// Encode a single `RecordBatch` into Avro **OCF** (uncompressed) using `apache-avro`.
fn encode_with_apache_avro_ocf(batch: &RecordBatch) -> Vec<u8> {
    // Build Avro schema JSON from the Arrow schema
    let avro_json = AvroSchema::try_from(batch.schema().as_ref())
        .expect("arrow->avro schema")
        .json_string;
    let schema = ApacheSchema::parse_str(&avro_json).expect("parse avro schema");

    // Append rows as apache_avro::types::Value::Record
    let mut writer = ApacheWriter::new(&schema, Vec::<u8>::with_capacity(1024));

    let id = batch
        .column_by_name("id")
        .expect("id col")
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("id i64");
    let name = batch
        .column_by_name("name")
        .expect("name col")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("name utf8");
    let score = batch
        .column_by_name("score")
        .expect("score col")
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("score f64");
    let active = batch
        .column_by_name("active")
        .expect("active col")
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("active bool");

    for i in 0..batch.num_rows() {
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

// -------------------------------
// Benchmarks
// -------------------------------

fn bench_pairwise(c: &mut Criterion) {
    for &rows in ROW_SIZES {
        let batch = make_batch(rows);

        // Precompute size via arrow-avro to set throughput on group
        let baseline_bytes = encode_with_arrow_avro_ocf(&batch).len() as u64;

        let mut group = c.benchmark_group(format!("pairwise/rows={rows}/bytes~={baseline_bytes}"));
        group.throughput(Throughput::Bytes(baseline_bytes));

        // Sampling / timing knobs (roughly match compare_readers defaults)
        if rows >= 100_000 {
            group.sample_size(18);
            group.sampling_mode(SamplingMode::Flat);
            group.measurement_time(Duration::from_secs(12));
            group.warm_up_time(Duration::from_secs(1));
        } else {
            group.sample_size(30);
            group.measurement_time(Duration::from_secs(10));
            group.warm_up_time(Duration::from_secs(1));
        }

        // --- arrow-avro: OCF (no compression) ---
        group.bench_with_input(BenchmarkId::new("arrow-avro/ocf", rows), &rows, |b, &_r| {
            b.iter(|| {
                let mem = alloc_counter::measure(|| {
                    let bytes = encode_with_arrow_avro_ocf(&batch);
                    black_box(bytes);
                });
                black_box((
                    mem.count_total,
                    mem.bytes_total,
                    mem.count_max,
                    mem.bytes_max,
                ))
            });
        });

        // --- apache-avro: OCF (no compression) ---
        group.bench_with_input(
            BenchmarkId::new("apache-avro/ocf", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let bytes = encode_with_apache_avro_ocf(&batch);
                        black_box(bytes);
                    });
                    black_box((
                        mem.count_total,
                        mem.bytes_total,
                        mem.count_max,
                        mem.bytes_max,
                    ))
                });
            },
        );

        group.finish();
    }
}

fn benches(c: &mut Criterion) {
    bench_pairwise(c);
}

criterion_group! {
    name = compare_writers;
    config = Criterion::default()
        .configure_from_args() // picks up --profile-time
        // Enable flamegraph generation when running with `--profile-time N`
        .with_profiler(FlamegraphProfiler::new(100));
    targets = benches
}

criterion_main!(compare_writers);
