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

//! Pairwise (apples-to-apples) Avro → Arrow `RecordBatch` benchmarks:
//!
//! Each `arrow-avro` case has a directly comparable `apache-avro` case with
//! the **same** batch size and projection.
//!
//! Pairs (3 total):
//!   • Full schema, bs=8192     → `arrow-avro/full/bs=8192`  vs `apache-avro/full/bs=8192`
//!   • Projected (id,name),8192 → `arrow-avro/projected/bs=8192` vs `apache-avro/projected/bs=8192`
//!   • Full schema, bs=1024     → `arrow-avro/full/bs=1024`  vs `apache-avro/full/bs=1024`
//!
//! Instrumentation:
//!   • Criterion 0.7 (HTML reports) + Plotters backend (no gnuplot dependency).
//!   • pprof flamegraphs via a Criterion 0.7-compatible local profiler hook
//!     (run benches with `--profile-time`).
//!   • allocation-counter for per-iteration allocation stats.
//!
//! Dev-dependencies (Cargo.toml):
//! ```toml
//! [dev-dependencies]
//! criterion = { version = "0.7.0", features = ["html_reports"] }
//! pprof = { version = "0.15", features = ["flamegraph", "criterion"] }
//! allocation-counter = "0.8"
//! tempfile = "3"
//! ```
//!
//! Notes / references:
//!   • `ReaderBuilder` is the documented entry point for arrow-avro reading,
//!     with knobs like `with_batch_size` and `with_reader_schema`. ¹ ²
//!   • Arrow builders expose `len`, `is_empty`, `finish`; use `with_capacity`
//!     to reduce reallocations. ³ ⁴
//!   • Criterion: `--profile-time` runs without saving usual results; we
//!     create our own profile dir in the profiler hook. ⁵ ⁶
//!   • Avro named-type resolution matches record names; keep the
//!     top-level record name consistent (`"topLevelRecord"`) in projections. ⁷
//!   • `BufReader` default capacity is 8 KiB; we set a larger explicit capacity
//!     for bulk I/O. ⁸
//!
//! ¹ https://arrow.apache.org/rust/arrow_avro/reader/struct.ReaderBuilder.html
//! ² https://arrow.apache.org/rust/arrow_avro/reader/
//! ³ https://docs.rs/arrow/latest/arrow/array/builder/trait.ArrayBuilder.html
//! ⁴ https://docs.rs/arrow/latest/arrow/array/type.StringBuilder.html
//! ⁵ https://bheisler.github.io/criterion.rs/book/user_guide/profiling.html
//! ⁶ https://docs.rs/criterion/latest/criterion/
//! ⁷ https://avro.apache.org/docs/1.11.1/specification/
//! ⁸ https://doc.rust-lang.org/std/io/struct.BufReader.html

use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use allocation_counter as alloc_counter;

use apache_avro::{types::Value, Reader as AvroReader};

use arrow_array::{
    builder::ArrayBuilder, // brings `len`, `is_empty`, `finish` into scope for builders
    builder::{BooleanBuilder, Float64Builder, Int64Builder, StringBuilder},
    ArrayRef,
    BooleanArray,
    Float64Array,
    Int64Array,
    RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};

use criterion::profiler::Profiler;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlottingBackend, SamplingMode,
    Throughput,
};
use tempfile::{tempdir, TempDir};

// arrow-avro reader/writer
use arrow_avro::reader::ReaderBuilder;
use arrow_avro::schema::AvroSchema;
use arrow_avro::writer::format::AvroOcfFormat;
use arrow_avro::writer::WriterBuilder;

// -------------------------------
// Local Criterion 0.7 pprof hook
// -------------------------------

/// Minimal Criterion 0.7 profiler using `pprof` to emit a flamegraph
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

impl Profiler for FlamegraphProfiler {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &std::path::Path) {
        self.guard = pprof::ProfilerGuard::new(self.frequency).ok();
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        if let Some(guard) = self.guard.take() {
            if let Ok(report) = guard.report().build() {
                let sanitized = benchmark_id.replace(['/', '\\'], "_");

                // Ensure the directory exists even in `--profile-time` mode, where
                // Criterion does not save its usual results. We write to:
                //   target/criterion/<bench>/profile/<sanitized>_flamegraph.svg
                let out_dir = benchmark_dir.join("profile");
                if let Err(e) = std::fs::create_dir_all(&out_dir) {
                    eprintln!("pprof: failed to create {:?}: {e}", out_dir);
                    return;
                }

                let out_path = out_dir.join(format!("{sanitized}_flamegraph.svg"));
                match File::create(&out_path) {
                    Ok(file) => match report.flamegraph(file) {
                        Ok(()) => eprintln!("pprof: flamegraph -> {}", out_path.display()),
                        Err(e) => {
                            eprintln!("pprof: failed to write flamegraph {:?}: {e}", out_path)
                        }
                    },
                    Err(e) => eprintln!("pprof: couldn't create {:?}: {e}", out_path),
                }
            }
        }
    }
}

// -------------------------------
// Benchmark knobs
// -------------------------------

/// I/O buffer for both readers (avoid small 8 KiB default buffers).
const IO_BUF_CAP: usize = 256 * 1024;

/// Dataset sizes (rows).
const ROW_SIZES: &[usize] = &[10_000, 100_000];

/// Arrow-avro reader batch sizes used in pairings.
const BS_8192: usize = 8_192;
const BS_1024: usize = 1_024;

// -------------------------------
// Data generation & OCF writing
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

/// Write uncompressed OCF and return file size (bytes).
fn write_ocf<P: AsRef<Path>>(path: P, batches: &[RecordBatch]) -> u64 {
    let file = File::create(path).expect("create OCF file");
    let writer_schema = batches[0].schema().as_ref().clone();

    // Uncompressed OCF (no compression benchmarking).
    let mut w = WriterBuilder::new(writer_schema)
        .build::<_, AvroOcfFormat>(file)
        .expect("WriterBuilder::build");

    for b in batches {
        w.write(b).expect("write batch");
    }
    w.finish().expect("finish OCF");

    // Return file size for throughput reporting
    let file = w.into_inner();
    file.metadata().expect("metadata").len()
}

/// Prepare one OCF file for a given row count.
/// Returns (dir_guard, path, schema, bytes).
fn prepare_dataset(rows: usize) -> (TempDir, PathBuf, Arc<Schema>, u64) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("compare_readers_{rows}.avro"));

    // Write two batches to exercise multi-batch iteration:
    let batch1 = make_batch(rows / 2);
    let batch2 = make_batch(rows - (rows / 2));
    let bytes = write_ocf(&path, &[batch1.clone(), batch2.clone()]);
    (dir, path, batch1.schema(), bytes)
}

// -------------------------------
// Readers
// -------------------------------

#[cfg(debug_assertions)]
#[inline]
fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Read OCF at `path` using arrow-avro with supplied knobs.
/// Returns all `RecordBatch`es.
fn read_with_arrow_avro(
    path: &Path,
    batch_size: usize,
    use_reader_projection: bool,
) -> Vec<RecordBatch> {
    let file = File::open(path).expect("open OCF");
    let reader = BufReader::with_capacity(IO_BUF_CAP, file);
    let mut builder = ReaderBuilder::new().with_batch_size(batch_size);
    if use_reader_projection {
        // Reuse a single parsed projection schema across iterations
        builder = builder.with_reader_schema(projection_schema().clone());
    }

    let mut r = builder.build(reader).expect("ReaderBuilder::build");
    let mut out = Vec::new();
    // Reader::next() -> Option<Result<RecordBatch, ArrowError>>
    while let Some(res) = r.next() {
        let batch = res.expect("read batch");
        out.push(batch);
    }
    out
}

/// Helper: flush Apache builders into a RecordBatch (full or projected).
fn flush_chunk_builders(
    id_b: &mut Int64Builder,
    name_b: &mut StringBuilder,
    score_b: &mut Float64Builder,
    active_b: &mut BooleanBuilder,
    schema_full: &Arc<Schema>,
    schema_proj: &Arc<Schema>,
    project: bool,
    out: &mut Vec<RecordBatch>,
) {
    if id_b.len() == 0 {
        return;
    }
    if project {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(name_b.finish()) as ArrayRef,
        ];
        let batch =
            RecordBatch::try_new(schema_proj.clone(), arrays).expect("build projected batch");
        out.push(batch);
    } else {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(id_b.finish()) as ArrayRef,
            Arc::new(name_b.finish()) as ArrayRef,
            Arc::new(score_b.finish()) as ArrayRef,
            Arc::new(active_b.finish()) as ArrayRef,
        ];
        let batch = RecordBatch::try_new(schema_full.clone(), arrays).expect("build full batch");
        out.push(batch);
    }
}

/// Read OCF with apache-avro and build Arrow **in chunks** using Arrow builders.
/// `chunk_size` matches `arrow-avro`'s batch_size for apples-to-apples comparison.
/// If `project` is true, only `{id,name}` are produced.
fn read_with_apache_avro_to_arrow_chunked(
    path: &Path,
    chunk_size: usize,
    project: bool,
) -> Vec<RecordBatch> {
    let file = File::open(path).expect("open OCF");
    let reader = AvroReader::new(BufReader::with_capacity(IO_BUF_CAP, file))
        .expect("apache-avro Reader::new");

    let schema_full = Arc::new(build_schema());
    let schema_proj = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let mut out_batches = Vec::new();

    // Pre-allocate builders to chunk_size; strings get an approximate byte budget.
    let mut id_b = Int64Builder::with_capacity(chunk_size);
    let mut name_b = StringBuilder::with_capacity(chunk_size, chunk_size * 16);
    let mut score_b = Float64Builder::with_capacity(chunk_size);
    let mut active_b = BooleanBuilder::with_capacity(chunk_size);
    let mut rows_in_chunk = 0usize;

    for value in reader {
        match value.expect("read avro value") {
            Value::Record(fields) => {
                let mut id_v: Option<i64> = None;
                let mut name_v: Option<String> = None;
                let mut score_v: Option<f64> = None;
                let mut active_v: Option<bool> = None;

                for (fname, v) in fields {
                    match (fname.as_str(), v) {
                        ("id", Value::Long(x)) => id_v = Some(x),
                        ("name", Value::String(s)) => name_v = Some(s),
                        ("score", Value::Double(x)) => score_v = Some(x),
                        ("active", Value::Boolean(b)) => active_v = Some(b),
                        // tolerate alternative numeric encodings
                        ("id", Value::Int(x)) => id_v = Some(x as i64),
                        ("score", Value::Float(x)) => score_v = Some(x as f64),
                        (col, other) => panic!("Unhandled field {col} with value {other:?}"),
                    }
                }

                // Push to builders (non-null synthetic data)
                id_b.append_value(id_v.expect("id"));
                name_b.append_value(&name_v.expect("name"));
                if !project {
                    score_b.append_value(score_v.expect("score"));
                    active_b.append_value(active_v.expect("active"));
                }

                rows_in_chunk += 1;
                if rows_in_chunk == chunk_size {
                    flush_chunk_builders(
                        &mut id_b,
                        &mut name_b,
                        &mut score_b,
                        &mut active_b,
                        &schema_full,
                        &schema_proj,
                        project,
                        &mut out_batches,
                    );
                    // Builders were finished and reset; retain capacity for next chunk
                    rows_in_chunk = 0;
                }
            }
            other => panic!("Unexpected top-level Avro value: {other:?}"),
        }
    }

    // Flush final partial chunk if any
    flush_chunk_builders(
        &mut id_b,
        &mut name_b,
        &mut score_b,
        &mut active_b,
        &schema_full,
        &schema_proj,
        project,
        &mut out_batches,
    );

    out_batches
}

// -------------------------------
// Projection reader schema (Avro JSON)
// -------------------------------

/// Use writer's top-level record name ("topLevelRecord") so Avro named-type resolution succeeds.
const PROJECTION_READERSCHEMA_JSON: &str = r#"{
    "type": "record",
    "name": "topLevelRecord",
    "fields": [
        { "name": "id", "type": "long" },
        { "name": "name", "type": "string" }
    ]
}"#;

static PROJECTION_SCHEMA: OnceLock<AvroSchema> = OnceLock::new();

fn projection_schema() -> &'static AvroSchema {
    PROJECTION_SCHEMA.get_or_init(|| AvroSchema::new(PROJECTION_READERSCHEMA_JSON.to_string()))
}

// -------------------------------
// Benchmarks (pairwise apples-to-apples)
// -------------------------------

fn bench_pairwise(c: &mut Criterion) {
    // Keep datasets alive for the duration of the function (no leaks).
    let mut _keep_dirs: Vec<TempDir> = Vec::with_capacity(ROW_SIZES.len());

    for &rows in ROW_SIZES {
        let (dir, ocf_path, _schema, ocf_bytes) = prepare_dataset(rows);
        _keep_dirs.push(dir);

        let mut group = c.benchmark_group(format!("pairwise/rows={rows}/bytes={ocf_bytes}"));
        group.throughput(Throughput::Bytes(ocf_bytes));

        // Tuning for long vs short runs
        if rows >= 100_000 {
            group.sample_size(18);
            group.sampling_mode(SamplingMode::Flat);
            group.measurement_time(Duration::from_secs(16));
            group.warm_up_time(Duration::from_secs(1));
        } else {
            group.sample_size(25);
            group.measurement_time(Duration::from_secs(10));
            group.warm_up_time(Duration::from_secs(1));
        }

        // --- Pair 1: Full schema, bs=8192 ---
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/full/bs=8192", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches = read_with_arrow_avro(&ocf_path, BS_8192, false);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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
        group.bench_with_input(
            BenchmarkId::new("apache-avro/full/bs=8192", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches =
                            read_with_apache_avro_to_arrow_chunked(&ocf_path, BS_8192, false);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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

        // --- Pair 2: Projected (id,name), bs=8192 ---
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/projected/bs=8192", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches = read_with_arrow_avro(&ocf_path, BS_8192, true);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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
        group.bench_with_input(
            BenchmarkId::new("apache-avro/projected/bs=8192", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches =
                            read_with_apache_avro_to_arrow_chunked(&ocf_path, BS_8192, true);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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

        // --- Pair 3: Full schema, bs=1024 ---
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/full/bs=1024", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches = read_with_arrow_avro(&ocf_path, BS_1024, false);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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
        group.bench_with_input(
            BenchmarkId::new("apache-avro/full/bs=1024", rows),
            &rows,
            |b, &_r| {
                b.iter(|| {
                    let mem = alloc_counter::measure(|| {
                        let batches =
                            read_with_apache_avro_to_arrow_chunked(&ocf_path, BS_1024, false);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
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

// -------------------------------
// Criterion harness
// -------------------------------

fn benches(c: &mut Criterion) {
    bench_pairwise(c);
}

criterion_group! {
    name = compare_readers;
    config = Criterion::default()
        .plotting_backend(PlottingBackend::Plotters) // use Plotters backend; no gnuplot needed
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(1))
        .configure_from_args()
        .with_profiler(FlamegraphProfiler::new(100));
    targets = benches
}
criterion_main!(compare_readers);
