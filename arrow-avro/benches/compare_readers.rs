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

//! Pairwise (apples-to-apples) Avro to Arrow `RecordBatch` benchmarks.
//!
//! Each `arrow-avro` case has a directly comparable `apache-avro` case with
//! the **same** batch size and projection.
//!
//! Pairs (4 total) **short IDs to avoid violin-label clipping**:
//!   • Full schema, bs=8192      `arrow-avro/f8`      vs `apache-avro/f8`
//!   • Projected (id,name),8192  `arrow-avro/p8`      vs `apache-avro/p8`   (pushdown vs. reader-schema resolution)
//!   • Full schema, bs=1024      `arrow-avro/f1`      vs `apache-avro/f1`
//!   • Projected, **no pushdown**,8192  `arrow-avro/np` vs `apache-avro/np`
//!     (both decode full, **materialize all 4 Arrow arrays**, then project to `{id,name}`)
//!
//! Instrumentation:
//!   • Criterion 0.7 (HTML reports) + Plotters backend (no gnuplot).
//!   • pprof flamegraphs via a Criterion 0.7-compatible local profiler hook (`--profile-time`).
//!   • allocation-counter for per-iteration allocation stats (feature-gated: `--features alloc_counts`).
//!
//! Notes on labels & clipping: Criterion’s violin plot builds y‑axis labels from
//! `BenchmarkId::as_title()` (function ID + parameter). Keeping both short avoids
//! clipping against Plotters’ default ~60 px label area. See Criterion sources.

use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

// Allocation measurement is feature-gated to avoid swapping the global allocator by default.
#[cfg(feature = "alloc_counts")]
use allocation_counter as alloc_counter;

use apache_avro::{types::Value, Reader as AvroReader, Schema as ApacheSchema};
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
use arrow_avro::reader::ReaderBuilder;
use arrow_avro::schema::AvroSchema;
use arrow_avro::writer::format::AvroOcfFormat;
use arrow_avro::writer::WriterBuilder;
use arrow_schema::{DataType, Field, Schema};
use criterion::profiler::Profiler;
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, PlottingBackend,
    SamplingMode, Throughput,
};
use tempfile::{tempdir, TempDir};

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

/// I/O buffer for both readers (avoid small 8 KiB default buffers).
const IO_BUF_CAP: usize = 256 * 1024;

/// Dataset sizes (rows).
const ROW_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Arrow-avro reader batch sizes used in pairings.
const BS_8192: usize = 8_192;
const BS_1024: usize = 1_024;

/// Indices used for `{id, name}` projection on `RecordBatch`.
/// Keep these in sync with `build_schema()` ordering (`id`, `name`, `score`, `active`).
const PROJ_COLS: [usize; 2] = [0, 1];

/// Whether to *disable* extra buffering around `Cursor<&[u8]>`
/// (both libraries) and pass the cursor directly as a `BufRead`.
///
/// Controlled via env var:
///   AVRO_BENCH_NO_BUF=1  → no wrapper
fn no_extra_buffering() -> bool {
    std::env::var_os("AVRO_BENCH_NO_BUF").is_some()
}

/// Wrap an in‑memory cursor into either a `BufReader` (default) or return it directly
/// (when `AVRO_BENCH_NO_BUF=1`), always as a `Box<dyn BufRead>`.
fn maybe_buffer<'a>(cursor: Cursor<&'a [u8]>) -> Box<dyn BufRead + 'a> {
    if no_extra_buffering() {
        Box::new(cursor)
    } else {
        Box::new(BufReader::with_capacity(IO_BUF_CAP, cursor))
    }
}

/// Build a compact label for the `rows` parameter used in `BenchmarkId::new(...)`.
fn short_rows(rows: usize) -> String {
    const K: usize = 1_000;
    const M: usize = 1_000_000;
    const B: usize = 1_000_000_000;
    if rows >= B && rows % B == 0 {
        format!("{}B", rows / B)
    } else if rows >= M && rows % M == 0 {
        format!("{}M", rows / M)
    } else if rows >= K && rows % K == 0 {
        format!("{}K", rows / K)
    } else {
        rows.to_string()
    }
}

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
/// Returns (dir_guard, path, schema, bytes_on_disk).
fn prepare_dataset(rows: usize) -> (TempDir, PathBuf, Arc<Schema>, u64) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("compare_readers_{rows}.avro"));
    let batch1 = make_batch(rows / 2);
    let batch2 = make_batch(rows - (rows / 2));
    let bytes = write_ocf(&path, &[batch1.clone(), batch2.clone()]);
    (dir, path, batch1.schema(), bytes)
}

#[cfg(feature = "alloc_counts")]
fn measure_decode<T, F: FnOnce() -> T>(f: F) -> (T, (u64, u64, u64, u64)) {
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
fn measure_decode<T, F: FnOnce() -> T>(f: F) -> (T, (u64, u64, u64, u64)) {
    (f(), (0, 0, 0, 0))
}

#[cfg(debug_assertions)]
#[inline]
fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Decode one run using arrow-avro from a `Cursor<&[u8]>`.
fn decode_arrow_avro_from_cursor(
    cursor: Cursor<&[u8]>,
    batch_size: usize,
    use_reader_projection: bool,
    rows_hint: usize,
) -> Vec<RecordBatch> {
    // `ReaderBuilder` is the documented entry point for Arrow Avro OCF reading
    // (projection: `with_reader_schema`, batch sizing: `with_batch_size`).
    let reader = maybe_buffer(cursor);
    let mut builder = ReaderBuilder::new().with_batch_size(batch_size);
    if use_reader_projection {
        // Reuse a single parsed projection schema across iterations
        builder = builder.with_reader_schema(projection_schema().clone());
    }
    let mut r = builder.build(reader).expect("ReaderBuilder::build");
    let mut out = Vec::with_capacity((rows_hint + batch_size - 1) / batch_size + 1);
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
    if id_b.is_empty() {
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

/// How to project with apache-avro.
#[derive(Clone, Copy)]
enum ApacheProjection {
    /// Decode full records and produce all columns.
    None,
    /// Use reader-schema pushdown; decode only the projected fields at the Avro layer.
    ReaderPushdown,
    /// Decode full records and **materialize all 4 Arrow arrays**, but only *return*
    /// `{id,name}` columns by projecting after decode (no Avro pushdown).
    ColumnsOnly,
}

/// Read with apache-avro from a `Cursor<&[u8]>` and build Arrow **in chunks** using Arrow builders.
/// `chunk_size` matches `arrow-avro`'s batch_size for apples-to-apples comparison.
fn decode_apache_avro_from_cursor(
    cursor: Cursor<&[u8]>,
    chunk_size: usize,
    mode: ApacheProjection,
    schema_full: &Arc<Schema>,
    schema_proj: &Arc<Schema>,
    rows_hint: usize,
) -> Vec<RecordBatch> {
    let src = maybe_buffer(cursor);
    let reader = match mode {
        ApacheProjection::ReaderPushdown => {
            AvroReader::with_schema(projection_schema_apache(), src)
        }
        _ => AvroReader::new(src),
    }
    .expect("apache-avro Reader");
    // Capacity: ceil(rows / chunk_size) + small slack.
    let mut out_batches = Vec::with_capacity((rows_hint + chunk_size - 1) / chunk_size + 1);
    // Pre-allocate builders to chunk_size; strings get an approximate byte budget (~16 B/user_N).
    let mut id_b = Int64Builder::with_capacity(chunk_size);
    let mut name_b = StringBuilder::with_capacity(chunk_size, chunk_size * 16);
    let mut score_b = Float64Builder::with_capacity(chunk_size);
    let mut active_b = BooleanBuilder::with_capacity(chunk_size);
    let mut rows_in_chunk = 0usize;
    let project_on_flush = matches!(mode, ApacheProjection::ReaderPushdown);
    // One-shot runtime assertion (release builds) that Avro-level projection truly projects.
    #[cfg(not(debug_assertions))]
    static READER_PUSHDOWN_ASSERT_ONCE: OnceLock<()> = OnceLock::new();
    for value in reader {
        match value.expect("read avro value") {
            Value::Record(fields) => {
                match mode {
                    ApacheProjection::ReaderPushdown => {
                        // Avro-level projection yields only id,name (2 fields).
                        debug_assert_eq!(fields.len(), 2);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(fields[0].0, "id");
                            debug_assert_eq!(fields[1].0, "name");
                        }
                        #[cfg(not(debug_assertions))]
                        {
                            READER_PUSHDOWN_ASSERT_ONCE.get_or_init(|| {
                                assert!(
                                    fields.len() == 2,
                                    "apache-avro ReaderPushdown returned {} fields (expected 2 for {{id,name}})",
                                    fields.len()
                                );
                                if fields.len() == 2 {
                                    assert_eq!(fields[0].0, "id");
                                    assert_eq!(fields[1].0, "name");
                                }
                                ()
                            });
                        }
                        let id = match &fields[0].1 {
                            Value::Long(x) => *x,
                            Value::Int(x) => *x as i64,
                            other => panic!("id {:?}", other),
                        };
                        let name = match &fields[1].1 {
                            Value::String(s) => s.as_str(),
                            other => panic!("name {:?}", other),
                        };
                        id_b.append_value(id);
                        name_b.append_value(name);
                    }
                    ApacheProjection::None | ApacheProjection::ColumnsOnly => {
                        debug_assert_eq!(fields.len(), 4);
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(fields[0].0, "id");
                            debug_assert_eq!(fields[1].0, "name");
                            debug_assert_eq!(fields[2].0, "score");
                            debug_assert_eq!(fields[3].0, "active");
                        }
                        let id = match &fields[0].1 {
                            Value::Long(x) => *x,
                            Value::Int(x) => *x as i64,
                            other => panic!("id {:?}", other),
                        };
                        let name = match &fields[1].1 {
                            Value::String(s) => s.as_str(),
                            other => panic!("name {:?}", other),
                        };
                        id_b.append_value(id);
                        name_b.append_value(name);
                        let score = match &fields[2].1 {
                            Value::Double(x) => *x,
                            Value::Float(x) => *x as f64,
                            other => panic!("score {:?}", other),
                        };
                        let active = match &fields[3].1 {
                            Value::Boolean(b) => *b,
                            other => panic!("active {:?}", other),
                        };
                        score_b.append_value(score);
                        active_b.append_value(active);
                    }
                }
                rows_in_chunk += 1;
                if rows_in_chunk == chunk_size {
                    // Only project *now* for ReaderPushdown; otherwise build full 4 columns.
                    flush_chunk_builders(
                        &mut id_b,
                        &mut name_b,
                        &mut score_b,
                        &mut active_b,
                        schema_full,
                        schema_proj,
                        project_on_flush,
                        &mut out_batches,
                    );
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
        schema_full,
        schema_proj,
        project_on_flush,
        &mut out_batches,
    );
    // For "no pushdown" case, mirror arrow-avro: build full arrays then project to `{id,name}`.
    if matches!(mode, ApacheProjection::ColumnsOnly) {
        out_batches
            .into_iter()
            .map(|rb| rb.project(&PROJ_COLS).expect("project [id,name]"))
            .collect()
    } else {
        out_batches
    }
}

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

static PROJECTION_APACHE: OnceLock<ApacheSchema> = OnceLock::new();
fn projection_schema_apache() -> &'static ApacheSchema {
    PROJECTION_APACHE.get_or_init(|| {
        ApacheSchema::parse_str(PROJECTION_READERSCHEMA_JSON).expect("parse projection JSON")
    })
}

#[cfg(target_os = "macos")]
fn set_high_qos() {
    // Prefer performance cores by raising QoS; see Apple docs.
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // From <qos.h>: QOS_CLASS_USER_INITIATED has value 0x21.
        const QOS_CLASS_USER_INITIATED: u32 = 0x21;
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

fn bench_pairwise(c: &mut Criterion) {
    let mut keep_dirs: Vec<TempDir> = Vec::with_capacity(ROW_SIZES.len());
    for &rows in ROW_SIZES {
        let (dir, ocf_path, schema_full_arc, ocf_bytes_on_disk) = prepare_dataset(rows);
        keep_dirs.push(dir);
        let ocf_vec = std::fs::read(&ocf_path).expect("read OCF into memory");
        let ocf_slice = ocf_vec.as_slice();
        let schema_full = schema_full_arc.clone();
        // The projection schema used when building projected batches.
        // Assert in debug builds to surface any future schema reorderings.
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(schema_full.field(PROJ_COLS[0]).name(), "id");
            debug_assert_eq!(schema_full.field(PROJ_COLS[1]).name(), "name");
        }
        let schema_proj = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let mut group = c.benchmark_group("R".to_string());
        group.throughput(Throughput::Bytes(ocf_bytes_on_disk));
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
        // f8 (full schema, 8k batch)
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/f8", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice), // setup (not timed)
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_arrow_avro_from_cursor(cur, BS_8192, false, rows)
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("apache-avro/f8", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_apache_avro_from_cursor(
                                cur,
                                BS_8192,
                                ApacheProjection::None,
                                &schema_full,
                                &schema_proj,
                                rows,
                            )
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        // p8 (projected id,name, 8k batch)
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/p8", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_arrow_avro_from_cursor(cur, BS_8192, true, rows)
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("apache-avro/p8", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_apache_avro_from_cursor(
                                cur,
                                BS_8192,
                                ApacheProjection::ReaderPushdown,
                                &schema_full,
                                &schema_proj,
                                rows,
                            )
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        // f1 (full schema, 1k batch)
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/f1", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_arrow_avro_from_cursor(cur, BS_1024, false, rows)
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("apache-avro/f1", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_apache_avro_from_cursor(
                                cur,
                                BS_1024,
                                ApacheProjection::None,
                                &schema_full,
                                &schema_proj,
                                rows,
                            )
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        // np (no pushdown; decode full then project id,name; 8k batch)
        group.bench_with_input(
            BenchmarkId::new("arrow-avro/np", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (mut batches, mem) = measure_decode(|| {
                            decode_arrow_avro_from_cursor(cur, BS_8192, false, rows)
                        });
                        batches = batches
                            .into_iter()
                            .map(|rb| rb.project(&PROJ_COLS).expect("project [id,name]"))
                            .collect();
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("apache-avro/np", short_rows(rows)),
            &rows,
            |b, &_r| {
                b.iter_batched(
                    || Cursor::new(ocf_slice),
                    |cur| {
                        let (batches, mem) = measure_decode(|| {
                            decode_apache_avro_from_cursor(
                                cur,
                                BS_8192,
                                ApacheProjection::ColumnsOnly,
                                &schema_full,
                                &schema_proj,
                                rows,
                            )
                        });
                        #[cfg(debug_assertions)]
                        {
                            debug_assert_eq!(total_rows(&batches), rows);
                        }
                        black_box(batches);
                        black_box(mem);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.finish();
    }
}

fn benches(c: &mut Criterion) {
    #[cfg(target_os = "macos")]
    set_high_qos();
    bench_pairwise(c);
}

criterion_group! {
    name = compare_readers;
    config = Criterion::default()
        .plotting_backend(PlottingBackend::Plotters)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(1))
        .configure_from_args()
        .with_profiler(FlamegraphProfiler::new(100));
    targets = benches
}
criterion_main!(compare_readers);
