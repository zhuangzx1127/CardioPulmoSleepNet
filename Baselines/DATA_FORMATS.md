# Prepared-input contracts

The JSON manifests define the participant splits. File paths below are relative
to the repository root.

## Main TrainingCache: ResSleepNet / OxiNet / wav2sleep

`TrainingCache/cache.json` has `format="nsrr-tfrecord-v1"`, `complete=true`,
`n_epochs=1200`, `stage_encoding="four_class"`, `signal_dtype="float16"` or
`"float32"`, and `manifest_sha256` equal to SHA256 of `Dataset/splits.json`.
Its `datasets[cohort][partition]` entries give record counts and `shards`, each
with relative `file`, `bytes`, `sha256`, `records`, and ordered `recording_ids`.

Each TFRecord contains one TensorFlow Example per recording, with little-endian
array bytes and the following scalar/array features:

| Feature | Representation / shape |
|---|---|
| `recording_id`, `participant_id` | UTF-8 bytes |
| `respi`, `ecg` | float16/float32, 1,228,800 samples each |
| `spo2` | float16/float32, 36,000 samples |
| `sleep_stage` | uint8, 1,200 epochs; W/Light/Deep/REM = 0/1/2/3 |
| `is_data` | uint8 binary mask, 1,200 epochs |
| `ahi` | scalar FloatList value, events/hour |

The arrays represent the existing processed ten-hour cache axis. Invalid/padded
positions remain at their original indices. The main split includes preprocessing
provenance fields and their original hashes; it is copied without reserialization.

## Native first-10-hour inputs: BCGNet / DRIVEN

`BaselineInputs/native_10h/index.json` is generated alongside the prepared
arrays according to `native_index.schema.json`. Its `manifest_sha256` refers to the main
`Dataset/splits.json` hash, not the separate native-subset split hash.

Required array keys in each native NPZ:

| Key | Shape / meaning |
|---|---|
| `recording_id`, `participant_id` | scalar public IDs |
| `abd_64hz` | (2,304,000,), raw-native ABD at 64 Hz |
| `spo2_1hz` | (36,000,), saturation in percent |
| `heartbeat` | (144,000,), ECG-derived IHR in beats/minute, 4 Hz |
| `breath` | (144,000,), filtered/normalized ABD, 4 Hz |
| `heartbeat_valid` | (144,000,), binary valid-IHR mask |
| `stage5` | (1,200,), W/N1/N2/Deep/REM = 0/1/2/3/4; unknown can be -1 |
| `stage_mask`, `bcg_stage_mask` | (1,200,), binary masks |
| `event`, `event_valid` | (36,000,), binary event labels / coverage mask |
| `ahi` | scalar reference night-level AHI |

Native signals use the raw MAT start, at most ten hours, with adjusted-XML
ClockTime alignment. Native labels follow the author-validated a0h3a adjusted
annotations, including retained `Unsure` events as positive events. This axis is
not interchangeable with the processed TrainingCache axis.

Each index `records[recording_id]` item supplies `file`, `sha256`, `cohort`,
`partition`, `source_duration_seconds`, `ahi`, `valid_stage_epochs`, and
`valid_bcg_stage_epochs`, plus ID metadata. Paths must stay inside native_10h.
The record membership and participant assignment must agree with the released
native `splits.json`. Extra source-provenance fields may be kept in the local
index. NPZ contents and index checksums must match the data used for a run.
