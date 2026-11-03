"""Respiration-only adapter for the unchanged nsrr-tfrecord-v1 cache."""
import hashlib
import json
from pathlib import Path
import tensorflow as tf

PARTS = ('train', 'val', 'test')
LENGTH = 1200 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(cache_dir, manifest):
    root = Path(cache_dir)
    meta = json.loads((root / 'cache.json').read_text())
    if (meta.get('format') != 'nsrr-tfrecord-v1' or not meta.get('complete')
            or meta.get('n_epochs') != 1200 or meta.get('stage_encoding') != 'four_class'
            or meta.get('signal_dtype') not in ('float16', 'float32')):
        raise ValueError('Incompatible or incomplete cache')
    if meta['manifest_sha256'] != sha(manifest):
        raise ValueError('Cache and splits.json hashes differ')
    doc = json.loads(Path(manifest).read_text())
    patients = {p: set() for p in PARTS}
    seen = set()
    for name, parts in meta['datasets'].items():
        for part in PARTS:
            info = parts[part]
            expected = {r['recording_id'] for r in doc[name][part]}
            ids = [rid for s in info['shards'] for rid in s['recording_ids']]
            if len(ids) != info['records'] or len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError('Cache membership differs from manifest: ' + name + '/' + part)
            if seen.intersection(ids):
                raise ValueError('Duplicate recording across partitions/cohorts')
            seen.update(ids)
            patients[part].update((name, r['participant_id']) for r in doc[name][part])
            for shard in info['shards']:
                path = root / shard['file']
                try:
                    path.resolve().relative_to(root.resolve())
                except ValueError:
                    raise ValueError('Shard path escapes cache')
                if not path.is_file() or path.stat().st_size != shard['bytes']:
                    raise ValueError('Missing/truncated shard: ' + str(path))
    if any(patients[a] & patients[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Patient leakage across partitions')
    return meta


def dataset(cache_dir, meta, names, part, batch_size, *, seed=42, parallel_reads=4,
            prefetch=1, shuffle_buffer=64, training=False):
    if part not in PARTS or batch_size < 1 or min(parallel_reads, prefetch, shuffle_buffer) < 1:
        raise ValueError('Invalid reader settings')
    if training and part != 'train':
        raise ValueError('Only train may be shuffled for training')
    files = [str(Path(cache_dir) / s['file']) for d in names for s in meta['datasets'][d][part]['shards']]
    count = sum(meta['datasets'][d][part]['records'] for d in names)
    dtype = tf.float16 if meta['signal_dtype'] == 'float16' else tf.float32
    schema = {k: tf.io.FixedLenFeature([], tf.string) for k in
              ('respi', 'sleep_stage', 'is_data', 'recording_id', 'participant_id')}
    schema['ahi'] = tf.io.FixedLenFeature([], tf.float32)

    def decode(examples):
        f = tf.io.parse_example(examples, schema)
        r = tf.ensure_shape(tf.io.decode_raw(f['respi'], dtype), [None, LENGTH])[..., None]
        stage = tf.cast(tf.ensure_shape(tf.io.decode_raw(f['sleep_stage'], tf.uint8), [None, 1200]), tf.int32)
        mask = tf.cast(tf.ensure_shape(tf.io.decode_raw(f['is_data'], tf.uint8), [None, 1200]), tf.bool)
        tf.debugging.assert_all_finite(tf.cast(r, tf.float32), 'Nonfinite respiration')
        tf.debugging.assert_all_finite(f['ahi'], 'Nonfinite AHI')
        tf.debugging.assert_non_negative(f['ahi'])
        tf.debugging.assert_positive(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1))
        tf.debugging.assert_less(tf.boolean_mask(stage, mask), 4)
        return r, stage, mask, f['ahi'], f['recording_id'], f['participant_id']

    ds_files = tf.data.Dataset.from_tensor_slices(files)
    if training:
        ds_files = ds_files.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(ds_files, num_parallel_reads=parallel_reads, buffer_size=8 * 1024 * 1024)
    ds = ds.apply(tf.data.experimental.assert_cardinality(count))
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).map(decode, num_parallel_calls=parallel_reads)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = max(2, parallel_reads)
    return ds.with_options(options).prefetch(prefetch)
