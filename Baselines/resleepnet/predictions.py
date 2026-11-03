"""Atomic ID-indexed export, retaining original linear AHI predictions."""
import json
from pathlib import Path
import numpy as np
from cache_reader import dataset


def export(model, args, meta, names, part, path):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.partial')
    seen = set()
    coverage = {}
    with partial.open('x') as stream:
        stream.write('{\n')
        for name in names:
            expected = {r for s in meta['datasets'][name][part]['shards'] for r in s['recording_ids']}
            ds = dataset(args.cache_dir, meta, [name], part, args.test_batch_size,
                         parallel_reads=args.parallel_reads, prefetch=args.prefetch)
            if args.smoke_steps:
                ds = ds.take(args.smoke_steps)
            cohort_seen = set()
            for r, stage, mask, ahi, rids, pids in ds:
                probs, pred_ahi = model(r, training=False)
                probs = np.asarray(probs, dtype=np.float32)
                pred_ahi = np.asarray(pred_ahi, dtype=np.float32).reshape(-1)
                if probs.shape != (*stage.shape, 4) or not np.isfinite(probs).all() or not np.isfinite(pred_ahi).all():
                    raise ValueError('Invalid model outputs')
                if not np.allclose(probs.sum(axis=-1), 1., atol=1e-5):
                    raise ValueError('Stage outputs are not normalized probabilities')
                for i, value in enumerate(rids.numpy()):
                    rid = value.decode()
                    if rid not in expected or rid in seen:
                        raise ValueError('Unexpected/duplicate recording: ' + rid)
                    indices = np.flatnonzero(mask[i].numpy())
                    selected = probs[i, indices]
                    result = dict(dataset=name, participant_id=pids[i].numpy().decode(),
                                  n_epochs_total=1200, n_epochs_valid=len(indices), epoch_indices=indices.tolist(),
                                  sleep_stage_probabilities=selected.tolist(), sleep_stage_pred=selected.argmax(-1).tolist(),
                                  sleep_stage_true=stage[i].numpy()[indices].tolist(),
                                  ahi_pred=float(pred_ahi[i]), ahi_true=float(ahi[i]),
                                  ahi_pred_nonnegative=float(max(0., pred_ahi[i])))
                    if seen:
                        stream.write(',\n')
                    stream.write(json.dumps(rid) + ':' + json.dumps(result, allow_nan=False, separators=(',', ':')))
                    seen.add(rid); cohort_seen.add(rid)
            if not args.smoke_steps and cohort_seen != expected:
                raise ValueError('Prediction coverage mismatch: ' + name)
            if not cohort_seen:
                raise ValueError('Empty prediction cohort: ' + name)
            coverage[name] = dict(expected=len(expected), exported=len(cohort_seen))
        stream.write('\n}\n')
    partial.replace(path)
    return coverage
