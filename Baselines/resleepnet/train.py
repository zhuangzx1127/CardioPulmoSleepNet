"""Train the two-task ResSleepNet baseline on the frozen public cohort splits."""
from runtime_support import configure_runtime, register_strategy_cleanup
configure_runtime()
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import tensorflow as tf
from cache_reader import dataset, load_metadata, sha
from model_respi import Sleep_NET, set_initialization_seed
from objective import objective
from predictions import export


def arguments():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_dir', type=Path, default=root / 'TrainingCache')
    p.add_argument('--manifest', type=Path, default=root / 'Dataset/splits.json')
    p.add_argument('--run_dir', type=Path, default=root / 'Runs/baselines/resleepnet/seed_42')
    p.add_argument('--datasets', nargs='+', choices=['mesa', 'mros', 'shhs'], default=['mesa', 'mros', 'shhs'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=32, help='Upstream global batch 32; four GPUs receive eight nights each')
    p.add_argument('--test_batch_size', type=int, default=1)
    p.add_argument('--n_epoch', type=int, default=1000)
    p.add_argument('--patience', type=int, default=20, help='Upstream early_stop=20; 0 explicitly disables early stopping')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--ahi_scale', type=float, default=500.)
    p.add_argument('--n_kernel', type=int, default=9)
    p.add_argument('--precision', choices=['float32', 'mixed_float16'], default='mixed_float16')
    p.add_argument('--expected_gpus', type=int, default=4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--parallel_reads', type=int, default=4)
    p.add_argument('--shuffle_buffer', type=int, default=64)
    p.add_argument('--prefetch', type=int, default=1)
    p.add_argument('--smoke_steps', type=int, default=0, help='Use train/val subsets; never read held-out test')
    p.add_argument('--check_data', action='store_true')
    p.add_argument('--verify_cache', action='store_true', help='Hash every selected shard, then exit')
    p.add_argument('--skip_test', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Export held-out predictions from a finished existing run')
    p.add_argument('--warm_start', type=Path, help='Old run with best_model/weights; NEW run directory, optimizer resets')
    return p.parse_args()


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def distribute_batch(strategy, batch, global_batch):
    """Split real rows evenly, including tails with at least one row per replica.

    Only tails smaller than the replica count are replicated to avoid empty
    BatchNorm inputs. Global loss denominators and metric divisor correct these
    repeated rows. No artificial padding rows enter BatchNorm.
    """
    count = int(tf.shape(batch[0])[0])
    replicas = strategy.num_replicas_in_sync
    if not 0 < count <= global_batch or global_batch % replicas:
        raise ValueError('Invalid batch count or global batch / replica count')
    repeated = count < replicas
    quotient, remainder = divmod(count, replicas)
    def value(ctx):
        if repeated:
            return batch
        index = ctx.replica_id_in_sync_group
        start = index * quotient + min(index, remainder)
        size = quotient + int(index < remainder)
        return tuple(x[start:start + size] for x in batch)
    return strategy.experimental_distribute_values_from_function(value), replicas if repeated else 1


def warm_start_info(args, meta):
    """Validate a legacy weights-only recovery without modifying the old run."""
    if not args.warm_start:
        return None
    if args.evaluate or args.smoke_steps or args.check_data or args.verify_cache:
        raise ValueError('--warm_start is only supported for full training')
    old = args.warm_start.resolve()
    if old == args.run_dir.resolve():
        raise ValueError('Warm start requires a NEW --run_dir; preserve the old run')
    source = json.loads((old / 'protocol.json').read_text())
    config = source['configuration']
    if source.get('smoke_steps', 0):
        raise ValueError('Cannot warm start a formal run from a smoke run')
    if source['manifest_sha256'] != meta['manifest_sha256'] or source['datasets'] != args.datasets:
        raise ValueError('Warm-start data split / datasets mismatch')
    if source.get('cache_metadata_sha256') != sha(args.cache_dir / 'cache.json'):
        raise ValueError('Warm-start cache metadata mismatch')
    for key in ('seed', 'precision', 'n_kernel', 'ahi_scale', 'lr', 'batch_size', 'patience'):
        if config.get(key) != getattr(args, key):
            raise ValueError('Warm-start configuration mismatch: ' + key)
    for name in ('model_respi.py', 'objective.py'):
        if source['source_sha256'].get(name) != sha(Path(__file__).with_name(name)):
            raise ValueError('Warm-start model/objective source mismatch: ' + name)
    epoch = int(source['best_epoch'])
    best = float(source['best_validation_selection'])
    if epoch < 1 or epoch >= args.n_epoch or not __import__('math').isfinite(best):
        raise ValueError('Invalid source best epoch/loss or no remaining epoch budget')
    prefix = old / 'best_model/weights'
    files = [prefix.with_suffix('.index')] + sorted(prefix.parent.glob('weights.data-*'))
    if len(files) < 2 or any(not f.is_file() or not f.stat().st_size for f in files):
        raise ValueError('Incomplete best-model checkpoint: ' + str(prefix))
    return dict(source_run=str(old), checkpoint_prefix=str(prefix),
                checkpoint_sha256={f.name: sha(f) for f in files},
                source_protocol_sha256=sha(old / 'protocol.json'),
                source_best_epoch=epoch, source_completed_epochs=source.get('completed_epochs'),
                source_best_validation_selection=best, optimizer_restored=False,
                loss_scale_restored=False, rng_restored=False, early_stop_counter_reset=True,
                epoch_origin='source protocol best_epoch; legacy weights contain no epoch counter',
                recovery_type='BEST_WEIGHTS_WARM_START_NOT_EXACT_RESUME')


def main(args):
    if (min(args.batch_size, args.test_batch_size, args.n_epoch, args.lr, args.ahi_scale, args.n_kernel,
            args.parallel_reads, args.shuffle_buffer, args.prefetch) <= 0 or args.patience < 0
            or args.smoke_steps < 0 or args.expected_gpus < 0 or len(set(args.datasets)) != len(args.datasets)):
        raise ValueError('Invalid configuration')
    if args.evaluate and (args.smoke_steps or args.check_data or args.verify_cache):
        raise ValueError('Evaluation cannot be combined with smoke/check/verify')
    if args.cpu:
        if args.expected_gpus:
            raise ValueError('--cpu requires --expected_gpus 0')
        tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    if args.expected_gpus and (not tf.test.is_built_with_cuda() or len(gpus) != args.expected_gpus):
        raise RuntimeError(f'Expected {args.expected_gpus} CUDA GPUs; found {gpus}')
    tf.keras.mixed_precision.set_global_policy(args.precision)
    set_initialization_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    meta = load_metadata(args.cache_dir, args.manifest)
    recovery = warm_start_info(args, meta)
    counts = {p: sum(meta['datasets'][d][p]['records'] for d in args.datasets) for p in ('train', 'val', 'test')}
    if args.verify_cache:
        for d in args.datasets:
            for part in ('train', 'val', 'test'):
                for shard in meta['datasets'][d][part]['shards']:
                    if sha(args.cache_dir / shard['file']) != shard['sha256']:
                        raise ValueError('Corrupt shard: ' + shard['file'])
                    print('verified', shard['file'], flush=True)
        return
    strategy = tf.distribute.MirroredStrategy(devices=['/CPU:0'] if args.cpu else None)
    register_strategy_cleanup(strategy)
    replicas = strategy.num_replicas_in_sync
    if args.batch_size % replicas:
        raise ValueError('Global batch must be divisible by replica count')
    print(f'TF={tf.__version__}; replicas={replicas}; global batch={args.batch_size}; counts={counts}', flush=True)
    raw = {}
    if not args.evaluate:
        for part in ('train', 'val'):
            ds = dataset(args.cache_dir, meta, args.datasets, part, args.batch_size,
                         seed=args.seed, parallel_reads=args.parallel_reads, prefetch=args.prefetch,
                         shuffle_buffer=args.shuffle_buffer, training=part == 'train')
            raw[part] = ds.take(args.smoke_steps) if args.smoke_steps else ds
    if args.check_data:
        for part, ds in raw.items():
            batch = next(iter(ds))
            print(part, [(x.shape.as_list(), x.dtype.name) for x in batch], flush=True)
            print('recording_ids', [r.decode() for r in batch[4].numpy()], flush=True)
        return
    with strategy.scope():
        model = Sleep_NET(args.n_kernel)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        if args.precision == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        if recovery:
            sample = next(iter(dataset(args.cache_dir, meta, [args.datasets[0]], 'val', 1, parallel_reads=1, prefetch=1)))
            model(sample[0], training=False)
            model.load_weights(recovery['checkpoint_prefix']).assert_consumed()
            print('WARM START: best weights restored; Adam, loss scale, RNG and early-stop counter reset.', flush=True)

    if args.evaluate:
        protocol = json.loads((args.run_dir / 'protocol.json').read_text())
        if protocol['status'] not in ('training_complete', 'complete') or protocol['smoke_steps']:
            raise ValueError('Evaluation requires a completed full training run')
        if (protocol['manifest_sha256'] != meta['manifest_sha256'] or protocol['datasets'] != args.datasets
                or protocol['configuration']['n_kernel'] != args.n_kernel
                or protocol['configuration']['precision'] != args.precision
                or protocol['seed'] != args.seed):
            raise ValueError('Evaluation configuration differs from training')
        for name, digest in protocol['source_sha256'].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError('Evaluation source differs from training: ' + name)
        sample = next(iter(dataset(args.cache_dir, meta, [args.datasets[0]], 'val', 1,
                                   parallel_reads=1, prefetch=1)))
        model(sample[0], training=False)
        model.load_weights(str(args.run_dir / 'best_model/weights')).assert_consumed()
        coverage = export(model, args, meta, args.datasets, 'test', args.run_dir / 'test_predictions.json')
        protocol.update(status='complete', prediction_coverage=coverage)
        write_json(args.run_dir / 'protocol.json', protocol)
        return
    if args.run_dir.exists():
        raise FileExistsError('Run directory already exists; choose --run_dir: ' + str(args.run_dir))
    args.run_dir.mkdir(parents=True)
    (args.run_dir / 'best_model').mkdir()
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol = dict(baseline='ResSleepNet two-task respiratory reproduction', upstream_commit='9fcfda7d7d71efbb1c71486c4a3827e69473a703',
                    status='running', initialization='best_weights_warm_start' if recovery else 'from scratch', seed=args.seed, smoke_steps=args.smoke_steps,
                    manifest_sha256=meta['manifest_sha256'], cache_metadata_sha256=sha(args.cache_dir / 'cache.json'),
                    datasets=args.datasets, partition_records=counts, configuration=configuration, replicas=replicas,
                    objective=f'CE summed over valid epochs / valid epochs + AHI MSE / {args.ahi_scale}',
                    checkpoint_selection='validation objective, excluding declared L2 and domain loss',
                    architecture_adaptation='domain head omitted; original encoder, stage and linear AHI heads retained',
                    train_tail='balanced real-row split when count >= replicas; only smaller tails replicated with corrected counts; no dummy BatchNorm rows',
                    tensorflow=tf.__version__, keras=tf.keras.__version__, python=sys.version, platform=platform.platform(),
                    source_sha256={})
    if recovery:
        protocol.update(recovery=recovery, best_epoch=recovery['source_best_epoch'],
                        best_validation_selection=recovery['source_best_validation_selection'],
                        completed_epochs=recovery['source_best_epoch'])
        model.save_weights(str(args.run_dir / 'best_model/weights'))
    for source in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copyfile(source, args.run_dir / source.name)
        protocol['source_sha256'][source.name] = sha(source)
    shutil.copytree(Path(__file__).with_name('upstream'), args.run_dir / 'upstream')
    shutil.copyfile(Path(__file__).with_name('upstream.patch'), args.run_dir / 'upstream.patch')
    shutil.copyfile(args.manifest, args.run_dir / 'splits.json')
    shutil.copyfile(args.cache_dir / 'cache.json', args.run_dir / 'cache.json')
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (args.run_dir / 'environment.txt').write_text(result.stdout + result.stderr)
    write_json(args.run_dir / 'protocol.json', protocol)

    def local_step(batch, training):
        with tf.GradientTape() as tape:
            loss, pieces = objective(model, batch, training, args.ahi_scale)
            scaled = optimizer.get_scaled_loss(loss) if args.precision == 'mixed_float16' else loss
        if training:
            grads = tape.gradient(scaled, model.trainable_variables)
            if args.precision == 'mixed_float16':
                grads = optimizer.get_unscaled_gradients(grads)
            if any(g is None for g in grads):
                raise ValueError('An active variable has no task gradient')
            if args.precision == 'float32':
                for g in grads:
                    tf.debugging.assert_all_finite(g, 'Nonfinite gradient')
            pieces['finite_replicas'] = tf.cast(tf.reduce_all(tf.stack([
                tf.reduce_all(tf.math.is_finite(g)) for g in grads])), tf.float32)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return pieces

    @tf.function
    def step(batch, training):
        values = strategy.run(local_step, args=(batch, training))
        return {k: strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None) for k, v in values.items()}

    def run_epoch(part):
        totals = dict(ce_sum=0., mse_sum=0., mae_sum=0., correct=0., epochs=0., nights=0.)
        steps = 0
        applied_updates = 0
        for batch in raw[part]:
            distributed, divisor = distribute_batch(strategy, batch[:4], args.batch_size)
            values = step(distributed, part == 'train')
            if part == 'train':
                applied_updates += int(float(values.pop('finite_replicas')) == replicas)
            for key, value in values.items():
                totals[key] += float(value) / divisor
            steps += 1
            write_json(args.run_dir / 'heartbeat.json', dict(partition=part, step=steps,
                       real_nights=int(totals['nights']), unix_time=time.time()))
        if min(totals['nights'], totals['epochs']) <= 0:
            raise ValueError('Empty partition: ' + part)
        if part == 'train' and applied_updates == 0:
            raise RuntimeError('No finite optimizer updates; increase smoke_steps or inspect mixed-precision loss scaling')
        if not args.smoke_steps and int(totals['nights']) != counts[part]:
            raise ValueError('Record coverage mismatch: ' + part)
        ce = totals['ce_sum'] / totals['epochs']
        mse = totals['mse_sum'] / totals['nights']
        result = dict(loss=ce + mse / args.ahi_scale, stage_ce=ce, ahi_mse=mse,
                    ahi_mae=totals['mae_sum'] / totals['nights'], accuracy=totals['correct'] / totals['epochs'],
                    nights=int(totals['nights']), valid_epochs=int(totals['epochs']), steps=steps)
        if part == 'train':
            result.update(applied_updates=applied_updates, skipped_updates=steps - applied_updates)
        return result

    start = time.monotonic(); best = recovery['source_best_validation_selection'] if recovery else float('inf'); stale = 0
    start_epoch = recovery['source_best_epoch'] if recovery else 0
    try:
        with (args.run_dir / 'train_log.csv').open('x', newline='') as stream:
            writer = None
            for epoch in range(start_epoch, args.n_epoch):
                epoch_start = time.monotonic()
                train = run_epoch('train'); val = run_epoch('val')
                if val['loss'] < best:
                    best = val['loss']; stale = 0
                    model.save_weights(str(args.run_dir / 'best_model/weights'))
                    protocol['best_epoch'] = epoch + 1
                else:
                    stale += 1
                row = dict(epoch=epoch + 1, **{'train_' + k: v for k, v in train.items()},
                           **{'val_' + k: v for k, v in val.items()}, lr=args.lr,
                           optimizer_iterations=int(optimizer.iterations.numpy()),
                           epoch_seconds=time.monotonic() - epoch_start, seconds=time.monotonic() - start)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); stream.flush()
                protocol.update(parameters=model.count_params(), completed_epochs=epoch + 1,
                                best_validation_selection=best, seconds=time.monotonic() - start)
                write_json(args.run_dir / 'protocol.json', protocol)
                print(json.dumps(row), flush=True)
                if args.patience and stale >= args.patience:
                    break
        protocol['status'] = 'smoke_trained' if args.smoke_steps else 'training_complete'
        write_json(args.run_dir / 'protocol.json', protocol)
        # Strict reload verification catches missing variables/checkpoint mismatches.
        model.load_weights(str(args.run_dir / 'best_model/weights')).assert_consumed()
        if not args.skip_test:
            part = 'val' if args.smoke_steps else 'test'
            name = 'smoke_val_predictions.json' if args.smoke_steps else 'test_predictions.json'
            protocol['prediction_coverage'] = export(model, args, meta, args.datasets, part, args.run_dir / name)
            protocol['status'] = 'smoke_complete' if args.smoke_steps else 'complete'
        write_json(args.run_dir / 'protocol.json', protocol)
    except BaseException as exc:
        protocol.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write_json(args.run_dir / 'protocol.json', protocol)
        raise


if __name__ == '__main__':
    main(arguments())
