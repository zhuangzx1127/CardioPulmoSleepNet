"""Bounded, CRC-checked TFRecord/Example reader; TensorFlow is not needed.

Only the wire types used by nsrr-tfrecord-v1 are accepted. No pickle loading,
signal-cache writes, test fitting, event-label guessing, or epoch compaction.
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARTS = ('train', 'val', 'test')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def varint(b, i):
    n = shift = 0
    while i < len(b) and shift < 70:
        v = b[i]; i += 1; n |= (v & 127) << shift
        if not v & 128: return n, i
        shift += 7
    raise ValueError('Malformed protobuf varint')

def fields(b):
    i = 0
    while i < len(b):
        tag, i = varint(b, i); wire = tag & 7
        if wire == 2:
            size, i = varint(b, i); value = b[i:i+size]; i += size
            if len(value) != size: raise ValueError('Truncated protobuf')
        elif wire == 0: value, i = varint(b, i)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4; value = b[i:i+size]; i += size
        else: raise ValueError('Unsupported protobuf wire type')
        yield tag >> 3, wire, value

def example(payload):
    features = [v for n, w, v in fields(payload) if n == 1 and w == 2]
    if len(features) != 1: raise ValueError('Malformed Example')
    out = {}
    for n, w, entry in fields(features[0]):
        if n != 1 or w != 2: raise ValueError('Malformed feature map')
        e = {n: v for n, w, v in fields(entry)}
        key = e[1].decode(); values = list(fields(e[2]))
        if len(values) != 1 or key in out: raise ValueError('Ambiguous feature')
        kind, _, nested = values[0]
        v = list(fields(nested))
        if len(v) != 1 or v[0][:2] != (1, 2): raise ValueError('Expected scalar list')
        if kind == 1: out[key] = v[0][2]
        elif kind == 2: out[key] = struct.unpack('<f', v[0][2])[0]
        else: raise ValueError('Unsupported cache feature type')
    return out

def crc(b):
    import google_crc32c
    v = google_crc32c.value(b)
    return ((v >> 15 | v << 17) + 0xa282ead8) & 0xffffffff

class Cache:
    def __init__(self, root=ROOT/'TrainingCache', manifest=ROOT/'Dataset/splits.json'):
        self.root, self.manifest = Path(root), Path(manifest)
        self.meta = json.loads((self.root/'cache.json').read_text())
        m = self.meta
        if (m.get('format'), m.get('complete'), m.get('n_epochs'), m.get('stage_encoding')) != ('nsrr-tfrecord-v1', True, 1200, 'four_class'):
            raise ValueError('Unsupported/incomplete cache')
        if m.get('signal_dtype') not in ('float16','float32'):raise ValueError('Unsupported cache dtype')
        if m['manifest_sha256'] != sha(self.manifest): raise ValueError('Manifest hash mismatch')
        self.doc = json.loads(self.manifest.read_text()); self.rows = {}; self.indexes = {}
        patients = {p:set() for p in PARTS}
        for name, parts in m['datasets'].items():
            for part in PARTS:
                records = self.doc[name][part]
                ids = [i for s in parts[part]['shards'] for i in s['recording_ids']]
                if len(ids) != parts[part]['records'] or len(ids) != len(set(ids)) or set(ids) != {r['recording_id'] for r in records}:
                    raise ValueError('Cache/manifest membership mismatch')
                for r in records:
                    if r['recording_id'] in self.rows: raise ValueError('Repeated recording')
                    self.rows[r['recording_id']] = dict(r, cohort=name, partition=part)
                    patients[part].add(r['participant_id'])
                for s in parts[part]['shards']:
                    path = (self.root/s['file']).resolve()
                    path.relative_to(self.root.resolve())
                    if not path.is_file() or path.stat().st_size != s['bytes']: raise ValueError('Missing/truncated shard: '+str(path))
        if any(patients[a]&patients[b] for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('Patient leakage')

    def index(self, cohorts, part):
        key = (tuple(cohorts), part)
        if key in self.indexes: return self.indexes[key]
        rows = []
        for name in cohorts:
            for shard in self.meta['datasets'][name][part]['shards']:
                path = self.root/shard['file']
                with path.open('rb') as f:
                    for rid in shard['recording_ids']:
                        offset = f.tell(); h = f.read(12)
                        if len(h) != 12 or crc(h[:8]) != struct.unpack('<I',h[8:])[0]: raise ValueError('TFRecord length CRC failed')
                        length = struct.unpack('<Q', h[:8])[0]
                        if length > 20*1024*1024: raise ValueError('Unexpected record size')
                        f.seek(length+4, 1); rows.append((path,offset,rid))
                    if f.tell() != shard['bytes']: raise ValueError('Shard record count mismatch')
        self.indexes[key] = rows
        return rows

    def read(self, ref):
        path, offset, rid = ref
        with path.open('rb') as f:
            f.seek(offset); h = f.read(12); length = struct.unpack('<Q',h[:8])[0]
            b = f.read(length); tail = f.read(4)
        if len(tail) != 4 or crc(h[:8]) != struct.unpack('<I',h[8:])[0] or crc(b) != struct.unpack('<I',tail)[0]:
            raise ValueError('TFRecord CRC failed: '+rid)
        raw = example(b)
        if raw['recording_id'].decode() != rid or raw['participant_id'].decode() != self.rows[rid]['participant_id']:
            raise ValueError('TFRecord ID mismatch')
        dtype = '<f2' if self.meta['signal_dtype']=='float16' else '<f4'
        out = {k:np.frombuffer(raw[k],dtype=dtype).astype(np.float32) for k in ('respi','ecg','spo2')}
        for k, length in [('respi',1228800),('ecg',1228800),('spo2',36000)]:
            if out[k].shape != (length,) or not np.isfinite(out[k]).all(): raise ValueError('Invalid signal: '+rid)
        rawmask=np.frombuffer(raw['is_data'],dtype=np.uint8)
        if not np.isin(rawmask,[0,1]).all():raise ValueError('Nonbinary cache mask')
        out.update(stage=np.frombuffer(raw['sleep_stage'],dtype=np.uint8).astype(np.int64),
                   mask=rawmask.astype(bool), ahi=float(raw['ahi']),rid=rid,pid=raw['participant_id'].decode())
        if out['stage'].size != 1200 or out['mask'].size != 1200 or not out['mask'].any() or np.any(out['stage'][out['mask']]>=4) or not np.isfinite(out['ahi']) or out['ahi']<0:
            raise ValueError('Invalid target: '+rid)
        return out

    def records(self, cohorts, part, seed=None, limit=0):
        refs = list(self.index(cohorts,part))
        if seed is not None:
            if part != 'train': raise ValueError('Only train may be shuffled')
            np.random.default_rng(seed).shuffle(refs)
        for ref in refs[:limit or None]: yield self.read(ref)

    def stage5(self, record):
        from scipy.io import loadmat
        row=self.rows[record['rid']]; path=self.manifest.parent/row['mat_relative_path']
        path.resolve().relative_to(self.manifest.parent.resolve())
        if sha(path) != row['mat_sha256']: raise ValueError('MAT changed after label audit')
        raw=np.asarray(loadmat(path,variable_names=['sleep_label'])['sleep_label']).ravel()
        if len(raw)!=1200 or not np.isfinite(raw).all() or not np.equal(raw,np.round(raw)).all(): raise ValueError('Invalid raw stages')
        y=np.full(1200,-1,dtype=np.int64)
        for a,b in {0:0,1:1,2:2,3:3,4:3,5:4}.items(): y[raw==a]=b
        y[~record['mask']]=-1
        merged=np.array([0,1,1,2,3])[np.maximum(y,0)]
        if np.any(y[record['mask']]<0) or not np.array_equal(merged[record['mask']],record['stage'][record['mask']]):
            raise ValueError('Five/four class alignment failed')
        return y

def signal_mask(r, points):
    return np.repeat(r['mask'],points)

def zscore(x, mask):
    x=np.asarray(x,dtype=np.float32); mask=np.asarray(mask,bool)
    if not mask.any(): raise ValueError('No observed samples')
    return np.where(mask,(x-x[mask].mean())/max(float(x[mask].std()),1e-6),0).astype(np.float32)

def bcg_features(r):
    """BCGNet data preparation is deferred at the user's explicit request.

    The processed ECG is filtered and is unsuitable for reconstructing native
    R peaks. Do not fabricate IHR, use filtered ECG as IHR, or silently drop it.
    """
    raise ValueError('BCGNET_INPUT_DEFERRED: verified IHR from ORIGINAL ECG is missing. R-peak extraction from filtered TrainingCache ECG is disabled by user instruction; no real-data BCGNet training is enabled.')
