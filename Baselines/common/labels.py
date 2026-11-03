"""Reviewed NSRR event sidecars; legacy MAT event codes are NEVER read."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .data import Cache, sha

class EventLabels:
    def __init__(self, root, cache):
        self.root=Path(root); path=self.root/'index.json'
        if not path.is_file():
            raise ValueError('MISSING_REVIEWED_BINARY_EVENTS: supply --event-labels DIR; see LABEL_REQUIREMENTS.md. MAT respi_event_label is rejected, not converted.')
        self.doc=json.loads(path.read_text())
        if self.doc.get('format')!='nsrr-reviewed-binary-events-v1' or self.doc.get('manifest_sha256')!=cache.meta['manifest_sha256']:
            raise ValueError('Event sidecar format/manifest mismatch')
        if self.doc.get('reviewed') is not True or not self.doc.get('scoring_rule'): raise ValueError('Event scoring rule has not been reviewed')

    def require(self, cache, cohorts, parts):
        ids=[r['recording_id'] for c in cohorts for p in parts for r in cache.doc[c][p]]
        missing=[rid for rid in ids if rid not in self.doc['records']]
        if missing: raise ValueError('MISSING_REVIEWED_BINARY_EVENTS: %d recordings; first: %s'%(len(missing),', '.join(missing[:3])))
        # Validate every selected label file before an optimizer update.
        for rid in ids: self.read_id(rid,cache)

    def read_id(self,rid,cache):
        item=self.doc['records'][rid]; p=(self.root/item['file']).resolve();p.relative_to(self.root.resolve())
        if sha(p)!=item['sha256']: raise ValueError('Event labels changed: '+rid)
        if item.get('reviewed') is not True or item.get('time_alignment_reviewed') is not True or item.get('cache_ahi_checked') is not True: raise ValueError('Unreviewed event labels / cache AHI not checked')
        with np.load(p,allow_pickle=False) as f:
            if not np.isin(f['event'],[0,1]).all() or not np.isin(f['valid'],[0,1]).all():raise ValueError('Nonbinary event labels/mask')
            y=f['event'].astype(np.int64); valid=f['valid'].astype(bool)
            if str(f['recording_id'])!=rid: raise ValueError('Event ID mismatch')
        if y.shape!=(36000,) or valid.shape!=(36000,) or not np.isin(y,[0,1]).all() or not valid.any(): raise ValueError('Expected 36000 binary events and coverage mask')
        if not np.isfinite([item['recomputed_full_night_ahi'],item['reference_ahi']]).all() or abs(item['recomputed_full_night_ahi']-item['reference_ahi'])>.1: raise ValueError('Events/AHI inconsistent')
        return y,valid

    def read(self,r,cache):
        y,valid=self.read_id(r['rid'],cache)
        if abs(self.doc['records'][r['rid']]['reference_ahi']-r['ahi'])>1e-3: raise ValueError('Sidecar/cache AHI mismatch')
        return y,valid & np.repeat(r['mask'],30)

def import_review(source,destination,cache):
    d=json.loads(Path(source).read_text())
    if d.get('format')!='nsrr-event-review-v1' or d.get('reviewed') is not True or not d.get('scoring_rule'):
        raise ValueError('Provide an explicitly reviewed NSRR event table and scoring rule')
    if d.get('manifest_sha256')!=cache.meta['manifest_sha256']: raise ValueError('Review manifest mismatch')
    if not d.get('records'):raise ValueError('No reviewed records')
    refs={rid:(p,offset,rid) for c in cache.meta['datasets'] for part in ('train','val','test') for p,offset,rid in cache.index([c],part)}
    dest=Path(destination); dest.mkdir(parents=True,exist_ok=False)
    index=dict(format='nsrr-reviewed-binary-events-v1',reviewed=True,scoring_rule=d['scoring_rule'],manifest_sha256=d['manifest_sha256'],source_review_sha256=sha(source),records={})
    try:
        for row in d['records']:
            rid=row['recording_id']
            if rid not in cache.rows or rid in index['records']: raise ValueError('Unknown/repeated recording ID')
            if abs(cache.read(refs[rid])['ahi']-float(row['reference_ahi']))>1e-3:raise ValueError('Review AHI differs from immutable cache: '+rid)
            if row.get('time_alignment_reviewed') is not True or row.get('ahi_rule_reviewed') is not True: raise ValueError('Review alignment and AHI rule for '+rid)
            source_sha=row['source_sha256']
            if len(source_sha)!=64 or any(c not in '0123456789abcdef' for c in source_sha): raise ValueError('Source SHA256 required')
            events=row['events']; ids=set(); count=0
            y=np.zeros(36000,dtype=np.uint8); valid=np.zeros(36000,dtype=np.uint8)
            origin=float(row['cache_zero_seconds']); duration=float(row['source_duration_seconds']); tst=float(row['full_night_tst_seconds'])
            if not 0<tst<=duration or origin<0 or not np.isfinite([origin,duration,tst,row['reference_ahi']]).all(): raise ValueError('Invalid time metadata')
            for start,end in row['known_coverage_intervals_seconds']:
                if not 0<=start<end<=duration: raise ValueError('Invalid coverage interval')
                # Entire one-second bin must have known annotation coverage.
                a=max(0,math.ceil(start-origin)); b=min(36000,math.floor(end-origin))
                if b>a:valid[a:b]=1
            for e in events:
                if e['event_id'] in ids or not e.get('type') or not e.get('qualification_basis') or type(e['qualifies_for_reference_ahi']) is not bool:
                    raise ValueError('Distinct ID/type/qualification decision and basis required')
                ids.add(e['event_id']); start=float(e['onset_seconds']); length=float(e['duration_seconds'])
                if not np.isfinite([start,length]).all() or start<0 or length<=0 or start+length>duration+.01: raise ValueError('Invalid event interval')
                if e['qualifies_for_reference_ahi']:
                    count+=1; a=max(0,math.floor(start-origin)); b=min(36000,math.ceil(start+length-origin))
                    if b>a:y[a:b]=1
            ahi=count*3600/tst
            if abs(ahi-float(row['reference_ahi']))>.1: raise ValueError('Full-night event count/TST disagrees with AHI: '+rid)
            if not valid.any(): raise ValueError('No label coverage')
            path=dest/(rid+'.npz');np.savez_compressed(path,recording_id=rid,event=y,valid=valid)
            index['records'][rid]=dict(file=path.name,sha256=sha(path),reviewed=True,time_alignment_reviewed=True,cache_ahi_checked=True,source_sha256=source_sha,cache_zero_seconds=origin,reference_ahi=float(row['reference_ahi']),recomputed_full_night_ahi=ahi,full_night_event_count=count,full_night_tst_seconds=tst)
        (dest/'source_review.json').write_text(json.dumps(d,indent=2))
        (dest/'index.json').write_text(json.dumps(index,indent=2))
    except Exception:
        # An interrupted/rejected import has no accepted index and cannot be trained.
        (dest/'IMPORT_FAILED.txt').write_text('Review/import failed. Fix source and use a new output directory.\n')
        raise
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--cache');p.add_argument('--manifest');a=p.parse_args()
    kw={k:v for k,v in [('root',a.cache),('manifest',a.manifest)] if v}
    result=import_review(a.input,a.output,Cache(**kw));print('Accepted reviewed records:',len(result['records']))
