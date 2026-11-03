"""Reader for new raw-MAT 10h baseline inputs; never mixes with old cache axes."""
from pathlib import Path
import json
import numpy as np
from .data import sha

class NativeInputs:
    def __init__(self,root):
        self.root=Path(root);self.meta=json.loads((self.root/'index.json').read_text())
        if self.meta.get('format')!='nsrr-native-baseline-inputs-v1':raise ValueError('Unsupported native inputs')
        self.rows=self.meta['records']
    def records(self,cohort,partition):
        for rid,item in self.rows.items():
            if item['cohort']==cohort and item['partition']==partition:yield self.read(rid)
    def read(self,rid):
        item=self.rows[rid];path=(self.root/item['file']).resolve();path.relative_to(self.root.resolve())
        if sha(path)!=item['sha256']:raise ValueError('Changed prepared record '+rid)
        with np.load(path,allow_pickle=False) as f:r={k:f[k] for k in f.files}
        if str(r['recording_id'])!=rid:raise ValueError('ID mismatch')
        for k,n in [('abd_64hz',2304000),('spo2_1hz',36000),('heartbeat',144000),('breath',144000),('stage5',1200),('event',36000)]:
            if r[k].shape!=(n,) or not np.isfinite(r[k]).all():raise ValueError('Invalid '+k)
        r['metadata']=item;return r

def driven_signal(r,channel):
    if channel=='spo2':return np.repeat(r['spo2_1hz']/100.,64)
    if channel!='abd':raise ValueError('Native channels are abd/spo2')
    x=r['abd_64hz'];mask=np.repeat(r['stage_mask'].astype(bool),1920)
    return np.where(mask,(x-x[mask].mean())/max(float(x[mask].std()),1e-8),0).astype(np.float32)

def driven_targets(r):
    starts=np.arange(0,35971,15);stage_valid=np.repeat(r['stage_mask'].astype(bool),30)
    valid=stage_valid&r['event_valid'].astype(bool);sleep=np.repeat(r['stage5']>0,30)
    return dict(starts=starts,valid=np.array([valid[s:s+30].all() for s in starts]),sleep=np.array([sleep[s:s+30].any() for s in starts]),event=np.array([r['event'][s:s+30].any() and sleep[s:s+30].any() for s in starts]))

def bcg_stage(r):
    y=r['stage5'].copy();y[~r['bcg_stage_mask'].astype(bool)]=-1
    return dict(heartbeat=r['heartbeat'],breath=r['breath']),y

def bcg_events(r):
    valid=r['event_valid'].astype(bool)&r['heartbeat_valid'].reshape(36000,4).all(1)&np.repeat(r['stage_mask'].astype(bool),30)
    return dict(heartbeat=r['heartbeat'],breath=r['breath']),r['event'],valid

class NativeCache(NativeInputs):
    """Training adapter with immutable native dataset fingerprint and model eligibility."""
    def __init__(self,root,model='driven'):
        super().__init__(root)
        self.manifest=self.root/'splits.json'
        self.meta['native_index_sha256']=sha(self.root/'index.json')
        self.model=model
        self.excluded={rid:'SHARED_DRIVEN_BCGNET_EXCLUSION_NO_VALID_IHR' for rid,v in self.rows.items() if v['valid_bcg_stage_epochs']==0}
        self.rows={rid:v for rid,v in self.rows.items() if rid not in self.excluded}
    def index(self,cohorts,part):
        return sorted(rid for rid,v in self.rows.items() if v['cohort'] in cohorts and v['partition']==part)
    def read(self,rid):
        r=super().read(rid)
        r['rid']=str(r['recording_id']);r['pid']=str(r['participant_id']);r['ahi']=float(r['ahi'])
        r['stage']=np.asarray([0,1,1,2,3],np.int64)[r['stage5'].clip(0,4)]
        r['mask']=r['bcg_stage_mask' if self.model=='bcgnet' else 'stage_mask'].astype(bool)
        return r
    def records(self,cohorts,partition,seed=None,limit=0):
        refs=self.index(cohorts,partition)
        if seed is not None:np.random.default_rng(seed).shuffle(refs)
        for rid in refs[:limit or None]:yield self.read(rid)
