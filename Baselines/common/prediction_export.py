"""Atomic, streaming export in the existing reviewer-table prediction schema."""
import json
from pathlib import Path
import numpy as np
from .protocol import write_json,metrics


def summary_record(pred,cohort):
    out=dict(recording_id=pred['recording_id'],participant_id=pred['participant_id'],dataset=cohort)
    if 'stage_prob' in pred:
        m=np.asarray(pred['mask'],bool);p=np.asarray(pred['stage_prob']);y=np.asarray(pred['stage_true'])
        if p.shape!=(len(m),4) or y.shape!=m.shape or not np.isfinite(p[m]).all():raise ValueError('Invalid four-class prediction')
        out.update(n_epochs_total=len(m),n_epochs_valid=int(m.sum()),epoch_indices=np.flatnonzero(m).tolist(),sleep_stage_true=y[m].astype(int).tolist(),sleep_stage_pred=p[m].argmax(-1).tolist(),sleep_stage_probabilities=p[m].tolist())
    if 'ahi_pred' in pred:
        value=float(pred['ahi_pred'])
        out.update(ahi_true=float(pred['ahi_true']),ahi_pred=value if np.isfinite(value) else None,ahi_pred_nonnegative=max(value,0.) if np.isfinite(value) else None,ahi_status=pred.get('ahi_status','VALID' if np.isfinite(value) else 'INVALID'))
    for key in ('sleep_source','ahi_method','predicted_event_count','predicted_sleep_hours','sleep_hours'):
        if key in pred:out[key]=pred[key]
    return out


def export_predictions(cache,cohorts,partition,run,predict,metadata=None):
    """Re-run inference from saved best weights; commit JSON only after all records."""
    run=Path(run);target=run/(partition+'_predictions.json');temp=target.with_suffix('.json.tmp')
    detail=run/('evaluation_'+partition);detail.mkdir(exist_ok=True)
    counts={};by_cohort={};seen=set()
    try:
        with temp.open('w') as f:
            f.write('{\n');first=True
            for cohort in cohorts:
                rows=[];counts[cohort]=0
                for r in cache.records([cohort],partition):
                    pred=predict(r);rid=pred['recording_id']
                    if rid in seen:raise ValueError('Duplicate prediction ID '+rid)
                    seen.add(rid);out=summary_record(pred,cohort)
                    if 'stage_prob' not in pred and 'ahi_pred' not in pred:raise ValueError('Prediction contains neither staging nor AHI: '+rid)
                    if not first:f.write(',\n')
                    json.dump(rid,f);f.write(':');json.dump(out,f,allow_nan=False,separators=(',',':'));first=False
                    tmp=detail/(rid+'.tmp.npz');np.savez_compressed(tmp,**pred);tmp.replace(detail/(rid+'.npz'))
                    rows.append({k:pred[k] for k in ('stage_prob','stage_true','mask','ahi_pred','ahi_true') if k in pred})
                    counts[cohort]+=1
                    if len(seen)%25==0:print(json.dumps(dict(phase='export_'+partition,recordings=len(seen))),flush=True)
                by_cohort[cohort]=metrics(rows)
                write_json(detail/(cohort+'_metrics.json'),by_cohort[cohort])
            if not seen:raise ValueError('Empty prediction export')
            f.write('\n}\n')
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    report=dict(status='COMPLETE',partition=partition,recordings=len(seen),counts=counts,metrics=by_cohort,manifest_sha256=cache.meta['manifest_sha256'],native_index_sha256=cache.meta.get('native_index_sha256'),checkpoint='best',test_used_for_fitting=False,**(metadata or {}))
    write_json(run/(partition+'_predictions.meta.json'),report)
    completion=run/'completion.json'
    if partition=='test' and completion.exists():
        state=json.loads(completion.read_text());state.update(test_evaluated=True,test_predictions=target.name,test_recordings=len(seen));write_json(completion,state)
    print('Exported:',target,'records:',len(seen),flush=True)
    return report
