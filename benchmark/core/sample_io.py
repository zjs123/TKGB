import csv
from pathlib import Path
from .schema import TimestampFact


def read_eval_sample(path,split):
    path=Path(path)
    if not path.exists():return []
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [TimestampFact(split=split,s_id=int(r['s_id']),r_id=int(r['r_id']),o_id=int(r['o_id']),t=r['t'],
                source_name=r.get('source_name',''),relation_label=r.get('relation_label',''),target_name=r.get('target_name',''),
                timestamp_source=r.get('timestamp_source','shared_eval_sample'),fact_hash=r.get('hash',''),metadata=dict(r)) for r in csv.DictReader(f)]


def write_eval_sample(path,facts):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['s_id','r_id','o_id','t','source_name','relation_label','target_name','timestamp_source','hash']);w.writeheader()
        for x in facts:w.writerow(dict(s_id=x.s_id,r_id=x.r_id,o_id=x.o_id,t=x.t,source_name=x.source_name,relation_label=x.relation_label,target_name=x.target_name,timestamp_source=x.timestamp_source,hash=x.fact_hash))
