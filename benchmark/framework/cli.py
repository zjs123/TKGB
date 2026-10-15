import argparse
import json
from pathlib import Path
from .tasks import evaluate_tail_file,evaluate_activation_file,evaluate_generation_file


def main():
    p=argparse.ArgumentParser(description='Model-independent evaluation of externally supplied predictions')
    p.add_argument('task',choices=['tail','activation','generation'])
    p.add_argument('--dataset',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--split',choices=['valid','test'],default='test');p.add_argument('--output',required=True)
    p.add_argument('--scope',choices=['all','transductive','inductive'],default='all')
    p.add_argument('--horizon',type=int);p.add_argument('--granularity',choices=['day','month','year'],default='day')
    p.add_argument('--threshold',type=float,default=.5)
    args=p.parse_args()
    with Path(args.predictions).open(encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
    if args.task=='tail':result=evaluate_tail_file(args.dataset,rows,args.split,args.scope,args.horizon,args.granularity)
    elif args.task=='activation':result=evaluate_activation_file(args.dataset,rows,args.split,args.threshold)
    else:result=evaluate_generation_file(args.dataset,rows,args.split)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['by_query','ranks','groups']},ensure_ascii=False))


if __name__=='__main__':main()
