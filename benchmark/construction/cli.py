"""Build one or all published dataset recipes; importing never runs a build."""
import argparse
import json
import os
from dataclasses import fields
from pathlib import Path
from .pipeline import PipelineConfig, run_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes',default='configs/construction_recipes.json')
    parser.add_argument('--dataset',required=True,help='A dataset name, or all')
    parser.add_argument('--dry-run',action='store_true',help='Validate and show recipes without network or LLM calls')
    parser.add_argument('--output-root',default='generated_datasets')
    args=parser.parse_args()
    recipes=json.loads(Path(args.recipes).read_text(encoding='utf-8'))
    names=list(recipes) if args.dataset=='all' else [args.dataset]
    allowed={f.name for f in fields(PipelineConfig)}
    for name in names:
        if name not in recipes:parser.error('Unknown dataset: '+name)
        config={k:v for k,v in recipes[name].items() if k in allowed}
        config.update(output_dir=str(Path(args.output_root)/name),openai_api_key_env='LLM_API_KEY',
                      openai_base_url=os.environ.get('LLM_BASE_URL',''),
                      openai_model=os.environ.get('LLM_MODEL',config.get('openai_model','')),
                      user_agent=os.environ.get('BENCHMARK_USER_AGENT','TemporalBenchmark/1.0'))
        cfg=PipelineConfig(**config)
        if args.dry_run:
            print(json.dumps(dict(dataset=name,domain=cfg.semantic_domain,granularity=cfg.dataset_time_granularity,
                                 history=[cfg.history_start_date,cfg.history_end_date],setting=cfg.setting),ensure_ascii=False))
        else:run_pipeline(cfg)


if __name__=='__main__':main()
