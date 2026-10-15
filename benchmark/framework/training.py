"""Callback-only training orchestration. No model, optimizer or tensor backend."""
import json
import math
import time
from pathlib import Path


def train_epochs(*,epochs,batches_for_epoch,train_step,output_dir,validate=None,save_checkpoint=None,patience=None):
    """The caller owns all model/backend state, data, seeding and checkpoints.

    train_step(batch) returns a finite scalar loss. validate(epoch) returns a
    finite scalar quality metric (higher is better). No test labels are passed
    into callbacks by this orchestrator. Train and validation times are separate.
    """
    if epochs<1:raise ValueError('epochs must be positive')
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    history=[];best=None;stale=0
    for epoch in range(epochs):
        start=time.perf_counter();losses=[]
        for batch in batches_for_epoch(epoch):
            loss=float(train_step(batch))
            if not math.isfinite(loss):raise ValueError('Nonfinite training loss')
            losses.append(loss)
        if not losses:raise ValueError('Epoch contains no batches')
        train_seconds=time.perf_counter()-start;start=time.perf_counter()
        metric=float(validate(epoch)) if validate else None
        if metric is not None and not math.isfinite(metric):raise ValueError('Nonfinite validation metric')
        validation_seconds=time.perf_counter()-start if validate else 0.
        row=dict(epoch=epoch+1,train_seconds=train_seconds,validation_seconds=validation_seconds,batches=len(losses),mean_batch_loss=sum(losses)/len(losses),validation_metric=metric)
        history.append(row)
        with (root/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        improved=metric is None or best is None or metric>best
        if improved:
            best=metric;stale=0
            if save_checkpoint:save_checkpoint(root/'best_checkpoint',epoch+1)
        else:stale+=1
        if patience is not None and metric is not None and stale>=patience:break
    return history
