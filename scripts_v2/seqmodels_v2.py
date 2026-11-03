


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import f1_score

import run_pilot as RP
from zoo import build_seq_model, n_params, pool_table


def train_seq_v2(name, Xtr, Mtr, ytr, Xva, Mva, yva, Xte, Mte, yte, seed, dev,
                 max_epochs=20, batch=256, patience_max=5,
                 lr=1e-3, warmup_frac=0.0, class_weight="balanced",
                 Xz=None, Mz=None, zout=None):


    RP.seed_everything(seed)
    model = build_seq_model(name, Xtr.shape[2], Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=2).astype(np.float64)

    if class_weight == "none":
        lossf = nn.CrossEntropyLoss()
    else:
        w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32, device=dev)
        lossf = nn.CrossEntropyLoss(weight=w)

    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
    Xtr_t, Mtr_t, ytr_t = t(Xtr), t(Mtr, torch.bool), t(ytr, torch.long)
    Xva_t, Mva_t = t(Xva), t(Mva, torch.bool)
    Xte_t, Mte_t = t(Xte), t(Mte, torch.bool)

    bs = batch
    n = len(ytr)

    _spe = max(1, int(np.ceil(n / bs)))
    _total = max(1, max_epochs * _spe)
    _warm = max(1, int(_total * warmup_frac)) if warmup_frac > 0 else 0

    def _lr_lambda(step):
        if _warm and step < _warm:
            return float(step + 1) / float(_warm)
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = np.random.default_rng(seed * 1000 + ep).permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t[idx], Mtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t, Mva_t), -1)[:, 1].cpu().numpy()
        f1 = f1_score(yva, (pv >= 0.5).astype(int), average="macro", zero_division=0)
        if f1 > best_f1 + 1e-6:
            best_f1, best_state, patience = float(f1), {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= patience_max:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():


        p_val = torch.softmax(model(Xva_t, Mva_t), -1)[:, 1].cpu().numpy()
        p = torch.softmax(model(Xte_t, Mte_t), -1)[:, 1].cpu().numpy()

        if Xz is not None and zout is not None:
            Xz_t = torch.from_numpy(np.ascontiguousarray(Xz)).to(dev)
            Mz_t = torch.from_numpy(np.ascontiguousarray(Mz)).to(dev)
            zout["p"] = torch.softmax(model(Xz_t, Mz_t), -1)[:, 1].cpu().numpy()
    return p, p_val, n_params(model), ep + 1


def train_rf_v2(Xtr, Mtr, ytr, Xva, Mva, Xte, Mte, seed,
                class_weight="balanced_subsample", Xz=None, Mz=None, zout=None):


    Ftr = pool_table(Xtr, Mtr)
    Fva = pool_table(Xva, Mva)
    Fte = pool_table(Xte, Mte)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=16,
                                 class_weight=class_weight, random_state=seed)
    clf.fit(Ftr, ytr)
    p_val = clf.predict_proba(Fva)[:, 1]
    p = clf.predict_proba(Fte)[:, 1]
    if Xz is not None and zout is not None:
        zout["p"] = clf.predict_proba(pool_table(Xz, Mz))[:, 1]
    return p, p_val, 0, 0


def train_hgb_v2(Xtr, Mtr, ytr, Xva, Mva, Xte, Mte, seed, class_weight="balanced",
                 Xz=None, Mz=None, zout=None):

    Ftr = pool_table(Xtr, Mtr)
    Fva = pool_table(Xva, Mva)
    Fte = pool_table(Xte, Mte)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.15,
        class_weight=class_weight, random_state=seed)
    clf.fit(Ftr, ytr)
    p_val = clf.predict_proba(Fva)[:, 1]
    p = clf.predict_proba(Fte)[:, 1]
    if Xz is not None and zout is not None:
        zout["p"] = clf.predict_proba(pool_table(Xz, Mz))[:, 1]
    return p, p_val, 0, int(clf.n_iter_)
