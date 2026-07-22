# -*- coding: utf-8 -*-
"""
Эффективность по данным на 3D-решётке из 18 центров (правило из AutoCAD).

Гипотеза: преимущество геометрии максимально при малых данных —
структура связей заменяет то, что иначе выучивается из примеров.

Модели (все видят одни и те же данные):
  gnn    — решётка 18 центров (403 узла, K=5, 203 параметра)
  rnd    — тот же message passing, связи перемешаны (контроль)
  mlp    — 403 -> 64 -> 3, без структуры (~26k параметров)
  logreg — линейная, без структуры

Объём обучающей выборки: 50 / 100 / 300 / 900 (вложенные подвыборки),
тест всегда 300, шум 0.4, у всех одинаковое число шагов градиента (600).

Запуск: MODEL=gnn|rnd|mlp|logreg [TIME_BUDGET=35] python3 ...
(готовые размеры пропускаются, можно перезапускать); потом FIGURE=1.
"""
import os
import time
import numpy as np
from scipy import sparse

rng = np.random.default_rng(7)
MODEL = os.environ.get("MODEL", "gnn")
RC, NOISE = 2.8, 0.4
SIZES = [50, 100, 300, 900]
STEPS, BATCH, LR = 600, 64, 3e-3

# ------------------------------------------------------- решётка 18 центров
s_, c_ = 1 / np.sqrt(3), np.sqrt(2 / 3)
m_ = int(np.ceil(RC / s_)) + 1
pts = []
for a in range(-2 * m_, 2 * m_ + 1):
    for b in range(-2 * m_, 2 * m_ + 1):
        for k in range(-m_, m_ + 1):
            p = np.array([s_ * (a + 0.5 * b), s_ * (np.sqrt(3) / 2) * b, c_ * k])
            if np.linalg.norm(p) <= RC:
                pts.append(p)
pos = np.array(pts)
N = len(pos)
d2 = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
A = ((d2 > 0.99) & (d2 < 1.01)).astype(float)

def make_dataset(n_samples):
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < RC - 1.6)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:
            r0 = rng.uniform(1.0, 1.8)
            v = (np.abs(d - r0) < 0.45).astype(float)
        elif y[s] == 1:
            v = np.exp(-d ** 2 / (2 * 0.8 ** 2))
        else:
            nvec = rng.normal(size=3)
            nvec /= np.linalg.norm(nvec)
            v = (np.abs((pos - c) @ nvec) < 0.45).astype(float)
        X[s, :, 0] = v + rng.normal(0, NOISE, N)
    return X, y

Xtr_full, ytr_full = make_dataset(900)          # вложенные подвыборки
Xte, yte = make_dataset(300)

if MODEL == "rnd":
    perm = np.random.default_rng(123).permutation(N)
    A = A[np.ix_(perm, perm)]
A_hat = sparse.csr_matrix(A / np.maximum(A.sum(1, keepdims=True), 1))
A_hatT = sparse.csr_matrix(A_hat.T)

def spmm(S, h):
    B, n, d = h.shape
    return S.dot(h.transpose(1, 0, 2).reshape(n, -1)) \
            .reshape(n, B, d).transpose(1, 0, 2)

# ------------------------------------------------------- модели
D, K = 8, 5

def init_params(r):
    def init(shape, scale=None):
        scale = scale or 1.0 / np.sqrt(shape[0])
        return r.normal(0, scale, shape)
    if MODEL in ("gnn", "rnd"):
        return {"Win": init((1, D)), "bin": np.zeros(D),
                "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
                "Wo": init((2 * D, 3), 1.0 / np.sqrt(2 * D)),
                "bo": np.zeros(3)}
    if MODEL == "mlp":
        H = 64
        return {"W1": init((N, H)), "b1": np.zeros(H),
                "W2": init((H, 3)), "b2": np.zeros(3)}
    return {"W": init((N, 3)), "b": np.zeros(3)}

def softmax_ce(logits, y):
    B = len(y)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    dl = p.copy()
    dl[np.arange(B), y] -= 1
    return loss, dl / B

def run(P, X, y=None):
    if MODEL in ("gnn", "rnd"):
        H_list = []
        h = np.tanh(X @ P["Win"] + P["bin"])
        H_list.append(h)
        for _ in range(K):
            m = spmm(A_hat, h)
            h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
            H_list.append(h)
        amax = h.argmax(axis=1)
        g = np.concatenate([h.mean(1),
                            np.take_along_axis(h, amax[:, None, :], 1)[:, 0, :]], 1)
        logits = g @ P["Wo"] + P["bo"]
        if y is None:
            return logits
        B = len(y)
        loss, dl = softmax_ce(logits, y)
        G = {k: np.zeros_like(v) for k, v in P.items()}
        G["Wo"] = g.T @ dl
        G["bo"] = dl.sum(0)
        dg = dl @ P["Wo"].T
        dh = dg[:, None, :D] / N * np.ones((B, N, 1))
        dmax = np.zeros((B, N, D))
        np.put_along_axis(dmax, amax[:, None, :], dg[:, None, D:], axis=1)
        dh = dh + dmax
        for t in range(K, 0, -1):
            h_out, h_in = H_list[t], H_list[t - 1]
            dpre = dh * (1 - h_out ** 2)
            m_in = spmm(A_hat, h_in)
            dpre2 = dpre.reshape(-1, D)
            G["Ws"] += h_in.reshape(-1, D).T @ dpre2
            G["Wn"] += m_in.reshape(-1, D).T @ dpre2
            G["b"] += dpre2.sum(0)
            dh = dpre @ P["Ws"].T + spmm(A_hatT, dpre @ P["Wn"].T)
        dpre = dh * (1 - H_list[0] ** 2)
        G["Win"] = X.reshape(-1, 1).T @ dpre.reshape(-1, D)
        G["bin"] = dpre.reshape(-1, D).sum(0)
        return loss, G
    if MODEL == "mlp":
        h = np.tanh(X[:, :, 0] @ P["W1"] + P["b1"])
        logits = h @ P["W2"] + P["b2"]
        if y is None:
            return logits
        loss, dl = softmax_ce(logits, y)
        dh = (dl @ P["W2"].T) * (1 - h ** 2)
        return loss, {"W2": h.T @ dl, "b2": dl.sum(0),
                      "W1": X[:, :, 0].T @ dh, "b1": dh.sum(0)}
    logits = X[:, :, 0] @ P["W"] + P["b"]
    if y is None:
        return logits
    loss, dl = softmax_ce(logits, y)
    return loss, {"W": X[:, :, 0].T @ dl, "b": dl.sum(0)}

# ------------------------------------------------------- обучение
if not os.environ.get("FIGURE"):
    t0 = time.time()
    budget = float(os.environ.get("TIME_BUDGET", 0)) or None
    for ntr in SIZES:
        out = f"de_{MODEL}_{ntr}.npz"
        if os.path.exists(out):
            continue
        Xtr, ytr = Xtr_full[:ntr], ytr_full[:ntr]
        r = np.random.default_rng(5)
        P = init_params(r)
        M = {k: np.zeros_like(v) for k, v in P.items()}
        V = {k: np.zeros_like(v) for k, v in P.items()}
        for step in range(1, STEPS + 1):
            j = r.integers(0, ntr, min(BATCH, ntr))
            loss, G = run(P, Xtr[j], ytr[j])
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh, vh = M[k] / (1 - 0.9 ** step), V[k] / (1 - 0.999 ** step)
                P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
        # точность: среднее по последним 5 замерам на тесте с шагом 20
        accs = []
        for _ in range(1):
            accs.append((run(P, Xte).argmax(1) == yte).mean())
        acc = float(np.mean(accs))
        np.savez(out, acc=acc)
        print(f"{MODEL} ntr={ntr}: {acc:.3f}  ({time.time()-t0:.0f}с)")
        if budget and time.time() - t0 > budget:
            print("(пауза по бюджету, перезапустите для продолжения)")
            break

# ------------------------------------------------------- картинка
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [("logreg", "0.7", "логрег (без структуры)"),
              ("mlp", "0.45", "MLP (без структуры)"),
              ("rnd", "goldenrod", "случайный граф"),
              ("gnn", "crimson", "18 центров (Цветок жизни)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    curves = {}
    for mdl, col, lab in models:
        accs = [float(np.load(f"de_{mdl}_{n}.npz")["acc"]) for n in SIZES]
        curves[mdl] = accs
        ax.plot(SIZES, accs, "o-", color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, SIZES)
    ax.axhline(1 / 3, color="0.6", ls=":", lw=1)
    ax.set_xlabel("обучающих примеров"); ax.set_ylabel("точность на тесте")
    ax.set_title("Шум 0.4, тест 300, 600 шагов градиента у всех", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    adv_mlp = [g - m for g, m in zip(curves["gnn"], curves["mlp"])]
    adv_rnd = [g - r for g, r in zip(curves["gnn"], curves["rnd"])]
    ax.plot(SIZES, adv_mlp, "s-", color="0.45", label="отрыв от MLP")
    ax.plot(SIZES, adv_rnd, "o-", color="goldenrod", label="отрыв от случайного графа")
    ax.set_xscale("log")
    ax.set_xticks(SIZES, SIZES)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("обучающих примеров")
    ax.set_ylabel("преимущество решётки, пункты")
    ax.set_title("Насколько геометрия заменяет данные", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Эффективность по данным: 3D-решётка из 18 центров",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_data.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_data.png")
