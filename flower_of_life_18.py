# -*- coding: utf-8 -*-
"""
3D: чья связность лучше — 12 (касания, FCC) или 18 (правило Цветка жизни)?

  LAT=fcc12 — FCC-упаковка, соседи = 12 касаний (как раньше)
  LAT=hex18 — решётка из правила «центр соседа на сфере»:
              гекс-слои с шагом r/sqrt(3), поставленные друг над другом
              на высоте r*sqrt(2/3); соседи = 18 узлов на расстоянии r
              (6 в плоскости + 6 сверху + 6 снизу)

Одинаковые: физический объём, узоры (в абсолютных единицах),
203 параметра, K шагов. Разные: плотность узлов (hex18 в ~3 раза
плотнее — это свойство самой конструкции) и связность.

Запуск: LAT=... TIME_BUDGET=35 [CONTINUE=1] python3 ...; потом FIGURE=1.
"""
import os
import numpy as np
from scipy import sparse

rng = np.random.default_rng(7)
LAT = os.environ.get("LAT", "hex18")
RC = 2.8
NOISE = float(os.environ.get("NOISE", 0.15))
SUF = "" if NOISE == 0.15 else f"_n{NOISE}"

def build(lat):
    pts = []
    if lat == "fcc12":
        m = int(np.ceil(RC * np.sqrt(2))) + 1
        for i in range(-m, m + 1):
            for j in range(-m, m + 1):
                for k in range(-m, m + 1):
                    if (i + j + k) % 2 == 0:
                        p = np.array([i, j, k]) / np.sqrt(2)
                        if np.linalg.norm(p) <= RC:
                            pts.append(p)
    else:
        s, c = 1 / np.sqrt(3), np.sqrt(2 / 3)
        m = int(np.ceil(RC / s)) + 1
        for a in range(-2 * m, 2 * m + 1):
            for b in range(-2 * m, 2 * m + 1):
                for k in range(-m, m + 1):
                    p = np.array([s * (a + 0.5 * b), s * (np.sqrt(3) / 2) * b,
                                  c * k])
                    if np.linalg.norm(p) <= RC:
                        pts.append(p)
    pos = np.array(pts)
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    A = ((d > 0.99) & (d < 1.01)).astype(float)   # рёбра = расстояние ровно r
    return pos, A

pos, A = build(LAT)
N = len(pos)
deg = A.sum(1)
A_hat = sparse.csr_matrix(A / np.maximum(deg[:, None], 1))
A_hatT = sparse.csr_matrix(A_hat.T)
print(f"{LAT}: {N} узлов, {int(A.sum()/2)} связей, "
      f"макс. соседей {int(deg.max())}")

def spmm(S, h):                                    # (N,N)@(B,N,D) через sparse
    B, n, d = h.shape
    return S.dot(h.transpose(1, 0, 2).reshape(n, -1)) \
            .reshape(n, B, d).transpose(1, 0, 2)

# ------------------------------------------------------- данные
def make_dataset(n_samples):
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < RC - 1.6)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:                              # оболочка
            r0 = rng.uniform(1.0, 1.8)
            v = (np.abs(d - r0) < 0.45).astype(float)
        elif y[s] == 1:                            # сгусток
            v = np.exp(-d ** 2 / (2 * 0.8 ** 2))
        else:                                      # слой
            nvec = rng.normal(size=3)
            nvec /= np.linalg.norm(nvec)
            v = (np.abs((pos - c) @ nvec) < 0.45).astype(float)
        X[s, :, 0] = v + rng.normal(0, NOISE, N)
    return X, y

Xtr, ytr = make_dataset(900)
Xte, yte = make_dataset(300)

# ------------------------------------------------------- модель
D, K = 8, 5

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {"Win": init((1, D)), "bin": np.zeros(D),
     "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
     "Wo": init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3)}

def forward(X):
    H_list = []
    h = np.tanh(X @ P["Win"] + P["bin"])
    H_list.append(h)
    for _ in range(K):
        m = spmm(A_hat, h)
        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
        H_list.append(h)
    amax = h.argmax(axis=1)
    g = np.concatenate([h.mean(axis=1),
                        np.take_along_axis(h, amax[:, None, :], 1)[:, 0, :]], 1)
    return g @ P["Wo"] + P["bo"], H_list, g, amax

def loss_and_grads(X, y):
    B = len(y)
    logits, H_list, g, amax = forward(X)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dl = p.copy()
    dl[np.arange(B), y] -= 1
    dl /= B
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

# ------------------------------------------------------- обучение
if not os.environ.get("FIGURE"):
    import time
    EPOCHS, BATCH, LR = 80, 64, 3e-3
    state_f = f"state18_{LAT}{SUF}.npz"
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    adam_t, acc_hist, ep0 = 0, [], 0
    if os.environ.get("CONTINUE") and os.path.exists(state_f):
        st = np.load(state_f, allow_pickle=True)
        for k in P:
            P[k] = st[f"P_{k}"]; M[k] = st[f"M_{k}"]; V[k] = st[f"V_{k}"]
        adam_t = int(st["adam_t"]); acc_hist = list(st["acc"])
        ep0 = len(acc_hist)
        print(f"(продолжаю с эпохи {ep0+1})")
    t0, budget = time.time(), float(os.environ.get("TIME_BUDGET", 0)) or None
    for ep in range(ep0, EPOCHS):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(ytr), BATCH):
            j = idx[i:i + BATCH]
            loss, G = loss_and_grads(Xtr[j], ytr[j])
            adam_t += 1
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
                P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
        logits, _, _, _ = forward(Xte)
        acc_hist.append((logits.argmax(1) == yte).mean())
        if (ep + 1) % 10 == 0:
            print(f"эпоха {ep+1:3d}  test acc {acc_hist[-1]:.3f}")
        if budget and time.time() - t0 > budget:
            print(f"(пауза по бюджету на эпохе {ep+1})")
            break
    save = {"adam_t": adam_t, "acc": np.array(acc_hist)}
    for k in P:
        save[f"P_{k}"] = P[k]; save[f"M_{k}"] = M[k]; save[f"V_{k}"] = V[k]
    np.savez(state_f, **save)
    if len(acc_hist) >= EPOCHS:
        logits, _, _, _ = forward(Xte)
        pred = logits.argmax(1)
        per = [((pred == yte) & (yte == c)).sum() / max((yte == c).sum(), 1)
               for c in range(3)]
        print(f"итог {LAT}: {np.mean(acc_hist[-10:]):.3f} "
              f"(оболочка {per[0]:.2f} сгусток {per[1]:.2f} слой {per[2]:.2f})")
        np.savez(f"hist18_{LAT}{SUF}.npz", acc=np.array(acc_hist), N=N,
                 E=int(A.sum() / 2), per=per)

# ------------------------------------------------------- картинка
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d12 = np.load("hist18_fcc12_n0.4.npz")
    d18 = np.load("hist18_hex18_n0.4.npz")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for d, lab, col in [(d12, "12 соседей (FCC, касания)", "steelblue"),
                        (d18, "18 соседей (Цветок жизни)", "crimson")]:
        ax.plot(np.arange(1, len(d["acc"]) + 1), d["acc"], color=col,
                label=f"{lab}: {int(d['N'])} узлов")
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность на тесте")
    ax.set_title("Один объём, одни узоры, 203 параметра, K=5, шум 0.4", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    finals = [d12["acc"][-10:].mean(), d18["acc"][-10:].mean()]
    ax.bar([0, 1], finals, color=["steelblue", "crimson"], width=0.55)
    for x, f, d in zip([0, 1], finals, [d12, d18]):
        ax.text(x, f + 0.004, f"{f:.3f}", ha="center", fontsize=11)
    ax.set_xticks([0, 1], ["12 нб\n(касания)", "18 нб\n(через центр)"])
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Итог (среднее за посл. 10 эпох)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # вид сверху на окрестность
    ax = axes[2]
    ang6 = np.arange(6) * np.pi / 3
    ang_h = ang6 + np.pi / 6
    r_s = 1 / np.sqrt(3)
    for dx, dy, kind in [(np.cos(ang6), np.sin(ang6), "plane")]:
        pass
    ax.scatter(np.cos(ang6), np.sin(ang6), s=90, c="0.35", zorder=3,
               label="6 в плоскости (оба)")
    ax.scatter(r_s * np.cos(ang_h), r_s * np.sin(ang_h), s=70,
               facecolors="crimson", edgecolors="crimson", zorder=4,
               label="верхние: 18нб — все 6")
    ax.scatter(r_s * np.cos(ang_h[::2]), r_s * np.sin(ang_h[::2]), s=200,
               facecolors="none", edgecolors="steelblue", lw=2, zorder=5,
               label="верхние: 12нб — только 3")
    ax.scatter([0], [0], s=120, c="black", marker="*", zorder=6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Вид сверху: куда кладутся соседи\nследующего слоя",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("18-связность из AutoCAD-модели против 12-связности упаковки",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_18.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_18.png")
