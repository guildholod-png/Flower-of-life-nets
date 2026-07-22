# -*- coding: utf-8 -*-
"""
Перенос 2D -> 3D: правило, выученное на СРЕЗЕ, прикладываем к ОБЪЁМУ.

2D-решётка здесь — это буквально слой k=0 вашей 18-центровой структуры
(гекс-слой с шагом r/sqrt(3), рёбра = расстояние r, 6 соседей).
3D — полная структура (те же слои + связи вверх/вниз, 18 соседей).

Веса переносимы, потому что сообщение — среднее по соседям:
форма весов не зависит от их числа (6 в срезе, 18 в объёме).

Классы соответствуют друг другу:
  кольцо (2D)  -> сферическая оболочка (3D)
  пятно (2D)   -> сгусток (3D)
  полоса (2D)  -> плоский слой (3D)

Сравнение: 2D-веса на 3D (zero-shot) против 3D-весов на 3D (потолок)
и против 2D-весов на 2D (исходное качество).
"""
import numpy as np
from scipy import sparse

rng = np.random.default_rng(7)
RC, NOISE = 2.8, 0.15
s_, c_ = 1 / np.sqrt(3), np.sqrt(2 / 3)

def build(dim3):
    m = int(np.ceil(RC / s_)) + 1
    pts = []
    ks = range(-m, m + 1) if dim3 else [0]
    for a in range(-2 * m, 2 * m + 1):
        for b in range(-2 * m, 2 * m + 1):
            for k in ks:
                p = np.array([s_ * (a + 0.5 * b),
                              s_ * (np.sqrt(3) / 2) * b, c_ * k])
                if np.linalg.norm(p) <= RC:
                    pts.append(p)
    pos = np.array(pts)
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    A = ((d > 0.99) & (d < 1.01)).astype(float)
    A_hat = sparse.csr_matrix(A / np.maximum(A.sum(1, keepdims=True), 1))
    return pos, A_hat, int(A.sum(1).max())

pos2, A2, deg2 = build(False)
pos3, A3, deg3 = build(True)
print(f"срез: {len(pos2)} узлов (соседей до {deg2}); "
      f"объём: {len(pos3)} узлов (соседей до {deg3})")

def spmm(S, h):
    B, n, d = h.shape
    return S.dot(h.transpose(1, 0, 2).reshape(n, -1)) \
            .reshape(n, B, d).transpose(1, 0, 2)

def make_dataset(pos, n_samples):
    N = len(pos)
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < RC - 1.6)[0]
    dim3 = np.ptp(pos[:, 2]) > 0
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:                                  # кольцо / оболочка
            r0 = rng.uniform(1.0, 1.8)
            v = (np.abs(d - r0) < 0.45).astype(float)
        elif y[s] == 1:                                # пятно / сгусток
            v = np.exp(-d ** 2 / (2 * 0.8 ** 2))
        else:                                          # полоса / слой
            if dim3:
                nvec = rng.normal(size=3)
            else:
                ang = rng.uniform(0, np.pi)
                nvec = np.array([np.cos(ang), np.sin(ang), 0.0])
            nvec /= np.linalg.norm(nvec)
            v = (np.abs((pos - c) @ nvec) < 0.45).astype(float)
        X[s, :, 0] = v + rng.normal(0, NOISE, N)
    return X, y

Xtr, ytr = make_dataset(pos2, 900)                     # обучение НА СРЕЗЕ
Xte2, yte2 = make_dataset(pos2, 300)
Xte3, yte3 = make_dataset(pos3, 300)                   # экзамен В ОБЪЁМЕ

# ------------------------------------------------------- модель
D, K = 8, 5

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {"Win": init((1, D)), "bin": np.zeros(D),
     "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
     "Wo": init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3)}

def forward(X, A_hat):
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
    return g @ P["Wo"] + P["bo"], H_list, g, amax

def loss_and_grads(X, y, A_hat):
    B, N = len(y), X.shape[1]
    logits, H_list, g, amax = forward(X, A_hat)
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
        dh = dpre @ P["Ws"].T + spmm(sparse.csr_matrix(A_hat.T),
                                     dpre @ P["Wn"].T)
    dpre = dh * (1 - H_list[0] ** 2)
    G["Win"] = X.reshape(-1, 1).T @ dpre.reshape(-1, D)
    G["bin"] = dpre.reshape(-1, D).sum(0)
    return loss, G

# ------------------------------------------------------- обучение на срезе
EPOCHS, BATCH, LR = 100, 64, 3e-3
M = {k: np.zeros_like(v) for k, v in P.items()}
V = {k: np.zeros_like(v) for k, v in P.items()}
adam_t = 0
for ep in range(EPOCHS):
    idx = rng.permutation(len(ytr))
    for i in range(0, len(ytr), BATCH):
        j = idx[i:i + BATCH]
        loss, G = loss_and_grads(Xtr[j], ytr[j], A2)
        adam_t += 1
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
            P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
    if (ep + 1) % 25 == 0:
        logits, _, _, _ = forward(Xte2, A2)
        print(f"эпоха {ep+1:3d}  2D test acc {(logits.argmax(1)==yte2).mean():.3f}")

# ------------------------------------------------------- экзамены
def acc_per(logits, y):
    pred = logits.argmax(1)
    per = [((pred == y) & (y == c)).sum() / max((y == c).sum(), 1)
           for c in range(3)]
    return (pred == y).mean(), per

logits, _, _, _ = forward(Xte2, A2)
a2, p2 = acc_per(logits, yte2)
logits, _, _, _ = forward(Xte3, A3)
a23, p23 = acc_per(logits, yte3)
print(f"\n2D-веса на 2D:            {a2:.3f}  "
      f"(кольцо {p2[0]:.2f} пятно {p2[1]:.2f} полоса {p2[2]:.2f})")
print(f"2D-веса на 3D (zero-shot): {a23:.3f}  "
      f"(оболочка {p23[0]:.2f} сгусток {p23[1]:.2f} слой {p23[2]:.2f})")

# потолок: веса, обученные в 3D (из прошлого эксперимента)
import os
res3d = None
if os.path.exists("state18_hex18.npz"):
    st = np.load("state18_hex18.npz")
    P_bak = {k: P[k].copy() for k in P}
    for k in P:
        P[k] = st[f"P_{k}"]
    logits, _, _, _ = forward(Xte3, A3)
    a33, p33 = acc_per(logits, yte3)
    res3d = (a33, p33)
    print(f"3D-веса на 3D (потолок):   {a33:.3f}")
    for k in P:
        P[k] = P_bak[k]

np.savez("res_2d3d.npz", a2=a2, a23=a23, p23=p23,
         a33=res3d[0] if res3d else -1)

# ------------------------------------------------------- картинка
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
bars = [("2D-веса\nна 2D", a2, "steelblue"),
        ("2D-веса\nна 3D\n(zero-shot)", a23, "crimson")]
if res3d:
    bars.append(("3D-веса\nна 3D\n(потолок)", res3d[0], "0.55"))
xs = np.arange(len(bars))
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.55)
for x, b in zip(xs, bars):
    ax.text(x, b[1] + 0.01, f"{b[1]:.3f}", ha="center", fontsize=11)
ax.axhline(1 / 3, color="0.6", ls=":", lw=1)
ax.set_xticks(xs, [b[0] for b in bars], fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("точность")
ax.set_title("Правило со среза — в объём", fontsize=10)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
names = ["оболочка", "сгусток", "слой"]
xs = np.arange(3)
ax.bar(xs - 0.15, p23, width=0.3, color="crimson", label="2D-веса (zero-shot)")
if res3d:
    ax.bar(xs + 0.15, res3d[1], width=0.3, color="0.55", label="3D-веса")
ax.set_xticks(xs, names)
ax.set_ylim(0, 1.1)
ax.set_title("По классам на 3D-тесте", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Обучение на гекс-срезе (6 соседей) -> работа в объёме "
             "(18 соседей), без дообучения", fontsize=11)
fig.tight_layout()
fig.savefig("flower_of_life_2d3d.png", dpi=150, bbox_inches="tight")
print("Сохранено: flower_of_life_2d3d.png")
