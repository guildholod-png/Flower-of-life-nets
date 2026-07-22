# -*- coding: utf-8 -*-
"""
Фальсификация: особенна ли гексагональная решётка Цветка жизни,
или работает любая регулярная сетка?

Квадратная решётка 11x11 (121 узел ~ 127 у гекса), те же узоры,
те же 203 параметра, K=4:

  LAT=sq4 — 4 соседа (фон Нейман)
  LAT=sq8 — 8 соседей (Мур, с диагоналями)

Гекс-результаты (6нб/12нб, K=4) берутся из hist_nb*_k4.npz.
Запуск: LAT=sq4 ...; LAT=sq8 ...; затем FIGURE=1 ...
"""
import os
import numpy as np

rng = np.random.default_rng(42)
LAT = os.environ.get("LAT", "sq4")

# ------------------------------------------------------- квадратная решётка
S = 5
pos = np.array([(i, j) for i in range(-S, S + 1) for j in range(-S, S + 1)],
               dtype=float)
N = len(pos)
d2 = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
CUTOFF = {"sq4": 1.1, "sq8": 1.5}[LAT]
A = ((d2 > 1e-9) & (d2 < CUTOFF)).astype(float)
A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)
print(f"{LAT}: {N} узлов, {int(A.sum()/2)} связей")

# ------------------------------------------------------- данные (те же узоры)
def make_dataset(n_samples):
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < 3.5)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:
            r0 = rng.uniform(1.6, 3.2)
            v = (np.abs(d - r0) < 0.55).astype(float)
        elif y[s] == 1:
            v = np.exp(-d ** 2 / (2 * 1.1 ** 2))
        else:
            ang = rng.uniform(0, np.pi)
            nvec = np.array([np.cos(ang), np.sin(ang)])
            v = (np.abs((pos - c) @ nvec) < 0.55).astype(float)
        X[s, :, 0] = v + rng.normal(0, 0.15, N)
    return X, y

Xtr, ytr = make_dataset(900)
Xte, yte = make_dataset(300)

# ------------------------------------------------------- модель (как у гекса)
D, K = 8, 4

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
        m = np.matmul(A_hat, h)
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
        m_in = np.matmul(A_hat, h_in)
        dpre2 = dpre.reshape(-1, D)
        G["Ws"] += h_in.reshape(-1, D).T @ dpre2
        G["Wn"] += m_in.reshape(-1, D).T @ dpre2
        G["b"] += dpre2.sum(0)
        dh = dpre @ P["Ws"].T + np.matmul(A_hat.T, dpre @ P["Wn"].T)
    dpre = dh * (1 - H_list[0] ** 2)
    G["Win"] = X.reshape(-1, 1).T @ dpre.reshape(-1, D)
    G["bin"] = dpre.reshape(-1, D).sum(0)
    return loss, G

# ------------------------------------------------------- обучение
if not os.environ.get("FIGURE"):
    EPOCHS, BATCH, LR = 100, 64, 3e-3
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    adam_t, acc_hist = 0, []
    for ep in range(EPOCHS):
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
        if (ep + 1) % 25 == 0:
            print(f"эпоха {ep+1:3d}  test acc {acc_hist[-1]:.3f}")
    print(f"итог {LAT}: {np.mean(acc_hist[-10:]):.3f}")
    np.savez(f"hist_{LAT}.npz", acc=np.array(acc_hist))

# ------------------------------------------------------- картинка
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [("hist_sq4.npz", "квадрат, 4 нб", "0.65"),
            ("hist_sq8.npz", "квадрат, 8 нб", "0.4"),
            ("hist_nb6_k4.npz", "гекс, 6 нб", "lightcoral"),
            ("hist_nb12_k4.npz", "гекс, 12 нб", "crimson")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    finals = []
    for fn, lab, col in runs:
        acc = np.load(fn)["acc"]
        finals.append(acc[-10:].mean())
        ax.plot(np.arange(1, len(acc) + 1), acc, color=col, label=lab)
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность на тесте")
    ax.set_title("K=4, 203 параметра у всех", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1]
    xs = np.arange(4)
    ax.bar(xs, finals, color=[r[2] for r in runs])
    for x, f in zip(xs, finals):
        ax.text(x, f + 0.003, f"{f:.3f}", ha="center", fontsize=10)
    ax.set_xticks(xs, [r[1].replace(", ", "\n") for r in runs], fontsize=9)
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Итог (среднее за посл. 10 эпох)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Гексагональная решётка против квадратной", fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_square.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_square.png")
