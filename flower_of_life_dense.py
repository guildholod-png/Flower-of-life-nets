# -*- coding: utf-8 -*-
"""
Эксперимент: сколько пересечений считать связями?

При радиусе окружности = шагу решётки (пропорция Цветка жизни)
окружность пересекается не только с 6 ближайшими, но и со вторым
поясом (расстояние sqrt(3) < 2 радиусов). Сравниваем:

  NB=6  — рёбра только «через центр соседа» (расстояние 1)
  NB=12 — рёбра по всем пересечениям (расстояния 1 и sqrt(3))

и число шагов распространения K=4 / K=8. Всего 4 конфигурации,
данные и стартовые веса одинаковые (seed).

Запуск: NB=6 K=8 python3 ...; после всех четырёх: FIGURE=1 python3 ...
"""
import os
import numpy as np

rng = np.random.default_rng(42)
NB = int(os.environ.get("NB", 6))
K = int(os.environ.get("K", 8))

# ------------------------------------------------------- решётка
R_MAX = 6
axial = [(q, r) for q in range(-R_MAX, R_MAX + 1) for r in range(-R_MAX, R_MAX + 1)
         if max(abs(q), abs(r), abs(q + r)) <= R_MAX]
pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
N = len(axial)

def build_A(cutoff):
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    A = ((d > 1e-9) & (d < cutoff)).astype(float)
    return A, A / np.maximum(A.sum(1, keepdims=True), 1)

CUTOFF = {6: 1.1, 12: 1.9}[NB]     # 1.9: включает sqrt(3)~1.73, исключает 2.0
A, A_hat = build_A(CUTOFF)
E = int(A.sum() / 2)
print(f"NB={NB} K={K}: {N} узлов, {E} связей, "
      f"сообщений за прогон: {E * K}")

# ------------------------------------------------------- данные (общие)
def make_dataset(n_samples):
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < R_MAX - 2.5)[0]
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

# ------------------------------------------------------- модель
D = 8

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {
    "Win": init((1, D)), "bin": np.zeros(D),
    "Ws":  init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
    "Wo":  init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3),
}

def forward(X):
    cache = {"X": X, "H": []}
    h = np.tanh(X @ P["Win"] + P["bin"])
    cache["H"].append(h)
    for _ in range(K):
        m = np.matmul(A_hat, h)
        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
        cache["H"].append(h)
    amax = h.argmax(axis=1)
    g = np.concatenate([h.mean(axis=1),
                        np.take_along_axis(h, amax[:, None, :], 1)[:, 0, :]], 1)
    logits = g @ P["Wo"] + P["bo"]
    cache["g"], cache["amax"] = g, amax
    return logits, cache

def loss_and_grads(X, y):
    B = len(y)
    logits, cache = forward(X)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dlogits = p.copy()
    dlogits[np.arange(B), y] -= 1
    dlogits /= B
    G["Wo"] = cache["g"].T @ dlogits
    G["bo"] = dlogits.sum(0)
    dg = dlogits @ P["Wo"].T
    dh = dg[:, None, :D] / N * np.ones((B, N, 1))
    dmax = np.zeros((B, N, D))
    np.put_along_axis(dmax, cache["amax"][:, None, :], dg[:, None, D:], axis=1)
    dh = dh + dmax
    for t in range(K, 0, -1):
        h_out, h_in = cache["H"][t], cache["H"][t - 1]
        dpre = dh * (1 - h_out ** 2)
        m_in = np.matmul(A_hat, h_in)
        dpre2 = dpre.reshape(-1, D)
        G["Ws"] += h_in.reshape(-1, D).T @ dpre2
        G["Wn"] += m_in.reshape(-1, D).T @ dpre2
        G["b"]  += dpre2.sum(0)
        dh = dpre @ P["Ws"].T + np.matmul(A_hat.T, dpre @ P["Wn"].T)
    dpre = dh * (1 - cache["H"][0] ** 2)
    G["Win"] = cache["X"].reshape(-1, 1).T @ dpre.reshape(-1, D)
    G["bin"] = dpre.reshape(-1, D).sum(0)
    return loss, G

# ------------------------------------------------------- обучение
if not os.environ.get("FIGURE"):
    import time
    EPOCHS, BATCH, LR = 100, 64, 3e-3
    budget = float(os.environ.get("TIME_BUDGET", 0)) or None
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    t0, adam_t, acc_hist = time.time(), 0, []
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
        logits, _ = forward(Xte)
        ta = (logits.argmax(1) == yte).mean()
        acc_hist.append(ta)
        if (ep + 1) % 20 == 0:
            print(f"эпоха {ep+1:3d}  test acc {ta:.3f}")
        if budget and time.time() - t0 > budget:
            print(f"(остановка по бюджету на эпохе {ep+1})")
            break
    print(f"итог NB={NB} K={K}: {acc_hist[-1]:.3f}, "
          f"время {time.time()-t0:.0f}с")
    np.savez(f"hist_nb{NB}_k{K}.npz", acc=np.array(acc_hist),
             edges=E, msgs=E * K, wall=time.time() - t0)

# ------------------------------------------------------- картинка
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    configs = [(6, 4), (6, 8), (12, 4), (12, 8)]
    colors = {(6, 4): "lightsteelblue", (6, 8): "steelblue",
              (12, 4): "salmon", (12, 8): "crimson"}
    data = {c: np.load(f"hist_nb{c[0]}_k{c[1]}.npz") for c in configs}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for c in configs:
        d = data[c]
        ax.plot(np.arange(1, len(d["acc"]) + 1), d["acc"], color=colors[c],
                label=f"{c[0]} соседей, K={c[1]}")
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность на тесте")
    ax.set_title("Кривые обучения", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    xs = np.arange(4)
    finals = [data[c]["acc"][-10:].mean() for c in configs]
    msgs = [int(data[c]["msgs"]) for c in configs]
    bars = ax.bar(xs, finals, color=[colors[c] for c in configs])
    for x, f, m in zip(xs, finals, msgs):
        ax.text(x, f + 0.004, f"{f:.3f}\n{m} сообщ.", ha="center", fontsize=9)
    ax.set_xticks(xs, [f"{c[0]}нб\nK={c[1]}" for c in configs])
    ax.set_ylim(0.8, 1.02)
    ax.set_title("Итог (среднее за посл. 10 эпох) и цена прогона", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # схемы связности центрального узла
    center = int(np.argmin(np.linalg.norm(pos, axis=1)))
    for col, nb in enumerate([6, 12]):
        ax = axes[1, col]
        Anb, _ = build_A({6: 1.1, 12: 1.9}[nb])
        near = np.linalg.norm(pos - pos[center], axis=1) < 2.6
        for i in np.where(near)[0]:
            ax.add_patch(Circle(pos[i], 1.0, fill=False, lw=0.6,
                                color="0.75", zorder=1))
        for j in np.where(Anb[center] > 0)[0]:
            ax.plot(*zip(pos[center], pos[j]), color="crimson", lw=1.4, zorder=2)
        ax.scatter(pos[near, 0], pos[near, 1], c="0.3", s=25, zorder=3)
        ax.scatter(*pos[center], c="crimson", s=60, zorder=4)
        ax.set_title(f"{nb} соседей: "
                     + ("рёбра через центры" if nb == 6
                        else "рёбра по всем пересечениям"), fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("Радиус как правило связности: чем больше пересечений считаем "
                 "связями, тем плотнее граф и быстрее распространение",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_dense.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_dense.png")
