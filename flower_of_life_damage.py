# -*- coding: utf-8 -*-
"""
Живучесть: обученную решётку повреждаем СЛУЧАЙНО при работе
(без дообучения) и смотрим кривую деградации.

Два вида повреждений:
  узлы  — узел «умирает»: вход обнуляется, все его связи рвутся
  связи — случайные рёбра рвутся, узлы живы

Агрегация сообщений — среднее по ВЫЖИВШИМ соседям (перенормировка
степени), поэтому мёртвый сосед не тянет сигнал вниз, просто исчезает.

Сравнение: решётка Цветка жизни (12нб, K=4) против случайного графа
(те же 203 параметра, той же плотности). Обе обучаются без повреждений,
повреждаются только на инференсе.

Запуск: MODEL=gnn|rnd (обучение), потом SWEEP=1 (повреждения + картинка).
"""
import os
import numpy as np

rng = np.random.default_rng(42)
MODEL = os.environ.get("MODEL", "gnn")

# ------------------------------------------------------- решётка и данные
R_MAX = 6
axial = [(q, r) for q in range(-R_MAX, R_MAX + 1) for r in range(-R_MAX, R_MAX + 1)
         if max(abs(q), abs(r), abs(q + r)) <= R_MAX]
pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
N = len(axial)
d2 = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
A0 = ((d2 > 1e-9) & (d2 < 1.9)).astype(float)          # 12 соседей

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
D, K = 8, 4

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {"Win": init((1, D)), "bin": np.zeros(D),
     "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
     "Wo": init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3)}

def norm(A):
    return A / np.maximum(A.sum(1, keepdims=True), 1)

def forward(X, A_hat):
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

def loss_and_grads(X, y, A_hat):
    B = len(y)
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
if not os.environ.get("SWEEP"):
    if MODEL == "rnd":
        perm = rng.permutation(N)
        A = A0[np.ix_(perm, perm)]
    else:
        A = A0
    A_hat = norm(A)
    EPOCHS, BATCH, LR = 100, 64, 3e-3
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    adam_t = 0
    for ep in range(EPOCHS):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(ytr), BATCH):
            j = idx[i:i + BATCH]
            loss, G = loss_and_grads(Xtr[j], ytr[j], A_hat)
            adam_t += 1
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
                P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
        if (ep + 1) % 25 == 0:
            logits, _, _, _ = forward(Xte, A_hat)
            print(f"эпоха {ep+1:3d}  test acc {(logits.argmax(1)==yte).mean():.3f}")
    save = {k: P[k] for k in P}
    save["A"] = A
    np.savez(f"dmg_{MODEL}.npz", **save)
    print(f"сохранено dmg_{MODEL}.npz")

# ------------------------------------------------------- повреждения
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fracs = np.arange(0, 0.55, 0.05)
    DRAWS = 8
    results = {}
    for model in ["gnn", "rnd"]:
        st = np.load(f"dmg_{model}.npz")
        for k in P:
            P[k] = st[k]
        A = st["A"]
        edges = np.array(np.where(np.triu(A) > 0)).T
        for mode in ["nodes", "edges"]:
            mean_acc, std_acc = [], []
            for f in fracs:
                accs = []
                for d in range(DRAWS):
                    r2 = np.random.default_rng(1000 + d)
                    Ad = A.copy()
                    Xd = Xte.copy()
                    alive = np.arange(N)
                    if mode == "nodes":
                        kill = r2.choice(N, int(f * N), replace=False)
                        Ad[kill, :] = 0; Ad[:, kill] = 0
                        Xd[:, kill, :] = 0
                        alive = np.setdiff1d(alive, kill)
                    else:
                        kill = r2.choice(len(edges), int(f * len(edges)),
                                         replace=False)
                        for i, j in edges[kill]:
                            Ad[i, j] = 0; Ad[j, i] = 0
                    # пуллинг только по живым узлам (мёртвые датчики известны)
                    h = np.tanh(Xd @ P["Win"] + P["bin"])
                    Ah = norm(Ad)
                    for _ in range(K):
                        m = np.matmul(Ah, h)
                        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
                    hs = h[:, alive]
                    am = hs.argmax(axis=1)
                    g = np.concatenate(
                        [hs.mean(1),
                         np.take_along_axis(hs, am[:, None, :], 1)[:, 0, :]], 1)
                    logits = g @ P["Wo"] + P["bo"]
                    accs.append((logits.argmax(1) == yte).mean())
                mean_acc.append(np.mean(accs)); std_acc.append(np.std(accs))
            results[(model, mode)] = (np.array(mean_acc), np.array(std_acc))
            print(f"{model}/{mode}: 0%={mean_acc[0]:.3f}  "
                  f"25%={mean_acc[5]:.3f}  50%={mean_acc[10]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    titles = {"nodes": "Удаление УЗЛОВ (шары умирают)",
              "edges": "Удаление СВЯЗЕЙ (рёбра рвутся)"}
    for ax, mode in zip(axes[:2], ["nodes", "edges"]):
        for model, col, lab in [("gnn", "crimson", "Цветок жизни"),
                                ("rnd", "goldenrod", "случайный граф")]:
            m, s = results[(model, mode)]
            ax.plot(fracs * 100, m, "o-", color=col, label=lab, ms=4)
            ax.fill_between(fracs * 100, m - s, m + s, color=col, alpha=0.15)
        ax.axhline(1 / 3, color="0.6", ls=":", label="случайное угадывание")
        ax.set_xlabel("удалено, %"); ax.set_ylabel("точность")
        ax.set_ylim(0.25, 1.02)
        ax.set_title(titles[mode], fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # иллюстрация: решётка с 30% мёртвых узлов
    ax = axes[2]
    r2 = np.random.default_rng(1001)
    kill = r2.choice(N, int(0.3 * N), replace=False)
    alive = np.setdiff1d(np.arange(N), kill)
    Ad = A0.copy(); Ad[kill, :] = 0; Ad[:, kill] = 0
    ii, jj = np.where(np.triu(Ad) > 0)
    for i, j in zip(ii, jj):
        ax.plot(*zip(pos[i], pos[j]), color="0.8", lw=0.5, zorder=1)
    ax.scatter(pos[alive, 0], pos[alive, 1], c="crimson", s=28, zorder=2,
               label="живые")
    ax.scatter(pos[kill, 0], pos[kill, 1], c="0.85", s=28, zorder=2,
               marker="x", label="мёртвые (30%)")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Решётка с 30% мёртвых узлов:\nсвязность сохраняется",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Живучесть: повреждения на инференсе, без дообучения "
                 f"(среднее по {DRAWS} розыгрышам)", fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_damage.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_damage.png")
