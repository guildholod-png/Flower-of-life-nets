# -*- coding: utf-8 -*-
"""
Много или мало 96.5%? Сравнение на ОДНИХ И ТЕХ ЖЕ данных:

  logreg — логистическая регрессия: узлы как мешок чисел, структуры нет
  mlp    — MLP 127->64->3: нелинейность есть, структуры нет
  gnn    — наша решётка (12 соседей, K=4): структура Цветка жизни
  rnd    — та же сеть, но связи случайно перемешаны: граф есть, геометрии нет

Последний — контроль: если rnd ~ gnn, геометрия ничего не даёт
и работает просто факт наличия графа.

Запуск: MODEL=logreg|mlp|gnn|rnd python3 ...; затем FIGURE=1 python3 ...
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
A = ((d2 > 1e-9) & (d2 < 1.9)).astype(float)          # 12 соседей

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

if MODEL == "rnd":                                     # перемешать геометрию
    perm = rng.permutation(N)
    A = A[np.ix_(perm, perm)]
A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)

# ------------------------------------------------------- модели
D, K = 8, 4

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

def softmax_ce(logits, y):
    B = len(y)
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
    dl = p.copy()
    dl[np.arange(B), y] -= 1
    return loss, dl / B, p

if MODEL == "logreg":
    P = {"W": init((N, 3)), "b": np.zeros(3)}
    def run(X, y=None):
        logits = X[:, :, 0] @ P["W"] + P["b"]
        if y is None:
            return logits
        loss, dl, _ = softmax_ce(logits, y)
        return loss, {"W": X[:, :, 0].T @ dl, "b": dl.sum(0)}

elif MODEL == "mlp":
    H = 64
    P = {"W1": init((N, H)), "b1": np.zeros(H),
         "W2": init((H, 3)), "b2": np.zeros(3)}
    def run(X, y=None):
        h = np.tanh(X[:, :, 0] @ P["W1"] + P["b1"])
        logits = h @ P["W2"] + P["b2"]
        if y is None:
            return logits
        loss, dl, _ = softmax_ce(logits, y)
        dh = (dl @ P["W2"].T) * (1 - h ** 2)
        return loss, {"W2": h.T @ dl, "b2": dl.sum(0),
                      "W1": X[:, :, 0].T @ dh, "b1": dh.sum(0)}

else:                                                  # gnn / rnd
    P = {"Win": init((1, D)), "bin": np.zeros(D),
         "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
         "Wo": init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3)}
    def run(X, y=None):
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
        logits = g @ P["Wo"] + P["bo"]
        if y is None:
            return logits
        B = len(y)
        loss, dl, _ = softmax_ce(logits, y)
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

N_PARAMS = sum(v.size for v in P.values())

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
            loss, G = run(Xtr[j], ytr[j])
            adam_t += 1
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
                P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
        ta = (run(Xte).argmax(1) == yte).mean()
        acc_hist.append(ta)
        if (ep + 1) % 25 == 0:
            print(f"эпоха {ep+1:3d}  test acc {ta:.3f}")
    print(f"итог {MODEL}: {np.mean(acc_hist[-10:]):.3f} "
          f"({N_PARAMS} параметров)")
    np.savez(f"base_{MODEL}.npz", acc=np.array(acc_hist), params=N_PARAMS)

# ------------------------------------------------------- картинка
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["logreg", "mlp", "rnd", "gnn"]
    labels = {"logreg": "логрег\n(без структуры)",
              "mlp": "MLP\n(без структуры)",
              "rnd": "случайный граф\n(контроль)",
              "gnn": "Цветок жизни\n(12 нб, K=4)"}
    colors = {"logreg": "0.7", "mlp": "0.5",
              "rnd": "goldenrod", "gnn": "crimson"}
    data = {m: np.load(f"base_{m}.npz") for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for m in models:
        ax.plot(np.arange(1, len(data[m]["acc"]) + 1), data[m]["acc"],
                color=colors[m], label=labels[m].replace("\n", " "))
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность на тесте")
    ax.set_title("Кривые обучения (данные одни и те же)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    xs = np.arange(len(models))
    finals = [data[m]["acc"][-10:].mean() for m in models]
    ax.bar(xs, finals, color=[colors[m] for m in models])
    for x, m, f in zip(xs, models, finals):
        ax.text(x, f + 0.005, f"{f:.3f}\n{int(data[m]['params'])} парам.",
                ha="center", fontsize=9)
    ax.set_xticks(xs, [labels[m] for m in models], fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Итог (среднее за посл. 10 эпох)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Стоит ли геометрия чего-нибудь? Та же задача, четыре модели",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_baselines.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_baselines.png")
