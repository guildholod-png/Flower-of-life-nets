# -*- coding: utf-8 -*-
"""
Нейросеть на решётке «Цветок жизни».

Узлы = центры окружностей (гексагональная решётка, R колец).
Рёбра = пересечения соседних окружностей (6 соседей у каждого узла).
Модель = рекуррентный message passing с общими весами (аналог свёртки),
обучение = ручной backprop на numpy, оптимизатор Adam.

Задача: классификация узоров активации на решётке (кольцо / пятно / полоса).
"""
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- решётка
def build_lattice(R):
    """Гексагональная решётка из R колец. Возвращает координаты и соседей."""
    axial = [(q, r) for q in range(-R, R + 1) for r in range(-R, R + 1)
             if max(abs(q), abs(r), abs(q + r)) <= R]
    index = {a: i for i, a in enumerate(axial)}
    pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    n = len(axial)
    A = np.zeros((n, n))
    for (q, r), i in index.items():
        for dq, dr in dirs:
            j = index.get((q + dq, r + dr))
            if j is not None:
                A[i, j] = 1.0
    A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)   # среднее по соседям
    return pos, A, A_hat

R = 6
pos, A, A_hat = build_lattice(R)
N = len(pos)                      # 127 узлов
print(f"Решётка: {N} узлов, {int(A.sum() / 2)} связей (пересечений)")

# ---------------------------------------------------------------- данные
def make_dataset(n_samples):
    """3 класса узоров на решётке: 0=кольцо, 1=пятно, 2=полоса."""
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < R - 2.5)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:                                # кольцо
            r0 = rng.uniform(1.6, 3.2)
            v = (np.abs(d - r0) < 0.55).astype(float)
        elif y[s] == 1:                              # пятно
            v = np.exp(-d ** 2 / (2 * 1.1 ** 2))
        else:                                        # полоса
            ang = rng.uniform(0, np.pi)
            nvec = np.array([np.cos(ang), np.sin(ang)])
            v = (np.abs((pos - c) @ nvec) < 0.55).astype(float)
        v = v + rng.normal(0, 0.15, N)               # шум
        X[s, :, 0] = v
    return X, y

# ---------------------------------------------------------------- модель
D, K = 8, 8          # размерность признаков узла, число шагов распространения

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
    h = np.tanh(X @ P["Win"] + P["bin"])             # (B,N,D)
    cache["H"].append(h)
    for _ in range(K):                               # волна по решётке
        m = np.matmul(A_hat, h)                      # среднее по соседям
        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
        cache["H"].append(h)
    amax = h.argmax(axis=1)                          # (B,D) индексы max-узлов
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
    acc = (p.argmax(1) == y).mean()

    G = {k: np.zeros_like(v) for k, v in P.items()}
    dlogits = p.copy()
    dlogits[np.arange(B), y] -= 1
    dlogits /= B
    G["Wo"] = cache["g"].T @ dlogits
    G["bo"] = dlogits.sum(0)
    dg = dlogits @ P["Wo"].T                          # (B,2D)
    dh = dg[:, None, :D] / N * np.ones((B, N, 1))     # градиент mean-пуллинга
    dmax = np.zeros((B, N, D))                        # градиент max-пуллинга
    np.put_along_axis(dmax, cache["amax"][:, None, :],
                      dg[:, None, D:], axis=1)
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
    h0 = cache["H"][0]
    dpre = dh * (1 - h0 ** 2)
    G["Win"] = cache["X"].reshape(-1, 1).T @ dpre.reshape(-1, D)
    G["bin"] = dpre.sum((0, 1))
    return loss, acc, G

# ---------------------------------------------------------------- обучение
def train(epochs=200, batch=64, lr=3e-3, time_budget=None):
    import time
    t_start = time.time()
    Xtr, ytr = make_dataset(900)
    Xte, yte = make_dataset(300)
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    history, t = [], 0
    for ep in range(epochs):
        idx = rng.permutation(len(ytr))
        ep_loss = ep_acc = nb = 0
        for i in range(0, len(ytr), batch):
            j = idx[i:i + batch]
            loss, acc, G = loss_and_grads(Xtr[j], ytr[j])
            t += 1
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh, vh = M[k] / (1 - 0.9 ** t), V[k] / (1 - 0.999 ** t)
                P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            ep_loss += loss; ep_acc += acc; nb += 1
        logits, _ = forward(Xte)
        te_acc = (logits.argmax(1) == yte).mean()
        history.append((ep_loss / nb, ep_acc / nb, te_acc))
        if (ep + 1) % 20 == 0:
            print(f"эпоха {ep+1:3d}  loss {ep_loss/nb:.3f}  "
                  f"train acc {ep_acc/nb:.3f}  test acc {te_acc:.3f}")
        if time_budget and time.time() - t_start > time_budget:
            print(f"(остановка по бюджету времени на эпохе {ep+1})")
            break
    return history, (Xte, yte)

# ---------------------------------------------------------------- запуск
if __name__ == "__main__":
    import os
    if os.environ.get("CONTINUE") and os.path.exists("weights.npz"):
        for k, v in np.load("weights.npz").items():
            P[k] = v
        print("(веса загружены, дообучение)")
    history, (Xte, yte) = train(
        time_budget=float(os.environ.get("TIME_BUDGET", 0)) or None)
    np.savez("weights.npz", **P)
    logits, cache = forward(Xte)
    pred = logits.argmax(1)
    print(f"\nИтоговая точность на тесте: {(pred == yte).mean():.3f}")
    names = ["кольцо", "пятно", "полоса"]
    for c in range(3):
        m = yte == c
        print(f"  {names[c]:7s}: {(pred[m] == c).mean():.3f}")

    # ------------------------------------------------------------ картинка
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def draw_flower(ax, values, title, sample_edges=True):
        vmax = np.abs(values).max() + 1e-9
        for i, (x, yy) in enumerate(pos):
            ax.add_patch(Circle((x, yy), 1.0, fill=False,
                                lw=0.4, color="0.75", zorder=1))
        if sample_edges:
            ii, jj = np.where(np.triu(A) > 0)
            for i, j in zip(ii, jj):
                ax.plot(*zip(pos[i], pos[j]), color="0.85", lw=0.4, zorder=0)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, s=42,
                        cmap="coolwarm", vmin=-vmax, vmax=vmax, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")

    # пример: как сигнал распространяется по решётке
    s = int(np.where(yte == 0)[0][0])                # образец «кольцо»
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    draw_flower(axes[0, 0], Xte[s, :, 0], "Вход: узор «кольцо»")
    for k, step in enumerate([1, 4, 8]):
        h = cache["H"][step][s]
        draw_flower(axes[0, 1] if k == 0 else axes[1, k - 1],
                    h[:, np.argmax(h.var(0))],
                    f"Активации после шага {step}")
    ep = np.arange(1, len(history) + 1)
    hist = np.array(history)
    ax = axes[0, 2]
    ax.plot(ep, hist[:, 1], label="train")
    ax.plot(ep, hist[:, 2], label="test")
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность")
    ax.set_title("Обучение", fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 2]
    cm = np.zeros((3, 3), int)
    for yt, yp in zip(yte, pred):
        cm[yt, yp] += 1
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_xlabel("предсказано"); ax.set_ylabel("истина")
    ax.set_title("Матрица ошибок (тест)", fontsize=10)
    fig.suptitle("Сеть на решётке «Цветок жизни»: "
                 f"{N} узлов, {int(A.sum()/2)} связей, {K} шагов распространения",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_net.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_net.png")
