# -*- coding: utf-8 -*-
"""
Растущая сеть «Цветок жизни»: буквальное прочтение рисунка.

Сеть начинается с ОДНОЙ окружности (1 узел) и наращивает кольца —
как окружности распространяются на рисунке — каждый раз, когда
качество выходит на плато («ёмкости перестало хватать»).

Ключевое свойство: веса общие для всех узлов (как ядро свёртки),
поэтому при добавлении кольца число параметров НЕ меняется —
выученное локальное правило мгновенно применяется к новым узлам.
Растёт не память сети, а её «поле зрения» над входным полем.

Запуск: MODE=grow (растущая) или MODE=fixed (базовая, полная решётка).
После второго запуска строится сравнительная картинка.
"""
import os
import numpy as np

rng = np.random.default_rng(42)
MODE = os.environ.get("MODE", "grow")

# ------------------------------------------------------- полная решётка R=6
R_MAX = 6
axial = [(q, r) for q in range(-R_MAX, R_MAX + 1) for r in range(-R_MAX, R_MAX + 1)
         if max(abs(q), abs(r), abs(q + r)) <= R_MAX]
index = {a: i for i, a in enumerate(axial)}
pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
ring = np.array([(abs(q) + abs(r) + abs(q + r)) // 2 for q, r in axial])
N_FULL = len(axial)
A_full = np.zeros((N_FULL, N_FULL))
for (q, r), i in index.items():
    for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        j = index.get((q + dq, r + dr))
        if j is not None:
            A_full[i, j] = 1.0

# ------------------------------------------------------- данные (полное поле)
def make_dataset(n_samples):
    X = np.zeros((n_samples, N_FULL, 1))
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
        X[s, :, 0] = v + rng.normal(0, 0.15, N_FULL)
    return X, y

Xtr, ytr = make_dataset(900)
Xte, yte = make_dataset(300)

# ------------------------------------------------------- модель (веса общие)
D = 8

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {
    "Win": init((1, D)), "bin": np.zeros(D),
    "Ws":  init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
    "Wo":  init((2 * D, 3), 1.0 / np.sqrt(2 * D)), "bo": np.zeros(3),
}
N_PARAMS = sum(v.size for v in P.values())

def forward(X, A_hat, K):
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

def loss_and_grads(X, y, A_hat, K):
    B, n = len(y), X.shape[1]
    logits, cache = forward(X, A_hat, K)
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
    dh = dg[:, None, :D] / n * np.ones((B, n, 1))
    dmax = np.zeros((B, n, D))
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

def sub(R):
    """Активная часть решётки: кольца 0..R."""
    mask = ring <= R
    A = A_full[np.ix_(mask, mask)]
    A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)
    return mask, A_hat

def test_acc(mask, A_hat, K):
    logits, _ = forward(Xte[:, mask], A_hat, K)
    return (logits.argmax(1) == yte).mean()

# ------------------------------------------------------- обучение
EPOCHS, BATCH, LR = 100, 64, 3e-3
PATIENCE, MIN_STAGE, EPS = 6, 5, 0.005

M = {k: np.zeros_like(v) for k, v in P.items()}
V = {k: np.zeros_like(v) for k, v in P.items()}
adam_t = 0

R = 0 if MODE == "grow" else R_MAX          # растущая стартует с 1 окружности
mask, A_hat = sub(R)
K = R + 2
acc_hist, nodes_hist, growth_epochs = [], [], []
best_acc, best_ep, stage_ep = -1, 0, 0

print(f"[{MODE}] старт: R={R}, узлов {mask.sum()}, параметров {N_PARAMS}")
for ep in range(EPOCHS):
    idx = rng.permutation(len(ytr))
    for i in range(0, len(ytr), BATCH):
        j = idx[i:i + BATCH]
        loss, G = loss_and_grads(Xtr[j][:, mask], ytr[j], A_hat, K)
        adam_t += 1
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
            P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
    ta = test_acc(mask, A_hat, K)
    acc_hist.append(ta); nodes_hist.append(int(mask.sum()))
    stage_ep += 1
    if ta > best_acc + EPS:
        best_acc, best_ep = ta, ep
    if (ep + 1) % 10 == 0:
        print(f"эпоха {ep+1:3d}  R={R}  узлов {mask.sum():3d}  test acc {ta:.3f}")
    # плато -> нарастить кольцо (только в режиме grow)
    if (MODE == "grow" and R < R_MAX and stage_ep >= MIN_STAGE
            and ep - best_ep >= PATIENCE):
        R += 1
        mask, A_hat = sub(R)
        K = R + 2
        growth_epochs.append(ep + 1)
        best_acc, best_ep, stage_ep = -1, ep, 0
        print(f"  >>> плато: наращиваю кольцо {R} "
              f"(узлов стало {mask.sum()}, параметров по-прежнему {N_PARAMS})")

logits, _ = forward(Xte[:, mask], A_hat, K)
pred = logits.argmax(1)
final = (pred == yte).mean()
print(f"\n[{MODE}] итоговая точность: {final:.3f} (R={R}, узлов {mask.sum()})")
cm = np.zeros((3, 3), int)
for yt, yp in zip(yte, pred):
    cm[yt, yp] += 1

np.savez(f"hist_{MODE}.npz", acc=np.array(acc_hist),
         nodes=np.array(nodes_hist), growth=np.array(growth_epochs),
         cm=cm, sample=Xte[int(np.where(yte == 0)[0][0]), :, 0])

# ------------------------------------------------------- сравнительная картинка
if os.path.exists("hist_grow.npz") and os.path.exists("hist_fixed.npz"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    g = np.load("hist_grow.npz")
    f = np.load("hist_fixed.npz")
    names = ["кольцо", "пятно", "полоса"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(g["acc"]) + 1), g["acc"],
            label="растущая (с 1 окружности)", color="crimson")
    ax.plot(np.arange(1, len(f["acc"]) + 1), f["acc"],
            label="фиксированная (сразу 127)", color="steelblue")
    for i, ge in enumerate(g["growth"]):
        ax.axvline(ge, color="crimson", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность на тесте")
    ax.set_title("Пунктир — моменты наращивания кольца", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.step(np.arange(1, len(g["nodes"]) + 1), g["nodes"],
            where="post", color="crimson", label="узлов (растущая)")
    ax.step(np.arange(1, len(f["nodes"]) + 1), f["nodes"],
            where="post", color="steelblue", label="узлов (фиксированная)")
    ax.axhline(N_PARAMS, color="gray", ls="--",
               label=f"параметров (обе): {N_PARAMS}")
    ax.set_xlabel("эпоха"); ax.set_title("Структура растёт — веса нет", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.imshow(g["cm"], cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, g["cm"][i, j], ha="center", va="center", fontsize=11)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set_title("Матрица ошибок растущей сети", fontsize=10)

    sample = g["sample"]
    vmax = np.abs(sample).max()
    for col, Rs in enumerate([1, 3, 6]):
        ax = axes[1, col]
        m = ring <= Rs
        for i in np.where(m)[0]:
            ax.add_patch(Circle(pos[i], 1.0, fill=False, lw=0.5,
                                color="0.6", zorder=1))
        ax.scatter(pos[m, 0], pos[m, 1], c=sample[m], s=40, cmap="coolwarm",
                   vmin=-vmax, vmax=vmax, zorder=2)
        ax.scatter(pos[~m, 0], pos[~m, 1], c="0.88", s=10, zorder=0)
        ax.set_xlim(pos[:, 0].min() - 1.3, pos[:, 0].max() + 1.3)
        ax.set_ylim(pos[:, 1].min() - 1.3, pos[:, 1].max() + 1.3)
        n_r = int(m.sum())
        note = "серое поле ещё не видно" if Rs < R_MAX else "видно всё поле"
        ax.set_title(f"Рост: {Rs} колец ({n_r} окружн.) — {note}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("Растущая сеть «Цветок жизни»: структура распространяется, "
                 "как окружности на рисунке; веса переносятся на новые узлы",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_grow.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_grow.png")
