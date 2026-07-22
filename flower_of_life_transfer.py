# -*- coding: utf-8 -*-
"""
Zero-shot перенос: правило, выученное на решётке R=6 (127 окружностей),
прикладывается БЕЗ ДООБУЧЕНИЯ к решёткам R=8, 10, 12 (до 469 окружностей).

Возможно это потому, что веса общие для всех узлов: локальное правило
не привязано к размеру структуры — как ядро свёртки к размеру картинки.
Узоры те же (в тех же абсолютных единицах), но поле больше,
и центры узоров разбросаны по всей новой площади.
"""
import numpy as np

rng = np.random.default_rng(42)

def build_hex(R, nb12=True):
    axial = [(q, r) for q in range(-R, R + 1) for r in range(-R, R + 1)
             if max(abs(q), abs(r), abs(q + r)) <= R]
    pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    A = ((d > 1e-9) & (d < (1.9 if nb12 else 1.1))).astype(float)
    return pos, A / np.maximum(A.sum(1, keepdims=True), 1)

def make_dataset(pos, n_samples):
    N = len(pos)
    Rc = np.max(np.linalg.norm(pos, axis=1))
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < Rc - 2.8)[0]
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

D, K = 8, 4

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
        m = np.matmul(A_hat, h)
        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
        H_list.append(h)
    amax = h.argmax(axis=1)
    g = np.concatenate([h.mean(axis=1),
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

# ------------------------------------------------------- обучение на R=6
pos6, A6 = build_hex(6)
Xtr, ytr = make_dataset(pos6, 900)
Xte6, yte6 = make_dataset(pos6, 300)
print(f"Обучение на R=6: {len(pos6)} узлов")

EPOCHS, BATCH, LR = 100, 64, 3e-3
M = {k: np.zeros_like(v) for k, v in P.items()}
V = {k: np.zeros_like(v) for k, v in P.items()}
adam_t = 0
for ep in range(EPOCHS):
    idx = rng.permutation(len(ytr))
    for i in range(0, len(ytr), BATCH):
        j = idx[i:i + BATCH]
        loss, G = loss_and_grads(Xtr[j], ytr[j], A6)
        adam_t += 1
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
            P[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
    if (ep + 1) % 25 == 0:
        logits, _, _, _ = forward(Xte6, A6)
        print(f"эпоха {ep+1:3d}  test acc {(logits.argmax(1)==yte6).mean():.3f}")

# ------------------------------------------------------- zero-shot на больших
results = []
for R in [6, 8, 10, 12]:
    posR, AR = build_hex(R)
    XR, yR = make_dataset(posR, 300)
    logits, _, _, _ = forward(XR, AR)
    acc = (logits.argmax(1) == yR).mean()
    per = [( (logits.argmax(1)==yR) & (yR==c) ).sum() / max((yR==c).sum(),1)
           for c in range(3)]
    results.append((R, len(posR), acc, per))
    print(f"R={R:2d}  {len(posR):3d} узлов  zero-shot acc {acc:.3f}  "
          f"(кольцо {per[0]:.2f} пятно {per[1]:.2f} полоса {per[2]:.2f})")

np.savez("transfer_results.npz",
         R=[r[0] for r in results], N=[r[1] for r in results],
         acc=[r[2] for r in results], per=[r[3] for r in results])

# ------------------------------------------------------- картинка
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Rs = [r[0] for r in results]
Ns = [r[1] for r in results]
accs = [r[2] for r in results]
pers = np.array([r[3] for r in results])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(Ns, accs, "o-", color="crimson", lw=2, label="все классы")
names = ["кольцо", "пятно", "полоса"]
for c, ls in zip(range(3), [":", "--", "-."]):
    ax.plot(Ns, pers[:, c], ls, color="0.5", label=names[c])
ax.axvline(Ns[0], color="steelblue", ls="--", alpha=0.7)
ax.text(Ns[0] + 5, 0.62, "обучалась\nтолько здесь", fontsize=9,
        color="steelblue")
ax.set_xlabel("узлов в решётке"); ax.set_ylabel("точность")
ax.set_ylim(0.5, 1.02)
ax.set_title("Zero-shot: веса с R=6 без дообучения", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for R, col in zip([6, 12], ["steelblue", "crimson"]):
    posR, _ = build_hex(R)
    ax.scatter(posR[:, 0], posR[:, 1], s=14 if R == 12 else 26, color=col,
               alpha=0.6 if R == 12 else 1.0,
               label=f"R={R}: {len(posR)} узлов")
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Одно и то же правило работает на обоих", fontsize=10)
ax.legend(fontsize=9, loc="upper right")

fig.suptitle("Локальное правило переносится на структуру любого размера",
             fontsize=12)
fig.tight_layout()
fig.savefig("flower_of_life_transfer.png", dpi=150, bbox_inches="tight")
print("Сохранено: flower_of_life_transfer.png")
