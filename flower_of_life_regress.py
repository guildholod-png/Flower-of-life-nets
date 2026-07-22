# -*- coding: utf-8 -*-
"""
Регрессия: НАЙТИ ЦЕНТР узора на решётке (не «что это», а «где это»).

Позиционный readout: сеть выдаёт вес внимания на каждый узел,
ответ = взвешенное среднее КООРДИНАТ узлов. Координаты берутся
из процесса роста от центра (та самая «бесплатная» система координат) —
сеть учится только тому, КУДА смотреть, а не где узлы находятся.

Узоры: кольцо и пятно (у полосы центр вдоль линии не определён).
Шум 0.4. Метрика: среднее расстояние до истинного центра (в шагах решётки).

Соперники:
  centroid — эвристика без обучения: центр масс положительной части входа
  mlp      — 127 -> 64 -> 2 координаты напрямую
"""
import numpy as np

rng = np.random.default_rng(42)

R_MAX, NOISE = 6, 0.4
axial = [(q, r) for q in range(-R_MAX, R_MAX + 1) for r in range(-R_MAX, R_MAX + 1)
         if max(abs(q), abs(r), abs(q + r)) <= R_MAX]
pos = np.array([(q + r / 2.0, r * np.sqrt(3) / 2.0) for q, r in axial])
N = len(axial)
d2 = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
A = ((d2 > 1e-9) & (d2 < 1.9)).astype(float)
A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)

def make_dataset(n_samples):
    X = np.zeros((n_samples, N, 1))
    C = np.zeros((n_samples, 2))
    inner = np.where(np.linalg.norm(pos, axis=1) < R_MAX - 2.5)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if rng.random() < 0.5:                         # кольцо
            r0 = rng.uniform(1.6, 3.2)
            v = (np.abs(d - r0) < 0.55).astype(float)
        else:                                          # пятно
            v = np.exp(-d ** 2 / (2 * 1.1 ** 2))
        X[s, :, 0] = v + rng.normal(0, NOISE, N)
        C[s] = c
    return X, C

Xtr, Ctr = make_dataset(900)
Xte, Cte = make_dataset(300)

# ------------------------------------------------------- модель с вниманием
D, K = 8, 4

def init(shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0])
    return rng.normal(0, scale, shape)

P = {"Win": init((1, D)), "bin": np.zeros(D),
     "Ws": init((D, D)), "Wn": init((D, D)), "b": np.zeros(D),
     "w": init((D,)), "bs": 0.0}

def forward(X):
    H_list = []
    h = np.tanh(X @ P["Win"] + P["bin"])
    H_list.append(h)
    for _ in range(K):
        m = np.matmul(A_hat, h)
        h = np.tanh(h @ P["Ws"] + m @ P["Wn"] + P["b"])
        H_list.append(h)
    s = h @ P["w"] + P["bs"]                           # (B,N) скоры внимания
    s = s - s.max(1, keepdims=True)
    alpha = np.exp(s); alpha /= alpha.sum(1, keepdims=True)
    chat = alpha @ pos                                 # (B,2)
    return chat, alpha, H_list

def loss_and_grads(X, C):
    B = len(C)
    chat, alpha, H_list = forward(X)
    diff = chat - C
    loss = (diff ** 2).sum(1).mean()
    G = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in P.items()}
    dchat = 2 * diff / B                               # (B,2)
    dalpha = dchat @ pos.T                             # (B,N)
    ds = alpha * (dalpha - (alpha * dalpha).sum(1, keepdims=True))
    h = H_list[K]
    G["w"] = np.einsum("bn,bnd->d", ds, h)
    G["bs"] = ds.sum()
    dh = ds[:, :, None] * P["w"][None, None, :]
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
EPOCHS, BATCH, LR = 120, 64, 3e-3
M = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in P.items()}
V = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in P.items()}
adam_t = 0
for ep in range(EPOCHS):
    idx = rng.permutation(len(Ctr))
    for i in range(0, len(Ctr), BATCH):
        j = idx[i:i + BATCH]
        loss, G = loss_and_grads(Xtr[j], Ctr[j])
        adam_t += 1
        for k in P:
            M[k] = 0.9 * M[k] + 0.1 * G[k]
            V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            mh, vh = M[k] / (1 - 0.9 ** adam_t), V[k] / (1 - 0.999 ** adam_t)
            P[k] = P[k] - LR * mh / (np.sqrt(vh) + 1e-8)
    if (ep + 1) % 30 == 0:
        chat, _, _ = forward(Xte)
        err = np.linalg.norm(chat - Cte, axis=1).mean()
        print(f"эпоха {ep+1:3d}  ср. ошибка {err:.3f} шага решётки")

chat, alpha, _ = forward(Xte)
err_gnn = np.linalg.norm(chat - Cte, axis=1)

# ------------------------------------------------------- соперники
w_pos = np.maximum(Xte[:, :, 0], 0)                    # центроид входа
cent = (w_pos @ pos) / w_pos.sum(1, keepdims=True)
err_cent = np.linalg.norm(cent - Cte, axis=1)

H = 64                                                  # MLP
Pm = {"W1": init((N, H)), "b1": np.zeros(H),
      "W2": init((H, 2)), "b2": np.zeros(2)}
Mm = {k: np.zeros_like(v) for k, v in Pm.items()}
Vm = {k: np.zeros_like(v) for k, v in Pm.items()}
t_ = 0
for ep in range(EPOCHS):
    idx = rng.permutation(len(Ctr))
    for i in range(0, len(Ctr), BATCH):
        j = idx[i:i + BATCH]
        hb = np.tanh(Xtr[j][:, :, 0] @ Pm["W1"] + Pm["b1"])
        out = hb @ Pm["W2"] + Pm["b2"]
        dout = 2 * (out - Ctr[j]) / len(j)
        dhb = (dout @ Pm["W2"].T) * (1 - hb ** 2)
        Gm = {"W2": hb.T @ dout, "b2": dout.sum(0),
              "W1": Xtr[j][:, :, 0].T @ dhb, "b1": dhb.sum(0)}
        t_ += 1
        for k in Pm:
            Mm[k] = 0.9 * Mm[k] + 0.1 * Gm[k]
            Vm[k] = 0.999 * Vm[k] + 0.001 * Gm[k] ** 2
            mh, vh = Mm[k] / (1 - 0.9 ** t_), Vm[k] / (1 - 0.999 ** t_)
            Pm[k] -= LR * mh / (np.sqrt(vh) + 1e-8)
hb = np.tanh(Xte[:, :, 0] @ Pm["W1"] + Pm["b1"])
err_mlp = np.linalg.norm(hb @ Pm["W2"] + Pm["b2"] - Cte, axis=1)

print(f"\nсредняя ошибка (в шагах решётки):")
print(f"  центроид входа (без обучения): {err_cent.mean():.3f}")
print(f"  MLP:                           {err_mlp.mean():.3f}")
print(f"  решётка + внимание:            {err_gnn.mean():.3f}")

# ------------------------------------------------------- картинка
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
data = [err_cent, err_mlp, err_gnn]
labels = ["центроид\n(без обучения)", "MLP", "решётка +\nвнимание"]
colors = ["0.7", "0.45", "crimson"]
bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel("ошибка, шагов решётки")
ax.set_title(f"Средние: {err_cent.mean():.2f} / {err_mlp.mean():.2f} / "
             f"{err_gnn.mean():.2f}", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# пример: образец с медианной ошибкой (типичный случай)
si = int(np.argsort(err_gnn)[len(err_gnn) // 2])
ax = axes[1]
for i, (x, yy) in enumerate(pos):
    ax.add_patch(Circle((x, yy), 1.0, fill=False, lw=0.3, color="0.85"))
ax.scatter(pos[:, 0], pos[:, 1], c=Xte[si, :, 0], s=36, cmap="coolwarm")
ax.plot(*Cte[si], "k*", ms=16, label="истинный центр")
ax.plot(*chat[si], "o", color="lime", ms=10, mec="k", label="предсказание")
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Вход (шум 0.4)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

ax = axes[2]
ax.scatter(pos[:, 0], pos[:, 1], c=alpha[si], s=36, cmap="viridis")
ax.plot(*Cte[si], "r*", ms=16)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Куда смотрит внимание сети", fontsize=10)

fig.suptitle("Регрессия центра узора: координаты узлов из процесса роста, "
             "сеть учится только вниманию", fontsize=11)
fig.tight_layout()
fig.savefig("flower_of_life_regress.png", dpi=150, bbox_inches="tight")
print("Сохранено: flower_of_life_regress.png")
