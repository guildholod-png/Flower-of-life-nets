# -*- coding: utf-8 -*-
"""
3D-версия сети «Цветок жизни»: шары, заполняющие пространство.

Узлы = центры шаров в FCC-упаковке (плотнейшая упаковка сфер,
у каждого шара 12 касающихся соседей — kissing number в 3D).
Рёбра = касания/пересечения соседних шаров.
Срез такой структуры по плоскости (111) — в точности гексагональный
«Цветок жизни» из 2D-версии: 3D-структура состоит из его слоёв.

Модель и обучение те же: рекуррентный message passing с общими весами,
ручной backprop на numpy, Adam.
Задача: классификация 3D-узоров (оболочка / сгусток / плоский слой).
"""
import numpy as np

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- решётка
def build_fcc(Rc):
    """FCC-упаковка: целые (i,j,k) с чётной суммой, |p|<=Rc.
    Декартовы координаты делим на sqrt(2), чтобы расстояние
    между соседними шарами было 1 (шары радиуса 1/2 касаются)."""
    m = int(np.ceil(Rc * np.sqrt(2))) + 1
    axial, pts = [], []
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            for k in range(-m, m + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) / np.sqrt(2)
                    if np.linalg.norm(p) <= Rc:
                        axial.append((i, j, k)); pts.append(p)
    index = {a: n for n, a in enumerate(axial)}
    pos = np.array(pts)
    dirs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
            for c in (-1, 0, 1) if abs(a) + abs(b) + abs(c) == 2]  # 12 соседей
    n = len(axial)
    A = np.zeros((n, n))
    for (i, j, k), u in index.items():
        for di, dj, dk in dirs:
            v = index.get((i + di, j + dj, k + dk))
            if v is not None:
                A[u, v] = 1.0
    A_hat = A / np.maximum(A.sum(1, keepdims=True), 1)
    return axial, pos, A, A_hat

Rc = 3.6
axial, pos, A, A_hat = build_fcc(Rc)
N = len(pos)
deg = A.sum(1)
print(f"FCC-решётка: {N} узлов (шаров), {int(A.sum()/2)} связей, "
      f"соседей у внутренних узлов: {int(deg.max())}")

# ---------------------------------------------------------------- данные
def make_dataset(n_samples):
    """3 класса 3D-узоров: 0=сферическая оболочка, 1=сгусток, 2=плоский слой."""
    X = np.zeros((n_samples, N, 1))
    y = rng.integers(0, 3, n_samples)
    inner = np.where(np.linalg.norm(pos, axis=1) < Rc - 1.6)[0]
    for s in range(n_samples):
        c = pos[rng.choice(inner)]
        d = np.linalg.norm(pos - c, axis=1)
        if y[s] == 0:                                # оболочка
            r0 = rng.uniform(1.2, 2.2)
            v = (np.abs(d - r0) < 0.45).astype(float)
        elif y[s] == 1:                              # сгусток
            v = np.exp(-d ** 2 / (2 * 0.85 ** 2))
        else:                                        # плоский слой
            nvec = rng.normal(size=3)
            nvec /= np.linalg.norm(nvec)
            v = (np.abs((pos - c) @ nvec) < 0.45).astype(float)
        X[s, :, 0] = v + rng.normal(0, 0.15, N)      # шум
    return X, y

# ---------------------------------------------------------------- модель
D, K = 8, 7          # размерность признаков узла, число шагов распространения

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
    for _ in range(K):                               # волна по 3D-решётке
        m = np.matmul(A_hat, h)                      # среднее по 12 соседям
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
    acc = (p.argmax(1) == y).mean()

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
    h0 = cache["H"][0]
    dpre = dh * (1 - h0 ** 2)
    G["Win"] = cache["X"].reshape(-1, 1).T @ dpre.reshape(-1, D)
    G["bin"] = dpre.reshape(-1, D).sum(0)
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
        if (ep + 1) % 10 == 0:
            print(f"эпоха {ep+1:3d}  loss {ep_loss/nb:.3f}  "
                  f"train acc {ep_acc/nb:.3f}  test acc {te_acc:.3f}")
        if time_budget and time.time() - t_start > time_budget:
            print(f"(остановка по бюджету времени на эпохе {ep+1})")
            break
    return history, (Xte, yte)

# ---------------------------------------------------------------- запуск
if __name__ == "__main__":
    import os
    if os.environ.get("CONTINUE") and os.path.exists("weights3d.npz"):
        for k, v in np.load("weights3d.npz").items():
            P[k] = v
        print("(веса загружены, дообучение)")
    history, (Xte, yte) = train(
        time_budget=float(os.environ.get("TIME_BUDGET", 0)) or None)
    np.savez("weights3d.npz", **P)
    logits, cache = forward(Xte)
    pred = logits.argmax(1)
    print(f"\nИтоговая точность на тесте: {(pred == yte).mean():.3f}")
    names = ["оболочка", "сгусток", "слой"]
    for c in range(3):
        m = yte == c
        print(f"  {names[c]:8s}: {(pred[m] == c).mean():.3f}")

    # ------------------------------------------------------------ картинка
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # срез (111): узлы с i+j+k == 0 образуют гексагональный слой
    layer = [n for n, (i, j, k) in enumerate(axial) if i + j + k == 0]
    u = np.array([1, -1, 0]) / np.sqrt(2)
    v = np.array([1, 1, -2]) / np.sqrt(6)
    lp = np.stack([pos[layer] @ u, pos[layer] @ v], 1)

    def draw3d(ax, values, title):
        vmax = np.abs(values).max() + 1e-9
        order = np.argsort(pos[:, 1])                 # грубая сортировка по глубине
        sz = 14 + 60 * (np.abs(values) / vmax)
        ax.scatter(pos[order, 0], pos[order, 1], pos[order, 2],
                   c=values[order], s=sz[order], cmap="coolwarm",
                   vmin=-vmax, vmax=vmax, alpha=0.85, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_box_aspect((1, 1, 1)); ax.axis("off")

    def draw_slice(ax, values, title):
        vmax = np.abs(values).max() + 1e-9
        for x, yy in lp:
            ax.add_patch(Circle((x, yy), 1.0, fill=False,
                                lw=0.5, color="0.75", zorder=1))
        ax.scatter(lp[:, 0], lp[:, 1], c=values, s=60, cmap="coolwarm",
                   vmin=-vmax, vmax=vmax, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")

    s = int(np.where(yte == 0)[0][0])                 # образец «оболочка»
    hK = cache["H"][K][s]
    ch = int(np.argmax(hK.var(0)))                    # самый информативный признак

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    draw3d(ax, Xte[s, :, 0], "Вход: 3D-узор «оболочка»")
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    draw3d(ax, cache["H"][3][s][:, ch], "Активации после шага 3")
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    draw3d(ax, hK[:, ch], f"Активации после шага {K}")

    ax = fig.add_subplot(2, 3, 4)
    draw_slice(ax, Xte[s, layer, 0],
               "Срез (111): слой FCC —\nтот самый «Цветок жизни»")
    ax = fig.add_subplot(2, 3, 5)
    hist = np.array(history)
    ep = np.arange(1, len(hist) + 1)
    ax.plot(ep, hist[:, 1], label="train")
    ax.plot(ep, hist[:, 2], label="test")
    ax.set_xlabel("эпоха"); ax.set_ylabel("точность")
    ax.set_title("Обучение", fontsize=10); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
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

    fig.suptitle("3D-сеть на FCC-упаковке шаров: "
                 f"{N} шаров, {int(A.sum()/2)} связей, 12 соседей у шара, "
                 f"{K} шагов распространения", fontsize=12)
    fig.tight_layout()
    fig.savefig("flower_of_life_net_3d.png", dpi=150, bbox_inches="tight")
    print("Сохранено: flower_of_life_net_3d.png")
