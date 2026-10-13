import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Модель ФицХью — Нагумо

    Интерактивная дискретная модель с ползунками.

    $$\begin{cases}
    \dfrac{dv}{dt} = v - \dfrac{v^3}{3} - w + I_{\text{ext}}, \\[6pt]
    \dfrac{dw}{dt} = \varepsilon\,(v + a - b\,w),
    \end{cases}$$

    где:

    - $v$ — мембранный потенциал (быстрая переменная);
    - $w$ — переменная восстановления (медленная переменная);
    - $I_{\text{ext}}$ — внешний ток;
    - $\varepsilon$ — малый параметр, $0 < \varepsilon \ll 1$;
    - $a, b$ — постоянные параметры модели.

    **Дискретизация (метод Эйлера с шагом $\Delta t$):**

    $$\begin{cases}
    v_{n+1} = v_n + \Delta t\left(v_n - \dfrac{v_n^3}{3} - w_n + I_{\text{ext}}\right), \\[6pt]
    w_{n+1} = w_n + \Delta t\,\varepsilon\,(v_n + a - b\,w_n).
    \end{cases}$$
    """)
    return


@app.cell
def _(mo):
    I_ext = mo.ui.slider(0.0, 1.5, 0.01, value=0.5, label="I_ext (внешний ток)")
    a     = mo.ui.slider(0.0, 1.5, 0.01, value=0.7, label="a")
    b     = mo.ui.slider(0.0, 1.5, 0.01, value=0.8, label="b")
    eps   = mo.ui.slider(0.01, 0.3, 0.01, value=0.08, label="ε")
    return I_ext, a, b, eps


@app.cell
def _(I_ext, a, b, eps, mo):
    controls = mo.vstack([
        mo.md("### Параметры модели"),
        I_ext, a, b, eps,
    ])
    controls
    return (controls,)


@app.cell
def _(I_ext, a, b, eps, np):
    # --- фиксированные параметры интегрирования ---
    dt = 0.01
    N  = 20000
    n  = np.arange(N)
    t  = n * dt

    # --- массивы решения ---
    v = np.zeros(N)
    w = np.zeros(N)
    v[0], w[0] = -1.0, -0.5

    # --- читаем значения ползунков ---
    I_val   = I_ext.value
    a_val   = a.value
    b_val   = b.value
    eps_val = eps.value

    # --- итерации ---
    for k in range(N - 1):
        dv = v[k] - v[k]**3 / 3.0 - w[k] + I_val
        dw = eps_val * (v[k] + a_val - b_val * w[k])
        v[k + 1] = v[k] + dt * dv
        w[k + 1] = w[k] + dt * dw
    return I_val, a_val, b_val, eps_val, t, v, w


@app.cell
def _(I_val, a_val, b_val, eps_val, plt, t, v, w):
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t, v, 'b', label='v (потенциал)')
    ax1.plot(t, w, 'r', label='w (восстановление)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Значение')
    ax1.set_title(f'Временные ряды (I_ext = {I_val:.2f}, a = {a_val:.2f}, b = {b_val:.2f}, ε = {eps_val:.2f})')
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(plt, v, w):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(v, w, 'k', lw=0.8)
    ax2.plot(v[0], w[0], 'go', label='старт')
    ax2.set_xlabel('v')
    ax2.set_ylabel('w')
    ax2.set_title('Фазовый портрет')
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(controls, mo, plt, t, v, w):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    axes[0].plot(t, v, 'b', label='v')
    axes[0].plot(t, w, 'r', label='w')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('Значение')
    axes[0].set_title('Временные ряды')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(v, w, 'k', lw=0.8)
    axes[1].set_xlabel('v'); axes[1].set_ylabel('w')
    axes[1].set_title('Фазовый портрет')
    axes[1].grid(True)

    fig.tight_layout()

    mo.hstack([controls, fig], justify="start", gap=2)
    return


if __name__ == "__main__":
    app.run()
