import os
import pickle
import random

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ranges of the data the model was trained on (the placement CSV). Used only for
# the "outside training range" notice and for drawing the charts.
TRAIN_CGPA = (3.3, 8.5)
TRAIN_IQ = (37.0, 233.0)


# --------------------------------------------------------------------------- #
# Model loading and prediction (same logic as before: model.pkl + scaler.pkl,
# features sent in the notebook's order: CGPA first, then IQ)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None


def to_model_input(cgpa, iq, scaler):
    """Build the input row in the exact feature order used in training (CGPA, IQ)."""
    if hasattr(scaler, "feature_names_in_"):
        return pd.DataFrame([[cgpa, iq]], columns=list(scaler.feature_names_in_))
    return np.array([[cgpa, iq]], dtype=float)


def predict_one(model, scaler, cgpa, iq):
    scaled = scaler.transform(to_model_input(cgpa, iq, scaler))
    label = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1]) if hasattr(model, "predict_proba") else float(label)
    return label, prob


def probability_curve(model, scaler, iq, points=101):
    """P(placed) for CGPA 0..10 at a fixed IQ. Drives the chart and the tipping point."""
    xs = np.linspace(0, 10, points)
    frame = pd.DataFrame({"cgpa": xs, "iq": float(iq)})
    if hasattr(scaler, "feature_names_in_"):
        frame = frame[list(scaler.feature_names_in_)]
    else:
        frame = frame.to_numpy()
    scaled = scaler.transform(frame)
    return xs, model.predict_proba(scaled)[:, 1]


def tipping_cgpa(xs, ps):
    """First CGPA where the model's probability reaches 50% (None if it never does)."""
    idx = np.where(ps >= 0.5)[0]
    return float(xs[idx[0]]) if len(idx) else None


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --ink: #0d0a1f;
  --ink-2: #17122f;
  --line: rgba(255,255,255,.10);
  --text: #eeeaff;
  --muted: #a9a3cc;
  --violet: #8b6cff;
  --mint: #2ee6a6;
  --coral: #ff5d73;
  --amber: #ffc857;
}

.stApp {
  background:
    radial-gradient(900px 500px at 12% -10%, rgba(139,108,255,.35), transparent 60%),
    radial-gradient(800px 500px at 100% 10%, rgba(46,230,166,.16), transparent 60%),
    radial-gradient(700px 600px at 50% 120%, rgba(255,93,115,.14), transparent 60%),
    var(--ink);
  color: var(--text);
  font-family: 'DM Sans', system-ui, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 780px; padding-top: 2.2rem; padding-bottom: 4rem; }

/* ---------- hero (the one orchestrated page-load moment) ---------- */
.hero { padding: 1.2rem 0 1.6rem; }
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2.3rem, 7vw, 3.8rem); line-height: 1.02; letter-spacing: -0.03em;
  margin: 0 0 .9rem; padding: 0; color: var(--text);
}
.hero h1 .grad {
  background: linear-gradient(100deg, var(--violet), var(--mint), var(--amber), var(--violet));
  background-size: 250% 100%; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
  animation: slide-grad 7s linear infinite;
}
@keyframes slide-grad { to { background-position: 250% 0; } }
.typer {
  font-family: 'JetBrains Mono', monospace; font-size: .98rem; color: var(--mint);
  display: inline-block; white-space: nowrap; overflow: hidden; max-width: 100%;
  border-right: 2px solid var(--mint); width: 34ch;
  animation: typing 2.2s steps(34, end) both, caret .8s step-end infinite;
}
@keyframes typing { from { width: 0; } }
@keyframes caret { 50% { border-color: transparent; } }
.sub { color: var(--muted); max-width: 56ch; margin: .8rem 0 0; line-height: 1.6; }

/* ---------- inputs ---------- */
div[data-testid="stForm"] {
  background: rgba(255,255,255,.04); border: 1px solid var(--line);
  border-radius: 20px; padding: 1.4rem 1.4rem 1.2rem;
  backdrop-filter: blur(10px);
}
div[data-testid="stForm"] h3 { font-family: 'Bricolage Grotesque', sans-serif; color: var(--text); }
.stApp label p, .stApp [data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 500; }
.stApp div[data-baseweb="input"], .stApp div[data-baseweb="base-input"] {
  background: var(--ink-2) !important; border-radius: 12px !important; border-color: var(--line) !important;
}
.stApp div[data-baseweb="input"]:focus-within { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(139,108,255,.35); }
.stApp input { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; font-weight: 600; font-size: 1.05rem; }
.stApp div[data-testid="stNumberInput"] button { background: transparent !important; color: var(--muted) !important; }
.stApp div[data-testid="stFormSubmitButton"] button {
  width: 100%; border: 0; border-radius: 14px; padding: .8rem 1rem; color: #0d0a1f;
  background: linear-gradient(100deg, var(--mint), var(--violet)); font-weight: 700; font-size: 1.05rem;
  transition: transform .15s ease, box-shadow .15s ease;
}
.stApp div[data-testid="stFormSubmitButton"] button p { color: #0d0a1f !important; font-weight: 700; }
.stApp div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(139,108,255,.45); }
.stApp div[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }

/* ---------- tabs ---------- */
.stApp button[data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
.stApp button[data-baseweb="tab"][aria-selected="true"] p { color: var(--text); }
.stApp div[data-baseweb="tab-highlight"] { background: var(--violet) !important; }

/* ---------- result card ---------- */
.verdict {
  position: relative; overflow: hidden; margin-top: 1.4rem; padding: 1.8rem 1.6rem;
  border-radius: 24px; border: 1px solid var(--line); background: rgba(255,255,255,.05);
}
.verdict.yes { border-color: rgba(46,230,166,.55); box-shadow: 0 0 60px rgba(46,230,166,.18); animation: pop .6s cubic-bezier(.2,1.2,.4,1) both; }
.verdict.no  { border-color: rgba(255,93,115,.55); box-shadow: 0 0 60px rgba(255,93,115,.16); animation: shake .6s ease both; }
@keyframes pop { from { transform: scale(.94); opacity: 0; } }
@keyframes shake { 20% { transform: translateX(-8px); } 40% { transform: translateX(7px); } 60% { transform: translateX(-4px); } 80% { transform: translateX(2px); } }

.v-row { display: flex; gap: 1.6rem; align-items: center; flex-wrap: wrap; }
.v-text { flex: 1 1 260px; min-width: 0; }
.stamp {
  display: inline-block; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(1.8rem, 6vw, 2.6rem); letter-spacing: .02em; padding: .1em .5em;
  border: 4px solid currentColor; border-radius: 12px; transform: rotate(-5deg);
  animation: stamp .55s cubic-bezier(.2,1.4,.4,1) .15s both;
}
.yes .stamp { color: var(--mint); } .no .stamp { color: var(--coral); }
@keyframes stamp { from { transform: rotate(-18deg) scale(3); opacity: 0; } }
.v-msg { margin: 1rem 0 0; font-size: 1.05rem; line-height: 1.55; color: var(--text); }
.v-msg b { font-weight: 600; }

.ring { position: relative; width: 150px; height: 150px; flex: 0 0 150px; }
.ring svg { transform: rotate(-90deg); width: 150px; height: 150px; }
.ring .track { fill: none; stroke: rgba(255,255,255,.10); stroke-width: 11; }
.ring .bar {
  fill: none; stroke-width: 11; stroke-linecap: round; stroke-dasharray: 327;
  stroke-dashoffset: var(--off); animation: fill 1.4s cubic-bezier(.2,.8,.2,1) .3s both;
}
.yes .bar { stroke: var(--mint); } .no .bar { stroke: var(--coral); }
@keyframes fill { from { stroke-dashoffset: 327; } }
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }
.pct {
  position: absolute; inset: 0; display: grid; place-items: center; text-align: center;
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.1rem;
  counter-reset: n var(--n); animation: count 1.4s cubic-bezier(.2,.8,.2,1) .3s both;
}
.pct::after { content: counter(n) "%"; }
@keyframes count { from { --n: 0; } }

.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem; }
.chip { font-size: .85rem; padding: .35rem .75rem; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,.05); color: var(--text); }
.chip.warn { border-color: rgba(255,200,87,.6); color: var(--amber); }

/* confetti (placed only, plays once) */
.confetti { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.confetti i { position: absolute; top: -14px; width: 8px; height: 14px; border-radius: 2px; opacity: 0; animation: fall 2.8s ease-in forwards; }
@keyframes fall { 0% { transform: translateY(0) rotate(0); opacity: 1; } 100% { transform: translateY(420px) rotate(560deg); opacity: 0; } }

/* ---------- panels ---------- */
.panel { margin-top: 1rem; padding: 1.2rem 1.3rem; border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,.04); }
.panel h4 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.1rem; margin: 0 0 .5rem; color: var(--text); }
.panel p, .panel li { color: var(--muted); line-height: 1.6; margin: .3rem 0; }
.panel b { color: var(--text); }
.chart svg { width: 100%; height: auto; display: block; }
.chart .axis { stroke: rgba(255,255,255,.18); stroke-width: 1; }
.chart .guide { stroke: rgba(255,255,255,.28); stroke-dasharray: 4 5; stroke-width: 1; }
.chart text { fill: var(--muted); font-size: 11px; font-family: 'DM Sans', sans-serif; }
.chart .curve { fill: none; stroke: url(#cg); stroke-width: 3.5; stroke-linecap: round; stroke-dasharray: 1; stroke-dashoffset: 0; animation: draw 1.6s ease .2s both; }
@keyframes draw { from { stroke-dashoffset: 1; } }
.chart .dot { animation: pulse 1.8s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
@keyframes pulse { 50% { transform: scale(1.5); opacity: .7; } }

.waiting { margin-top: 1.4rem; padding: 1.6rem; text-align: center; border: 1px dashed rgba(255,255,255,.22); border-radius: 22px; color: var(--muted); }
.waiting .big { font-size: 2.2rem; display: block; animation: bob 2.4s ease-in-out infinite; }
@keyframes bob { 50% { transform: translateY(-6px); } }

.bars .row { display: grid; grid-template-columns: 70px 1fr 52px; align-items: center; gap: .7rem; margin: .5rem 0; color: var(--text); font-size: .92rem; }
.bars .track { height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }
.bars .fillbar { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--violet), var(--mint)); width: var(--w); animation: grow 1.1s ease .1s both; }
@keyframes grow { from { width: 0; } }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin-top: .6rem; }
.step { padding: .8rem .9rem; border-radius: 14px; border: 1px solid var(--line); background: rgba(255,255,255,.04); font-size: .9rem; color: var(--muted); }
.step b { display: block; color: var(--text); margin-bottom: .15rem; }
.foot { margin-top: 2rem; text-align: center; color: var(--muted); font-size: .82rem; }

@media (max-width: 560px) {
  .block-container {
    padding: 1rem 0.85rem 2rem;
  }

  .hero {
    padding: 0.8rem 0 1.2rem;
  }

  .hero h1 {
    font-size: 2.3rem;
    line-height: 1.12;
  }

  .typer {
    width: auto;
    white-space: normal;
    animation: none;
    border-right: none;
    font-size: 0.82rem;
  }

  div[data-testid="stForm"] {
    padding: 1rem;
    border-radius: 16px;
  }

  .verdict {
    padding: 1.2rem 1rem;
    border-radius: 18px;
  }

  .v-row {
    flex-direction: column;
    align-items: center;
    gap: 1.2rem;
  }

  .v-text {
    width: 100%;
    flex-basis: auto;
  }

  .stamp {
    font-size: clamp(1.6rem, 8vw, 2.2rem);
  }

  .ring {
    width: 125px;
    height: 125px;
    flex: 0 0 125px;
    margin: 0 auto;
  }

  .ring svg {
    width: 125px;
    height: 125px;
  }

  .panel {
    padding: 1rem;
  }

  .bars .row {
    grid-template-columns: 52px minmax(0, 1fr) 38px;
    gap: 0.45rem;
    font-size: 0.8rem;
  }

  .steps {
    grid-template-columns: 1fr;
  }
}
@media (prefers-reduced-motion: reduce) {
  .stApp *, .stApp *::before, .stApp *::after { animation: none !important; transition: none !important; }
  .confetti { display: none; }
}
</style>
"""


def h(*parts):
    """Join HTML pieces on one line so Markdown never turns indentation into code blocks."""
    return " ".join(line.strip() for line in "".join(parts).splitlines() if line.strip())


def hero():
    return h("""
    <div class="hero">
      <h1>Will you get <span class="grad">placed?</span></h1>
      <span class="typer">Two numbers in. One verdict out.</span>
      <p class="sub">Enter an IQ score and a CGPA. A logistic regression model trained on 100 students
      estimates whether that student gets placed, and how sure it is.</p>
    </div>""")


def waiting_card():
    return h("""
    <div class="waiting"><span class="big">🎯</span>
      Your verdict appears here. Fill in both fields and press <b style="color:#eeeaff">Check my placement</b>.
    </div>""")


def probability_chart(xs, ps, cgpa, p_now):
    W, H, L, R, T, B = 560, 240, 44, 16, 14, 40
    sx = lambda x: L + (x / 10.0) * (W - L - R)
    sy = lambda p: T + (1 - p) * (H - T - B)
    path = "M" + " L".join(f"{sx(x):.1f},{sy(p):.1f}" for x, p in zip(xs, ps))
    ticks = "".join(
        f'<text x="{sx(v):.1f}" y="{H - 22}" text-anchor="middle">{v}</text>' for v in range(0, 11, 2)
    )
    yt = "".join(
        f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>' for v in (0, 0.5, 1)
    )
    color = "#2ee6a6" if p_now >= 0.5 else "#ff5d73"
    return h(f"""
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Placement probability against CGPA">
      <defs><linearGradient id="cg" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0" stop-color="#ff5d73"/><stop offset=".5" stop-color="#ffc857"/><stop offset="1" stop-color="#2ee6a6"/>
      </linearGradient></defs>
      <line class="axis" x1="{L}" y1="{sy(0)}" x2="{W - R}" y2="{sy(0)}"/>
      <line class="guide" x1="{L}" y1="{sy(.5)}" x2="{W - R}" y2="{sy(.5)}"/>
      <path class="curve" pathLength="1" d="{path}"/>
      <line class="guide" x1="{sx(cgpa)}" y1="{sy(p_now)}" x2="{sx(cgpa)}" y2="{sy(0)}"/>
      <circle class="dot" cx="{sx(cgpa)}" cy="{sy(p_now)}" r="6" fill="{color}" stroke="#0d0a1f" stroke-width="2"/>
      {ticks}{yt}
      <text x="{(W + L) / 2}" y="{H - 4}" text-anchor="middle">CGPA</text>
    </svg>""")


def verdict_card(label, prob, cgpa, iq, tip, run_id):
    placed = label == 1
    pct = int(round(prob * 100))
    off = 327 * (1 - prob)
    cls = "yes" if placed else "no"
    stamp = "PLACED" if placed else "NOT YET"
    if placed:
        msg = (f"The model predicts this student <b>will be placed</b> with a CGPA of <b>{cgpa:.2f}</b> "
               f"and an IQ of <b>{iq:.0f}</b>.")
    else:
        msg = (f"The model predicts this student <b>will not be placed</b> with a CGPA of <b>{cgpa:.2f}</b> "
               f"and an IQ of <b>{iq:.0f}</b>. That can change.")
    confetti = ""
    if placed:
        rnd = random.Random(run_id)
        colors = ["#2ee6a6", "#8b6cff", "#ffc857", "#ff5d73", "#eeeaff"]
        confetti = '<div class="confetti" aria-hidden="true">' + "".join(
            f'<i style="left:{rnd.randint(2, 98)}%;background:{rnd.choice(colors)};'
            f'animation-delay:{rnd.uniform(0, .9):.2f}s;animation-duration:{rnd.uniform(2.2, 3.4):.2f}s"></i>'
            for _ in range(36)
        ) + "</div>"
    chips = f'<span class="chip">{tip}</span>'
    if not (TRAIN_CGPA[0] <= cgpa <= TRAIN_CGPA[1]):
        chips += (f'<span class="chip warn">CGPA is outside the training range '
                  f'({TRAIN_CGPA[0]} to {TRAIN_CGPA[1]}), so treat this with extra care</span>')
    if not (TRAIN_IQ[0] <= iq <= TRAIN_IQ[1]):
        chips += '<span class="chip warn">IQ is outside the training range</span>'
    animation_css = f"""
    <style>
    @keyframes pop_{run_id} {{
      from {{ transform: scale(.94); opacity: 0; }}
      to {{ transform: scale(1); opacity: 1; }}
    }}

    @keyframes shake_{run_id} {{
      20% {{ transform: translateX(-8px); }}
      40% {{ transform: translateX(7px); }}
      60% {{ transform: translateX(-4px); }}
      80% {{ transform: translateX(2px); }}
    }}

    @keyframes stamp_{run_id} {{
      from {{ transform: rotate(-18deg) scale(3); opacity: 0; }}
      to {{ transform: rotate(-5deg) scale(1); opacity: 1; }}
    }}

    @keyframes fill_{run_id} {{
      from {{ stroke-dashoffset: 327; }}
      to {{ stroke-dashoffset: {off:.1f}; }}
    }}

    @keyframes count_{run_id} {{
      from {{ --n: 0; }}
      to {{ --n: {pct}; }}
    }}

    @keyframes fall_{run_id} {{
      0% {{ transform: translateY(0) rotate(0); opacity: 1; }}
      100% {{ transform: translateY(420px) rotate(560deg); opacity: 0; }}
    }}
    </style>
    """
    card_animation = f"pop_{run_id}" if placed else f"shake_{run_id}"
    stamp_html = (
        f'<div class="stamp" style="animation-name:stamp_{run_id}">'
        f'{stamp}</div>'
    )
    confetti = confetti.replace(
        "animation-duration:",
        f"animation-name:fall_{run_id};animation-duration:"
    ) if confetti else ""

    return h(f"""
    {animation_css}
    <div class="verdict {cls}" role="status"
         style="animation-name:{card_animation}">
      {confetti}
      <div class="v-row">
        <div class="v-text">
          {stamp_html}
          <p class="v-msg">{msg}</p>
        </div>
        <div class="ring"
             aria-label="Estimated placement probability {pct} percent">
          <svg viewBox="0 0 120 120">
            <circle class="track" cx="60" cy="60" r="52"/>
            <circle class="bar" cx="60" cy="60" r="52"
                    style="--off:{off:.1f};animation-name:fill_{run_id}"/>
          </svg>
          <div class="pct"
               style="--n:{pct};animation-name:count_{run_id}"></div>
        </div>
      </div>
      <div class="chips">{chips}</div>
    </div>""")


def about_section(model):
    raw = np.ravel(model.coef_)
    coef = np.abs(raw)
    share = coef / coef.sum() * 100 if coef.sum() else coef
    rows = "".join(
        f'<div class="row"><span>{name}</span><div class="track"><div class="fillbar" style="--w:{s:.1f}%"></div></div>'
        f'<span>{s:.0f}%</span></div>'
        for name, s in zip(["CGPA", "IQ"], share)
    )
    note = ""
    if len(raw) == 2 and share[0] > 80:
        note = ("In this dataset the model leans almost entirely on <b>CGPA</b>. IQ barely moves the result, "
                "which is worth mentioning if someone asks how the model behaves.")
    return rows, note


def scatter_chart(model, scaler, df):
    W, H, L, R, T, B = 560, 300, 44, 16, 12, 40
    cx0, cx1 = 3, 9
    iy0, iy1 = 30, 240
    sx = lambda c: L + (c - cx0) / (cx1 - cx0) * (W - L - R)
    sy = lambda i: T + (1 - (i - iy0) / (iy1 - iy0)) * (H - T - B)
    dots = "".join(
        f'<circle cx="{sx(c):.1f}" cy="{sy(i):.1f}" r="4.5" fill="{"#2ee6a6" if p else "#ff5d73"}" fill-opacity=".85"/>'
        for c, i, p in zip(df["cgpa"], df["iq"], df["placement"])
    )
    # Decision boundary: w1*z_cgpa + w2*z_iq + b = 0, drawn in original units.
    w1, w2 = np.ravel(model.coef_)
    b = float(np.ravel(model.intercept_)[0])
    m1, m2 = scaler.mean_
    s1, s2 = scaler.scale_
    line = ""
    if abs(w1) > 1e-9:
        def c_at(i):
            return s1 * (-(b + w2 * (i - m2) / s2) / w1) + m1
        line = (f'<line x1="{sx(c_at(iy0)):.1f}" y1="{sy(iy0):.1f}" x2="{sx(c_at(iy1)):.1f}" y2="{sy(iy1):.1f}" '
                f'stroke="#ffc857" stroke-width="2.5" stroke-dasharray="6 5"/>')
    xt = "".join(f'<text x="{sx(v):.1f}" y="{H - 24}" text-anchor="middle">{v}</text>' for v in range(3, 10))
    yt = "".join(f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>' for v in range(50, 241, 50))
    return h(f"""
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Training data with decision boundary">
      <line class="axis" x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}"/><line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{H - B}"/>
      {dots}{line}{xt}{yt}
      <text x="{(W + L) / 2}" y="{H - 6}" text-anchor="middle">CGPA</text>
      <text x="4" y="{T + 8}">IQ</text>
    </svg>""")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(hero(), unsafe_allow_html=True)

model, scaler = load_artifacts()
if model is None or scaler is None:
    st.error("Model files are missing. Add both `model.pkl` and `scaler.pkl` to the repository root.")
    st.markdown("**Create the files:** run `Basic_toy_project.ipynb` to the last cell (it saves both files), "
                "then upload them to GitHub next to `app.py`.")
    st.stop()

tab_predict, tab_about = st.tabs(["Predict", "How it works"])

if "report_count" not in st.session_state:
    st.session_state.report_count = 0

with tab_predict:
    with st.form("predict_form"):
        st.subheader("Student details")
        c1, c2 = st.columns(2)
        with c1:
            iq = st.number_input("IQ score", min_value=1.0, max_value=200.0, value=100.0, step=1.0)
        with c2:
            cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1, format="%.2f")
        submitted = st.form_submit_button("Check my placement")

    if not submitted:
        st.markdown(waiting_card(), unsafe_allow_html=True)
    else:
        st.session_state.report_count += 1
        run_id = st.session_state.report_count

        try:
            label, prob = predict_one(model, scaler, cgpa, iq)
            xs, ps = probability_curve(model, scaler, iq)
            tip_cg = tipping_cgpa(xs, ps)
        except Exception as exc:  # e.g. scikit-learn version mismatch with the pickles
            st.error("The model could not make a prediction. Check that `requirements.txt` pins "
                     "`scikit-learn==1.6.1`, the version used to create the .pkl files.")
            st.caption(f"Technical detail: {exc}")
            st.stop()

        if tip_cg is None:
            tip = f"At IQ {iq:.0f}, no CGPA up to 10 tips the model to placed"
        elif tip_cg <= 0:
            tip = f"At IQ {iq:.0f}, the model says placed at any CGPA"
        elif label == 1:
            tip = f"Tipping point at IQ {iq:.0f}: CGPA {tip_cg:.1f}. You are {cgpa - tip_cg:+.2f} above it"
        else:
            tip = f"Tipping point at IQ {iq:.0f}: CGPA {tip_cg:.1f}. You need about {max(tip_cg - cgpa, 0):.2f} more"

        st.markdown(
            verdict_card(label, prob, cgpa, iq, tip, run_id),
            unsafe_allow_html=True
        )
        if label == 1:
            st.balloons()
        st.markdown(
            h(f"""<div class="panel chart"><h4>How the CGPA changes your odds</h4>
            <p>The curve shows the model's placement probability at IQ {iq:.0f}. The dashed line marks 50%, where the verdict flips.</p>
            {probability_chart(xs, ps, cgpa, prob)}</div>"""),
            unsafe_allow_html=True,
        )
        st.caption("This is a model estimate, not a guarantee. Results depend on the training data.")

with tab_about:
    rows, note = about_section(model)
    st.markdown(
        h(f"""<div class="panel"><h4>What the model pays attention to</h4>
        <div class="bars">{rows}</div><p>{note}</p></div>"""),
        unsafe_allow_html=True,
    )
    for name in ("placement (1).csv", "placement__1_.csv", "placement.csv"):
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p).iloc[:, 1:]
                st.markdown(
                    h(f"""<div class="panel chart"><h4>The 100 students it learned from</h4>
                    <p>Green dots were placed, red dots were not. The amber dashed line is the model's decision boundary.</p>
                    {scatter_chart(model, scaler, df)}</div>"""),
                    unsafe_allow_html=True,
                )
            except Exception:
                pass  # the chart is optional; never break the page over it
            break
    st.markdown(
        h("""<div class="panel"><h4>The pipeline</h4><div class="steps">
        <div class="step"><b>Data</b>100 students with CGPA, IQ and a placed / not placed label</div>
        <div class="step"><b>Split</b>90 for training, 10 for testing</div>
        <div class="step"><b>Scale</b>StandardScaler puts both inputs on the same footing</div>
        <div class="step"><b>Train</b>Logistic regression learns the boundary</div>
        <div class="step"><b>Serve</b>model.pkl and scaler.pkl power this page</div>
        </div><p>The notebook reports 90% accuracy, but that is 9 correct out of only 10 test students,
        so it is a rough guide, not a promise.</p></div>"""),
        unsafe_allow_html=True,
    )

st.markdown('<div class="foot">A learning project built with Python, scikit-learn and Streamlit. '
            'Not a real-world placement guarantee.</div>', unsafe_allow_html=True)
