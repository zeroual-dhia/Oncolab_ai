#!/usr/bin/env python3
"""
OncoLab AI — FOLFOX Complication Risk Demo
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import gradio as gr

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# ── Constants ──────────────────────────────────────────────────────────────────
TARGETS = [
    "anemia", "neutropenia", "thrombocytopenia",
    "renal_toxicity", "hepatic_toxicity",
]
TARGET_LABELS = {
    "anemia":           "Anemia",
    "neutropenia":      "Neutropenia",
    "thrombocytopenia": "Thrombocytopenia",
    "renal_toxicity":   "Renal Toxicity",
    "hepatic_toxicity": "Hepatic Toxicity",
}
TARGET_ICONS = {
    "anemia":           "🩸",
    "neutropenia":      "🦠",
    "thrombocytopenia": "🔵",
    "renal_toxicity":   "🫘",
    "hepatic_toxicity": "🟡",
}
ETAT_OPTIONS  = ["Normal", "Underweight", "Overweight"]
ETAT_MAP      = {"Underweight": 0, "Normal": 1, "Overweight": 2}
MODEL_NAMES   = ["RandomForest", "XGBoost", "LightGBM"]

RISK_LOW  = 0.30
RISK_HIGH = 0.60
COLOR_LOW  = "#10b981"
COLOR_MED  = "#f59e0b"
COLOR_HIGH = "#ef4444"

EXAMPLE = dict(
    age=67, sex_M=True, bmi=25.9, etat="Overweight",
    hta=True, diabetes=False, renal_disease=False, hepatic_disease=False,
    T_stage=3.0, N_stage=2.0, M_stage=1, has_metastasis=True,
    treatment_neoadj=True,
    cycle_num=3, interval_days=21,
    dose_oxali=200.0, dose_5fu_bolus=0.0, dose_5fu_cont=0.0, dose_af=0.0,
    hb=8.7,   neut=9.13,  plq=427.0, uree=0.23, creat=10.2,
    asat=5.0, alat=20.0,  bil=4.0,
    hb_t1=8.7,   neut_t1=7.27, plq_t1=343.0, creat_t1=9.7,
    anemia_t1=True, neutropenia_t1=False, thrombocytopenia_t1=False,
    renal_toxicity_t1=False, hepatic_toxicity_t1=False,
    hb_t2=10.2, neut_t2=7.11, plq_t2=389.0, creat_t2=11.0,
    anemia_t2=True, neutropenia_t2=False, thrombocytopenia_t2=False,
    renal_toxicity_t2=False, hepatic_toxicity_t2=False,
)

# ── Load models ────────────────────────────────────────────────────────────────
def _load_models():
    out = {}
    for target in TARGETS:
        out[target] = {}
        for mname in MODEL_NAMES:
            p = MODELS_DIR / f"{target}_{mname}.pkl"
            if p.exists():
                out[target][mname] = joblib.load(p)
    return out

LOADED = _load_models()

# ── Prediction logic ───────────────────────────────────────────────────────────
def _build_feature_vector(inputs: dict) -> np.ndarray:
    hb, neut, plq       = inputs["hb"],   inputs["neut"],   inputs["plq"]
    hb_t1, neut_t1, plq_t1 = inputs["hb_t1"], inputs["neut_t1"], inputs["plq_t1"]
    hb_t2, neut_t2, plq_t2 = inputs["hb_t2"], inputs["neut_t2"], inputs["plq_t2"]

    delta_hb   = hb   - hb_t1
    delta_neut = neut - neut_t1
    delta_plq  = plq  - plq_t1
    trend_neut = float(np.mean([neut, neut_t1, neut_t2]))
    drop_rate  = (plq - plq_t2) / plq_t2 if plq_t2 != 0 else 0.0
    var_neut   = float(np.std([neut, neut_t1, neut_t2]))

    comorbidity_count = (
        int(inputs["hta"]) + int(inputs["diabetes"])
        + int(inputs["renal_disease"]) + int(inputs["hepatic_disease"])
    )
    row = {
        "age": inputs["age"], "sex_M": int(inputs["sex_M"]),
        "bmi": inputs["bmi"], "etat": ETAT_MAP.get(inputs["etat"], 1),
        "hta": int(inputs["hta"]), "diabetes": int(inputs["diabetes"]),
        "renal_disease": int(inputs["renal_disease"]),
        "hepatic_disease": int(inputs["hepatic_disease"]),
        "comorbidity_count": comorbidity_count,
        "T_stage": inputs["T_stage"], "N_stage": inputs["N_stage"],
        "M_stage": inputs["M_stage"], "has_metastasis": int(inputs["has_metastasis"]),
        "cycle_num": inputs["cycle_num"], "interval_days": inputs["interval_days"],
        "dose_oxali": inputs["dose_oxali"], "dose_5fu_bolus": inputs["dose_5fu_bolus"],
        "dose_5fu_cont": inputs["dose_5fu_cont"], "dose_af": inputs["dose_af"],
        "hb": hb, "neut": neut, "plq": plq,
        "uree": inputs["uree"], "creat": inputs["creat"],
        "asat": inputs["asat"], "alat": inputs["alat"], "bil": inputs["bil"],
        "hb_t1": hb_t1, "neut_t1": neut_t1, "plq_t1": plq_t1, "creat_t1": inputs["creat_t1"],
        "hb_t2": hb_t2, "neut_t2": neut_t2, "plq_t2": plq_t2, "creat_t2": inputs["creat_t2"],
        "anemia_t1": int(inputs["anemia_t1"]), "neutropenia_t1": int(inputs["neutropenia_t1"]),
        "thrombocytopenia_t1": int(inputs["thrombocytopenia_t1"]),
        "renal_toxicity_t1": int(inputs["renal_toxicity_t1"]),
        "hepatic_toxicity_t1": int(inputs["hepatic_toxicity_t1"]),
        "anemia_t2": int(inputs["anemia_t2"]), "neutropenia_t2": int(inputs["neutropenia_t2"]),
        "thrombocytopenia_t2": int(inputs["thrombocytopenia_t2"]),
        "renal_toxicity_t2": int(inputs["renal_toxicity_t2"]),
        "hepatic_toxicity_t2": int(inputs["hepatic_toxicity_t2"]),
        "delta_hb": delta_hb, "delta_neut": delta_neut, "delta_plq": delta_plq,
        "trend_neut_3cycles": trend_neut, "drop_rate_plq": drop_rate,
        "variability_neut": var_neut,
        "treatment_intent_Neoadj": int(inputs["treatment_neoadj"]),
    }
    feature_order = LOADED[TARGETS[0]][MODEL_NAMES[0]]["features"]
    return np.array([row[f] for f in feature_order]).reshape(1, -1)


def _predict_all(X: np.ndarray) -> dict:
    results = {}
    for target in TARGETS:
        probas = [pkg["model"].predict_proba(pkg["scaler"].transform(X))[0, 1]
                  for pkg in LOADED[target].values()]
        results[target] = float(np.mean(probas))
    return results


def _risk_class(p: float) -> str:
    return "high" if p >= RISK_HIGH else ("mod" if p >= RISK_LOW else "low")


def _risk_label(p: float) -> str:
    return "High" if p >= RISK_HIGH else ("Moderate" if p >= RISK_LOW else "Low")


def _risk_color(p: float) -> str:
    return COLOR_HIGH if p >= RISK_HIGH else (COLOR_MED if p >= RISK_LOW else COLOR_LOW)


# ── Enhanced dark chart ────────────────────────────────────────────────────────
def _make_chart(probs: dict) -> plt.Figure:
    labels = [TARGET_LABELS[t] for t in TARGETS]
    values = [probs[t] for t in TARGETS]
    colors = [_risk_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Background track
    ax.barh(labels[::-1], [100] * 5, color="#161b22", height=0.58, zorder=1,
            left=0, linewidth=0)
    # Glow halo
    ax.barh(labels[::-1], [v * 100 for v in values[::-1]],
            color=colors[::-1], height=0.72, zorder=2, alpha=0.18, linewidth=0)
    # Main bars
    bars = ax.barh(labels[::-1], [v * 100 for v in values[::-1]],
                   color=colors[::-1], height=0.50, zorder=3,
                   linewidth=0, alpha=0.95)

    for bar, val, col in zip(bars, values[::-1], colors[::-1]):
        pct = val * 100
        ax.text(min(bar.get_width() + 1.5, 100), bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", ha="left",
                fontsize=12, fontweight="bold", color=col, zorder=4)

    ax.axvline(RISK_LOW * 100,  color="#2d3748", ls="--", lw=1.2, zorder=5)
    ax.axvline(RISK_HIGH * 100, color="#2d3748", ls="--", lw=1.2, zorder=5)
    ax.text(RISK_LOW * 100 - 0.5, -0.75, "30%",  fontsize=7.5, color="#4b5563", ha="center")
    ax.text(RISK_HIGH * 100 - 0.5, -0.75, "60%", fontsize=7.5, color="#4b5563", ha="center")

    ax.set_xlim(0, 115)
    ax.set_title("Next-cycle complication risk  (RF · XGB · LGB ensemble)",
                 fontsize=11, fontweight="bold", color="#e2e8f0", pad=14)
    ax.set_xlabel("Risk probability (%)", fontsize=9, color="#6b7280", labelpad=8)
    ax.tick_params(axis="y", labelsize=10, colors="#cbd5e1", pad=6)
    ax.tick_params(axis="x", labelsize=8,  colors="#4b5563")
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#1a2332", linewidth=0.6, zorder=0)

    patches = [
        mpatches.Patch(color=COLOR_LOW,  label="Low  (<30%)"),
        mpatches.Patch(color=COLOR_MED,  label="Moderate (30–60%)"),
        mpatches.Patch(color=COLOR_HIGH, label="High (>60%)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.4, facecolor="#0d1117", edgecolor="#2d3748",
              labelcolor="#9ca3af")

    plt.tight_layout(pad=1.5)
    return fig


# ── HTML risk cards ────────────────────────────────────────────────────────────
def _make_results_html(probs: dict) -> str:
    high_names = [TARGET_LABELS[t] for t in TARGETS if probs[t] >= RISK_HIGH]
    mod_names  = [TARGET_LABELS[t] for t in TARGETS if RISK_LOW <= probs[t] < RISK_HIGH]

    if high_names:
        alert = (f'<div class="alert danger">'
                 f'<span class="alert-icon">⚠</span>'
                 f'<div><strong>High risk detected:</strong> {", ".join(high_names)}'
                 f'<br><small>Immediate clinical attention recommended.</small></div></div>')
    elif mod_names:
        alert = (f'<div class="alert warning">'
                 f'<span class="alert-icon">⚡</span>'
                 f'<div><strong>Moderate risk:</strong> {", ".join(mod_names)}'
                 f'<br><small>Close monitoring advised before next cycle.</small></div></div>')
    else:
        alert = ('<div class="alert ok">'
                 '<span class="alert-icon">✓</span>'
                 '<div><strong>All complications at low risk</strong>'
                 '<br><small>Continue standard protocol — reassess at next cycle.</small></div></div>')

    cards = ""
    for t in TARGETS:
        p    = probs[t]
        pct  = p * 100
        cls  = _risk_class(p)
        lv   = _risk_label(p)
        icon = TARGET_ICONS[t]
        bar_w = min(pct, 100)
        cards += f"""
        <div class="rcard {cls}">
            <div class="rcard-glow"></div>
            <div class="rcard-icon">{icon}</div>
            <div class="rcard-name">{TARGET_LABELS[t]}</div>
            <div class="rcard-pct">{pct:.0f}<span class="rcard-unit">%</span></div>
            <div class="rbar-wrap">
                <div class="rbar" style="width:{bar_w:.1f}%"></div>
            </div>
            <span class="rbadge">{lv}</span>
        </div>"""

    return f"""
    <div class="results-wrap">
        {alert}
        <div class="rcard-grid">{cards}</div>
        <div class="rdisclaimer">
            🤖 <strong>Ensemble model</strong>: RandomForest · XGBoost · LightGBM
            &nbsp;·&nbsp; ⚕ For clinical decision support — does not replace physician judgement.
        </div>
    </div>"""


# ── Gradio predict wrapper ─────────────────────────────────────────────────────
def predict(
    age, sex_M, bmi, etat,
    hta, diabetes, renal_disease, hepatic_disease,
    T_stage, N_stage, M_stage, has_metastasis, treatment_neoadj,
    cycle_num, interval_days,
    dose_oxali, dose_5fu_bolus, dose_5fu_cont, dose_af,
    hb, neut, plq, uree, creat, asat, alat, bil,
    hb_t1, neut_t1, plq_t1, creat_t1,
    anemia_t1, neutropenia_t1, thrombocytopenia_t1, renal_toxicity_t1, hepatic_toxicity_t1,
    hb_t2, neut_t2, plq_t2, creat_t2,
    anemia_t2, neutropenia_t2, thrombocytopenia_t2, renal_toxicity_t2, hepatic_toxicity_t2,
):
    inputs = dict(
        age=age, sex_M=sex_M, bmi=bmi, etat=etat,
        hta=hta, diabetes=diabetes, renal_disease=renal_disease, hepatic_disease=hepatic_disease,
        T_stage=T_stage, N_stage=N_stage, M_stage=M_stage,
        has_metastasis=has_metastasis, treatment_neoadj=treatment_neoadj,
        cycle_num=cycle_num, interval_days=interval_days,
        dose_oxali=dose_oxali, dose_5fu_bolus=dose_5fu_bolus,
        dose_5fu_cont=dose_5fu_cont, dose_af=dose_af,
        hb=hb, neut=neut, plq=plq, uree=uree, creat=creat,
        asat=asat, alat=alat, bil=bil,
        hb_t1=hb_t1, neut_t1=neut_t1, plq_t1=plq_t1, creat_t1=creat_t1,
        anemia_t1=anemia_t1, neutropenia_t1=neutropenia_t1,
        thrombocytopenia_t1=thrombocytopenia_t1,
        renal_toxicity_t1=renal_toxicity_t1, hepatic_toxicity_t1=hepatic_toxicity_t1,
        hb_t2=hb_t2, neut_t2=neut_t2, plq_t2=plq_t2, creat_t2=creat_t2,
        anemia_t2=anemia_t2, neutropenia_t2=neutropenia_t2,
        thrombocytopenia_t2=thrombocytopenia_t2,
        renal_toxicity_t2=renal_toxicity_t2, hepatic_toxicity_t2=hepatic_toxicity_t2,
    )
    X          = _build_feature_vector(inputs)
    probs      = _predict_all(X)
    chart      = _make_chart(probs)
    cards_html = _make_results_html(probs)
    return chart, cards_html


# ── Custom CSS ─────────────────────────────────────────────────────────────────
CSS = """
/* ── Reset / base ─────────────────────────────────────── */
.gradio-container {
    background: linear-gradient(160deg, #060d1f 0%, #0a1628 60%, #060d1f 100%) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    min-height: 100vh;
}
* { box-sizing: border-box; }

/* ── App header ──────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2.2rem 1rem 1.8rem;
    background: linear-gradient(135deg,
        rgba(0,212,255,0.07) 0%,
        rgba(99,102,241,0.07) 50%,
        rgba(167,139,250,0.07) 100%);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.app-header .pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(99,102,241,0.15));
    border: 1px solid rgba(0,212,255,0.25);
    color: #67e8f9;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.app-header h1 {
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #00d4ff 0%, #818cf8 50%, #a78bfa 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.5rem !important;
    letter-spacing: -0.5px;
    line-height: 1.15 !important;
}
.app-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
}
.app-header .model-pills {
    margin-top: 0.9rem;
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.app-header .mpill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #64748b;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Section header labels ───────────────────────────── */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.2rem 0 !important;
    letter-spacing: 0.3px;
}
.section-icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(99,102,241,0.2));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}

/* ── Accordions ──────────────────────────────────────── */
.gr-accordion, details {
    background: rgba(13, 20, 38, 0.85) !important;
    border: 1px solid rgba(30, 58, 95, 0.5) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 0.6rem !important;
    overflow: hidden !important;
}
details > summary {
    padding: 0.85rem 1.1rem !important;
    cursor: pointer;
}
details > summary span {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── Inputs ──────────────────────────────────────────── */
.gr-form, .gr-box { background: transparent !important; }
.gr-input-label span, label span, .svelte-1gfkn6j {
    color: #64748b !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
input[type="number"], input[type="text"], select, .gr-select select {
    background: rgba(8, 14, 28, 0.9) !important;
    border: 1px solid rgba(30, 58, 95, 0.7) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.07) !important;
    outline: none !important;
}
.gr-checkbox-label, .svelte-eycpg label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* ── Hint text ───────────────────────────────────────── */
.hint-text {
    color: #475569;
    font-size: 0.76rem;
    padding: 0.4rem 0.8rem;
    background: rgba(0,212,255,0.03);
    border-left: 2px solid rgba(0,212,255,0.2);
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.2rem;
    line-height: 1.5;
}

/* ── Predict button ──────────────────────────────────── */
.predict-btn button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 60%, #8b5cf6 100%) !important;
    border: none !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    box-shadow: 0 4px 30px rgba(14,165,233,0.35) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
}
.predict-btn button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 10px 40px rgba(14,165,233,0.5) !important;
}
.predict-btn button:active {
    transform: translateY(0) !important;
}

/* ── Results: cards ──────────────────────────────────── */
.results-wrap { font-family: 'Inter', system-ui, sans-serif; }

.alert {
    display: flex; align-items: flex-start; gap: 0.9rem;
    padding: 0.85rem 1.1rem; border-radius: 12px;
    margin-bottom: 1.1rem; line-height: 1.5;
}
.alert.danger  { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.25);  color:#fca5a5; }
.alert.warning { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); color:#fcd34d; }
.alert.ok      { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); color:#6ee7b7; }
.alert-icon    { font-size:1.3rem; line-height:1; flex-shrink:0; margin-top:0.05rem; }
.alert strong  { font-weight: 700; font-size: 0.9rem; }
.alert small   { font-size: 0.78rem; opacity: 0.8; }

.rcard-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
@media (max-width: 900px) { .rcard-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { .rcard-grid { grid-template-columns: repeat(2, 1fr); } }

.rcard {
    background: rgba(13,20,38,0.95);
    border-radius: 16px;
    padding: 1.3rem 0.9rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.rcard:hover { transform: translateY(-4px); }
.rcard::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.rcard.low::after  { background: linear-gradient(90deg, #10b981, #34d399); }
.rcard.mod::after  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.rcard.high::after { background: linear-gradient(90deg, #dc2626, #ef4444, #f87171); }

.rcard.high {
    box-shadow: 0 0 28px rgba(239,68,68,0.22);
    animation: glow-red 2.2s ease-in-out infinite;
}
.rcard.mod { box-shadow: 0 0 18px rgba(245,158,11,0.15); }
.rcard.low { box-shadow: 0 0 12px rgba(16,185,129,0.08); }

@keyframes glow-red {
    0%,100% { box-shadow: 0 0 28px rgba(239,68,68,0.22); }
    50%      { box-shadow: 0 0 44px rgba(239,68,68,0.42); }
}

.rcard-glow {
    position: absolute; top: -30px; left: 50%; transform: translateX(-50%);
    width: 80px; height: 80px; border-radius: 50%;
    pointer-events: none;
}
.rcard.low  .rcard-glow { background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%); }
.rcard.mod  .rcard-glow { background: radial-gradient(circle, rgba(245,158,11,0.15) 0%, transparent 70%); }
.rcard.high .rcard-glow { background: radial-gradient(circle, rgba(239,68,68,0.20) 0%, transparent 70%); }

.rcard-icon { font-size: 1.55rem; margin-bottom: 0.45rem; line-height: 1; }
.rcard-name {
    color: #64748b; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.55rem;
}
.rcard-pct {
    font-size: 2rem; font-weight: 900; line-height: 1; margin-bottom: 0.6rem;
}
.rcard-unit { font-size: 1rem; font-weight: 600; opacity: 0.7; }
.rcard.low  .rcard-pct { color: #10b981; }
.rcard.mod  .rcard-pct { color: #f59e0b; }
.rcard.high .rcard-pct { color: #ef4444; }

.rbar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 999px; height: 5px;
    margin: 0 0.2rem 0.7rem; overflow: hidden;
}
.rbar { height: 100%; border-radius: 999px; }
.rcard.low  .rbar { background: linear-gradient(90deg, #059669, #10b981); }
.rcard.mod  .rbar { background: linear-gradient(90deg, #d97706, #f59e0b); }
.rcard.high .rbar { background: linear-gradient(90deg, #b91c1c, #ef4444, #f87171); }

.rbadge {
    display: inline-block; padding: 0.2rem 0.65rem;
    border-radius: 999px; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase;
}
.rcard.low  .rbadge { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.2); }
.rcard.mod  .rbadge { background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.2); }
.rcard.high .rbadge { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.2);  }

/* ── Disclaimer ──────────────────────────────────────── */
.rdisclaimer {
    text-align: center; color: #374151;
    font-size: 0.72rem; padding: 0.6rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    line-height: 1.6;
}
.rdisclaimer strong { color: #4b5563; }

/* ── Chart container ─────────────────────────────────── */
.chart-wrap {
    background: rgba(13,20,38,0.8);
    border: 1px solid rgba(30,58,95,0.4);
    border-radius: 16px;
    overflow: hidden;
    padding: 0.2rem;
}

/* ── Divider ─────────────────────────────────────────── */
.results-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 1.2rem 0 1rem;
}
.results-label {
    color: #1e3a5f;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 1rem;
}
"""

HEADER_HTML = """
<div class="app-header">
    <div class="pill">Clinical AI · Decision Support</div>
    <h1>OncoLab AI</h1>
    <p>
        FOLFOX chemotherapy complication risk predictor<br>
        Colorectal cancer · Next-cycle biological toxicity assessment
    </p>
    <div class="model-pills">
        <span class="mpill">Random Forest</span>
        <span class="mpill">XGBoost</span>
        <span class="mpill">LightGBM</span>
        <span class="mpill">Ensemble</span>
        <span class="mpill">5 targets</span>
        <span class="mpill">52 features</span>
    </div>
</div>
"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_ui():
    e = EXAMPLE
    with gr.Blocks(title="OncoLab AI — FOLFOX Risk Predictor", css=CSS) as demo:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            # ── LEFT: input form ─────────────────────────────────────────────
            with gr.Column(scale=5):

                # Section 1 — Patient Profile
                with gr.Accordion(label="👤  Patient Profile", open=True):
                    with gr.Row():
                        age  = gr.Number(label="Age (years)",   value=e["age"],  minimum=18, maximum=100, step=1)
                        bmi  = gr.Number(label="BMI (kg/m²)",   value=e["bmi"],  minimum=10, maximum=60)
                        etat = gr.Dropdown(ETAT_OPTIONS, label="Weight status", value=e["etat"])
                        sex_M = gr.Checkbox(label="Male sex", value=e["sex_M"])
                    with gr.Row():
                        hta             = gr.Checkbox(label="Hypertension",   value=e["hta"])
                        diabetes        = gr.Checkbox(label="Diabetes",       value=e["diabetes"])
                        renal_disease   = gr.Checkbox(label="Renal disease",  value=e["renal_disease"])
                        hepatic_disease = gr.Checkbox(label="Hepatic disease",value=e["hepatic_disease"])
                    with gr.Row():
                        T_stage          = gr.Number(label="T stage",  value=e["T_stage"],  minimum=1, maximum=4, step=1)
                        N_stage          = gr.Number(label="N stage",  value=e["N_stage"],  minimum=0, maximum=3, step=1)
                        M_stage          = gr.Number(label="M stage",  value=e["M_stage"],  minimum=0, maximum=1, step=1)
                        has_metastasis   = gr.Checkbox(label="Metastasis",         value=e["has_metastasis"])
                        treatment_neoadj = gr.Checkbox(label="Neoadjuvant intent", value=e["treatment_neoadj"])

                # Section 2 — Current Cycle
                with gr.Accordion(label="💊  Current Cycle (t) — Doses", open=True):
                    with gr.Row():
                        cycle_num     = gr.Number(label="Cycle no.", value=e["cycle_num"],     minimum=1, maximum=30, step=1)
                        interval_days = gr.Number(label="Interval (days)", value=e["interval_days"], minimum=1, maximum=60, step=1)
                    with gr.Row():
                        dose_oxali     = gr.Number(label="Oxaliplatin (mg)",    value=e["dose_oxali"],     minimum=0)
                        dose_5fu_bolus = gr.Number(label="5-FU bolus (mg)",     value=e["dose_5fu_bolus"], minimum=0)
                        dose_5fu_cont  = gr.Number(label="5-FU continuous (mg)",value=e["dose_5fu_cont"],  minimum=0)
                        dose_af        = gr.Number(label="Leucovorin (mg)",     value=e["dose_af"],        minimum=0)

                # Section 3 — Current biology
                with gr.Accordion(label="🔬  Biology — Current Cycle (t)", open=True):
                    gr.HTML('<div class="hint-text">Neutrophils & Platelets in <strong>G/L</strong> &nbsp;·&nbsp; Hb in <strong>g/dL</strong> &nbsp;·&nbsp; Creatinine in <strong>mg/dL</strong> &nbsp;·&nbsp; ASAT/ALAT in <strong>IU/L</strong></div>')
                    with gr.Row():
                        hb   = gr.Number(label="Hb (g/dL)",          value=e["hb"],   minimum=0)
                        neut = gr.Number(label="Neutrophils (G/L)",   value=e["neut"], minimum=0)
                        plq  = gr.Number(label="Platelets (G/L)",     value=e["plq"],  minimum=0)
                        uree = gr.Number(label="Urea (g/L)",          value=e["uree"], minimum=0)
                    with gr.Row():
                        creat = gr.Number(label="Creatinine (mg/dL)", value=e["creat"],minimum=0)
                        asat  = gr.Number(label="ASAT (IU/L)",        value=e["asat"], minimum=0)
                        alat  = gr.Number(label="ALAT (IU/L)",        value=e["alat"], minimum=0)
                        bil   = gr.Number(label="Bilirubin",          value=e["bil"],  minimum=0)

                # Section 4 — t-1
                with gr.Accordion(label="📋  Previous Cycle (t-1)", open=False):
                    with gr.Row():
                        hb_t1   = gr.Number(label="Hb t-1",               value=e["hb_t1"],   minimum=0)
                        neut_t1 = gr.Number(label="Neutrophils t-1 (G/L)",value=e["neut_t1"], minimum=0)
                        plq_t1  = gr.Number(label="Platelets t-1 (G/L)",  value=e["plq_t1"],  minimum=0)
                        creat_t1= gr.Number(label="Creatinine t-1",       value=e["creat_t1"],minimum=0)
                    gr.HTML('<div class="hint-text">Complications observed at cycle t-1?</div>')
                    with gr.Row():
                        anemia_t1           = gr.Checkbox(label="Anemia",         value=e["anemia_t1"])
                        neutropenia_t1      = gr.Checkbox(label="Neutropenia",    value=e["neutropenia_t1"])
                        thrombocytopenia_t1 = gr.Checkbox(label="Thrombocytopenia", value=e["thrombocytopenia_t1"])
                        renal_toxicity_t1   = gr.Checkbox(label="Renal toxicity", value=e["renal_toxicity_t1"])
                        hepatic_toxicity_t1 = gr.Checkbox(label="Hepatic toxicity",value=e["hepatic_toxicity_t1"])

                # Section 5 — t-2
                with gr.Accordion(label="📋  Two Cycles Ago (t-2)", open=False):
                    with gr.Row():
                        hb_t2   = gr.Number(label="Hb t-2",               value=e["hb_t2"],   minimum=0)
                        neut_t2 = gr.Number(label="Neutrophils t-2 (G/L)",value=e["neut_t2"], minimum=0)
                        plq_t2  = gr.Number(label="Platelets t-2 (G/L)",  value=e["plq_t2"],  minimum=0)
                        creat_t2= gr.Number(label="Creatinine t-2",       value=e["creat_t2"],minimum=0)
                    gr.HTML('<div class="hint-text">Complications observed at cycle t-2?</div>')
                    with gr.Row():
                        anemia_t2           = gr.Checkbox(label="Anemia",         value=e["anemia_t2"])
                        neutropenia_t2      = gr.Checkbox(label="Neutropenia",    value=e["neutropenia_t2"])
                        thrombocytopenia_t2 = gr.Checkbox(label="Thrombocytopenia", value=e["thrombocytopenia_t2"])
                        renal_toxicity_t2   = gr.Checkbox(label="Renal toxicity", value=e["renal_toxicity_t2"])
                        hepatic_toxicity_t2 = gr.Checkbox(label="Hepatic toxicity",value=e["hepatic_toxicity_t2"])

                # Predict button
                with gr.Row(elem_classes=["predict-btn"]):
                    predict_btn = gr.Button("⚡  Predict Next-Cycle Risk", variant="primary", size="lg")

            # ── RIGHT: results ────────────────────────────────────────────────
            with gr.Column(scale=4):
                gr.HTML('<div class="results-label">── Risk Assessment ──</div>')
                cards_out = gr.HTML(
                    value='<div style="color:#1e3a5f;text-align:center;padding:3rem 1rem;font-size:0.85rem;">'
                          '← Fill in patient data and click <strong style="color:#0ea5e9">Predict</strong>'
                          '</div>'
                )
                gr.HTML('<hr class="results-divider"><div class="results-label">── Probability Chart ──</div>')
                with gr.Group(elem_classes=["chart-wrap"]):
                    chart_out = gr.Plot(label="", show_label=False)

        # ── Wire up ────────────────────────────────────────────────────────────
        all_inputs = [
            age, sex_M, bmi, etat,
            hta, diabetes, renal_disease, hepatic_disease,
            T_stage, N_stage, M_stage, has_metastasis, treatment_neoadj,
            cycle_num, interval_days,
            dose_oxali, dose_5fu_bolus, dose_5fu_cont, dose_af,
            hb, neut, plq, uree, creat, asat, alat, bil,
            hb_t1, neut_t1, plq_t1, creat_t1,
            anemia_t1, neutropenia_t1, thrombocytopenia_t1,
            renal_toxicity_t1, hepatic_toxicity_t1,
            hb_t2, neut_t2, plq_t2, creat_t2,
            anemia_t2, neutropenia_t2, thrombocytopenia_t2,
            renal_toxicity_t2, hepatic_toxicity_t2,
        ]
        predict_btn.click(fn=predict, inputs=all_inputs, outputs=[chart_out, cards_out])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
