"""
app.py
------
Auto Neural Network Designer — Neural Architecture Search via Genetic Algorithms.
Classification only. Supports light / dark theme toggle.

Run with:  streamlit run app.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

from core.genetic_algorithm import (init_population, create_next_generation,
                                     genome_to_label)
from core.model_builder      import compute_fitness, build_final_model
from core.comparator         import (run_classification_baselines,
                                     evaluate_nas_model_classification)
from utils.preprocessor      import preprocess
from utils.plotter           import (plot_fitness_over_generations, plot_architecture,
                                     plot_comparison_classification,
                                     plot_training_curve, plot_rank_table)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto NAS — Neural Architecture Search",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Toggle button — top-right via columns
_hdr_l, _hdr_r = st.columns([8, 1])
with _hdr_r:
    _label = '🌙 Dark' if st.session_state.theme == 'light' else '☀️ Light'
    if st.button(_label, key='theme_toggle'):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

theme = st.session_state.theme  # 'light' or 'dark'

# ─────────────────────────────────────────────────────────────────────────────
# THEME-AWARE CSS
# ─────────────────────────────────────────────────────────────────────────────
if theme == 'dark':
    BG          = '#0a0e1a'
    CARD_BG     = '#161b2e'
    SIDEBAR_BG  = '#0d1117'
    BORDER      = '#21262d'
    TEXT        = '#c9d1d9'
    ACCENT      = '#00d4ff'
    ACCENT2     = '#7b2ff7'
    SUBTITLE_C  = '#58a6ff'
    INFO_BG     = '#0d1a2e'
    INFO_BORDER = '#1a3a5c'
    INFO_TEXT   = '#8bb8d4'
    LABEL_C     = '#8b949e'
    METRIC_BG   = '#161b2e'
    WINNER_BG   = 'linear-gradient(135deg,#003d1a,#001a3d)'
    WINNER_BC   = '#00ff88'
    WINNER_TEXT = '#c9d1d9'
    BEST_BG     = 'linear-gradient(135deg,#0d2137,#1a0d2e)'
    BEST_BC     = '#7b2ff7'
    BEST_BL     = '#00d4ff'
    BAR_BG      = '#21262d'
    GEN_C       = '#7b2ff7'
    BTN_BG      = 'linear-gradient(135deg,#00d4ff20,#7b2ff720)'
    BTN_BORDER  = '#00d4ff'
    BTN_TEXT    = '#00d4ff'
    BTN_HOVER   = 'linear-gradient(135deg,#00d4ff40,#7b2ff740)'
    PROG_CSS    = 'background: linear-gradient(90deg,#00d4ff,#7b2ff7) !important;'
    TITLE_GRAD  = 'linear-gradient(135deg,#00d4ff,#7b2ff7)'
else:
    BG          = '#f4f6fb'
    CARD_BG     = '#ffffff'
    SIDEBAR_BG  = '#ffffff'
    BORDER      = '#dde3ed'
    TEXT        = '#1a1a2e'
    ACCENT      = '#0077cc'
    ACCENT2     = '#6a0dad'
    SUBTITLE_C  = '#0055aa'
    INFO_BG     = '#e8f0fb'
    INFO_BORDER = '#b0c8e8'
    INFO_TEXT   = '#1a3a5c'
    LABEL_C     = '#555577'
    METRIC_BG   = '#ffffff'
    WINNER_BG   = 'linear-gradient(135deg,#e6f9ee,#e6f0ff)'
    WINNER_BC   = '#00aa55'
    WINNER_TEXT = '#1a1a2e'
    BEST_BG     = 'linear-gradient(135deg,#e8f0ff,#f0e8ff)'
    BEST_BC     = '#6a0dad'
    BEST_BL     = '#0077cc'
    BAR_BG      = '#dde3ed'
    GEN_C       = '#6a0dad'
    BTN_BG      = 'linear-gradient(135deg,#0077cc15,#6a0dad15)'
    BTN_BORDER  = '#0077cc'
    BTN_TEXT    = '#0077cc'
    BTN_HOVER   = 'linear-gradient(135deg,#0077cc30,#6a0dad30)'
    PROG_CSS    = 'background: linear-gradient(90deg,#0077cc,#6a0dad) !important;'
    TITLE_GRAD  = 'linear-gradient(135deg,#0077cc,#6a0dad)'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700;900&display=swap');
    html, body, [class*="css"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'JetBrains Mono', monospace;
    }}
    .main-title {{
        font-family: 'Orbitron', sans-serif; font-size: 2.4rem; font-weight: 900;
        background: {TITLE_GRAD};
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; letter-spacing: 2px; margin-bottom: 0.2rem;
    }}
    .subtitle {{
        text-align: center; color: {SUBTITLE_C}; font-size: 0.85rem;
        letter-spacing: 3px; text-transform: uppercase; margin-bottom: 2rem;
    }}
    .genome-card {{
        background: {CARD_BG}; border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT}; border-radius: 8px;
        padding: 12px 16px; margin: 6px 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
        color: {TEXT};
    }}
    .best-card {{
        background: {BEST_BG};
        border: 1px solid {BEST_BC}; border-left: 4px solid {BEST_BL};
        border-radius: 10px; padding: 16px 20px; margin: 10px 0;
        color: {TEXT};
    }}
    .winner-banner {{
        background: {WINNER_BG};
        border: 2px solid {WINNER_BC}; border-radius: 12px;
        padding: 20px 24px; margin: 16px 0; text-align: center;
    }}
    .info-box {{
        background: {INFO_BG}; border: 1px solid {INFO_BORDER};
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
        font-size: 0.8rem; color: {INFO_TEXT};
    }}
    .stat-box {{
        background: {CARD_BG}; border: 1px solid {BORDER};
        border-radius: 8px; padding: 14px; text-align: center;
    }}
    .stat-number {{ font-family: 'Orbitron', sans-serif; font-size: 1.5rem; color: {ACCENT}; font-weight: 700; }}
    .stat-label  {{ font-size: 0.75rem; color: {LABEL_C}; text-transform: uppercase; letter-spacing: 1px; }}
    .gen-header  {{
        font-family: 'Orbitron', sans-serif; font-size: 1rem; color: {GEN_C};
        letter-spacing: 2px; border-bottom: 1px solid {BORDER};
        padding-bottom: 6px; margin: 16px 0 10px 0;
    }}
    .section-title {{
        font-family: 'Orbitron', sans-serif; font-size: 1.1rem;
        color: {ACCENT}; letter-spacing: 1px; margin: 1.5rem 0 0.8rem 0;
    }}
    div[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG} !important; border-right: 1px solid {BORDER}; }}
    .stButton > button {{
        background: {BTN_BG};
        border: 1px solid {BTN_BORDER}; color: {BTN_TEXT};
        font-family: 'Orbitron', sans-serif; font-size: 0.85rem;
        letter-spacing: 2px; border-radius: 6px;
        padding: 0.6rem 1.5rem; width: 100%; transition: all 0.2s;
    }}
    .stButton > button:hover {{ background: {BTN_HOVER}; }}
    .stProgress > div > div {{ {PROG_CSS} }}
    div[data-testid="stMetric"] {{
        background: {METRIC_BG}; border: 1px solid {BORDER};
        border-radius: 8px; padding: 10px 14px;
    }}
    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {{
        color: {LABEL_C} !important; font-size: 0.78rem !important;
        letter-spacing: 1px; text-transform: uppercase;
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧬 AUTO NAS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Neural Architecture Search · Genetic Algorithms · Classification Benchmarking</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dataset")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    target_col    = None
    df            = None

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ {df.shape[0]} rows × {df.shape[1]} cols")
        target_col = st.selectbox("🎯 Target Column", df.columns.tolist())

    st.markdown("---")
    st.markdown("### 🧬 GA Parameters")

    st.markdown(f"""
    <div class="info-box">
    💡 <b>Recommended for best results:</b><br>
    Pop: 20 · Gen: 10 · Top-K: 5 · Mutation: 0.25 · Epochs: 20
    </div>
    """, unsafe_allow_html=True)

    pop_size      = st.slider("Population Size",   5,  30, 10)
    generations   = st.slider("Generations",       2,  20,  5)
    top_k         = st.slider("Top-K Selection",   2,  10,  4)
    mutation_rate = st.slider("Mutation Rate",     0.1, 0.9, 0.25, 0.05)
    epochs        = st.slider("Epochs per Model",  5,  40, 20)

    st.markdown("---")
    st.markdown("### 🔧 Advanced Options")
    use_smote = st.checkbox(
        "Apply SMOTE (fix class imbalance)",
        value=False,
        help="Requires: pip install imbalanced-learn"
    )

    st.markdown("---")
    run_button = st.button("🚀 LAUNCH EVOLUTION", disabled=(df is None))

# ─────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if df is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">🧬</div><div class="stat-label">Genetic Evolution</div>
            <br><small style="color:{LABEL_C}">Evolves architecture, activation, dropout &amp; optimizer simultaneously</small>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">📈</div><div class="stat-label">Training Curves</div>
            <br><small style="color:{LABEL_C}">Loss, accuracy and learning rate curves across training epochs</small>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-number">⚔️</div><div class="stat-label">Full Benchmark</div>
            <br><small style="color:{LABEL_C}">NAS vs up to 9 classical ML models — accuracy, F1, ROC-AUC &amp; compute time</small>
        </div>""", unsafe_allow_html=True)
    st.info("👈 Upload a CSV dataset from the sidebar to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Dataset Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",    df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Target",  target_col)

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_button:

    # ── STEP 1: Preprocess ────────────────────────────────────────────────────
    with st.spinner("⚙️ Preprocessing dataset..."):
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, num_classes, feature_names = \
                preprocess(df, target_col, 'classification', use_smote=use_smote)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    st.markdown(f"""<div class="genome-card">
    ✅ Preprocessing complete &nbsp;|&nbsp;
    Train: <b>{len(X_train)}</b> &nbsp;|&nbsp;
    Val: <b>{len(X_val)}</b> &nbsp;|&nbsp;
    Test: <b>{len(X_test)}</b> &nbsp;|&nbsp;
    Features: <b>{X_train.shape[1]}</b> &nbsp;|&nbsp;
    Classes: <b>{num_classes}</b>
    {"&nbsp;|&nbsp; SMOTE: <b>ON</b>" if use_smote else ""}
    </div>""", unsafe_allow_html=True)

    # ── STEP 2: GA Evolution ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🧬 PHASE 1 — GENETIC EVOLUTION</div>", unsafe_allow_html=True)
    st.caption(f"Evolving {pop_size} architectures × {generations} generations "
               f"| Search space: layers + activation + dropout + optimizer")

    population         = init_population(pop_size)
    generation_stats   = []
    global_best_genome = None
    global_best_score  = -np.inf

    progress_bar    = st.progress(0)
    status_text     = st.empty()
    gen_container   = st.container()
    chart_container = st.empty()

    for gen in range(1, generations + 1):
        scored_population = []
        gen_results       = []

        for idx, genome in enumerate(population):
            label = genome_to_label(genome)
            status_text.markdown(
                f"<div class='gen-header'>⚡ GEN {gen}/{generations} "
                f"— [{idx+1}/{len(population)}]: <code>{label}</code></div>",
                unsafe_allow_html=True
            )

            fitness, num_params, _ = compute_fitness(
                genome, X_train, y_train, X_val, y_val,
                'classification', num_classes, epochs=epochs
            )
            scored_population.append((genome, fitness))
            gen_results.append({
                'genome':  genome,
                'label':   label,
                'fitness': fitness,
                'params':  num_params
            })

            if fitness > global_best_score:
                global_best_score  = fitness
                global_best_genome = genome

        gen_results.sort(key=lambda x: x['fitness'], reverse=True)
        best_g = gen_results[0]['fitness']
        avg_g  = round(np.mean([r['fitness'] for r in gen_results]), 5)

        generation_stats.append({
            'generation':   gen,
            'best_fitness': best_g,
            'avg_fitness':  avg_g
        })

        with gen_container:
            st.markdown(
                f"<div class='gen-header'>📊 GENERATION {gen} RESULTS</div>",
                unsafe_allow_html=True
            )
            for rank, r in enumerate(gen_results):
                medal = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉" if rank == 2 else f"#{rank+1}"
                bar_w = int((r['fitness'] / max(global_best_score, 0.01)) * 100)
                st.markdown(f"""<div class="genome-card">
                {medal} &nbsp; <b>{r['label']}</b><br>
                Fitness: <span style="color:{ACCENT}"><b>{r['fitness']:.5f}</b></span>
                &nbsp;|&nbsp; Params: <span style="color:#e05c3a">{r['params']:,}</span>
                <div style="height:4px;background:{BAR_BG};border-radius:2px;margin-top:6px">
                <div style="height:4px;width:{bar_w}%;background:linear-gradient(90deg,{ACCENT},{ACCENT2});border-radius:2px"></div>
                </div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="best-card">
            🏆 <b>Best this gen:</b>
            <span style="color:{ACCENT}"><b>{genome_to_label(gen_results[0]['genome'])}</b></span><br>
            Fitness: <span style="color:{ACCENT2}"><b>{best_g:.5f}</b></span>
            &nbsp;|&nbsp; Population avg: <b>{avg_g:.5f}</b>
            </div>""", unsafe_allow_html=True)

        chart_container.pyplot(plot_fitness_over_generations(generation_stats, theme=theme))

        if gen < generations:
            population = create_next_generation(
                scored_population, pop_size, top_k, mutation_rate
            )
        progress_bar.progress(gen / generations)

    status_text.markdown(
        f"<div class='gen-header' style='color:{ACCENT}'>✅ EVOLUTION COMPLETE</div>",
        unsafe_allow_html=True
    )

    # ── STEP 3: Best Architecture ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🏆 BEST ARCHITECTURE FOUND</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Layers",       str(global_best_genome['layers']))
    c2.metric("Activation",   global_best_genome['activation'])
    c3.metric("Dropout",      global_best_genome['dropout'])
    c4.metric("Optimizer",    global_best_genome.get('optimizer', 'adam'))
    c5.metric("Best Fitness", f"{global_best_score:.5f}")

    st.pyplot(plot_architecture(global_best_genome, theme=theme))

    # ── STEP 4: Train Final NAS Model ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🔧 PHASE 2 — FINAL MODEL TRAINING (up to 100 epochs)</div>",
        unsafe_allow_html=True
    )
    st.caption("Best genome trained fully with Early Stopping + Learning Rate Scheduling.")

    with st.spinner("Training final model..."):
        X_final = np.concatenate([X_train, X_val])
        y_final = np.concatenate([y_train, y_val])
        nas_train_start = time.time()
        final_model, final_history = build_final_model(
            global_best_genome, X_final, y_final,
            'classification', num_classes, epochs=100
        )
        nas_total_train_time = round(time.time() - nas_train_start, 3)

    t0 = time.time()
    final_model.predict(X_test, verbose=0)
    nas_predict_time = round(time.time() - t0, 6)

    actual_epochs = len(final_history.history['loss'])
    st.markdown(f"""<div class="genome-card">
    ✅ Training finished &nbsp;|&nbsp;
    Ran <b>{actual_epochs}</b> epochs
    {"(early stopping triggered)" if actual_epochs < 100 else "(ran full 100 epochs)"}
    &nbsp;|&nbsp; Total train time: <b>{nas_total_train_time}s</b>
    &nbsp;|&nbsp; Inference time: <b>{nas_predict_time}s</b>
    </div>""", unsafe_allow_html=True)

    summary_lines = []
    final_model.summary(print_fn=lambda x: summary_lines.append(x))
    with st.expander("📋 NAS Model Architecture Summary", expanded=False):
        st.code("\n".join(summary_lines), language="text")

    # ── STEP 5: Training Curves ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>📈 PHASE 3 — TRAINING & VALIDATION CURVES</div>",
        unsafe_allow_html=True
    )
    st.caption("Loss, accuracy, and learning rate decay across all training epochs.")

    st.pyplot(plot_training_curve(final_history.history, 'classification', theme=theme))

    with st.expander("🔢 Epoch-by-Epoch Training Log", expanded=False):
        h = final_history.history
        lr_vals = h.get('lr', [])
        log_df = pd.DataFrame({
            'Epoch':      range(1, actual_epochs + 1),
            'Train Loss': [round(v, 5) for v in h['loss']],
            'Val Loss':   [round(v, 5) for v in h['val_loss']],
            'Train Acc':  [round(v, 5) for v in h['accuracy']],
            'Val Acc':    [round(v, 5) for v in h['val_accuracy']],
            'LR':         [round(float(v), 7) for v in lr_vals] if lr_vals else ['N/A'] * actual_epochs
        })
        st.dataframe(log_df, use_container_width=True)

    # ── STEP 6: NAS Test Results ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🧪 NAS MODEL — FINAL TEST SET RESULTS</div>",
        unsafe_allow_html=True
    )
    st.caption("Evaluated on the held-out test set — never seen during training or evolution.")

    nas_metrics = evaluate_nas_model_classification(
        final_model, X_test, y_test, num_classes
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",       f"{nas_metrics['accuracy']:.5f}")
    m2.metric("F1 Score",       f"{nas_metrics['f1']:.5f}")
    m3.metric("ROC-AUC",        f"{nas_metrics['roc_auc']:.5f}" if nas_metrics['roc_auc'] else "N/A (multiclass)")
    m4.metric("Inference Time", f"{nas_predict_time}s")

    # ── STEP 7: Baseline Comparison ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>⚔️ PHASE 4 — COMPARISON vs CLASSICAL ML</div>",
        unsafe_allow_html=True
    )
    st.caption("Same train/test split. Fair head-to-head comparison.")

    baseline_status = st.empty()
    with st.spinner("Training baseline models..."):
        baseline_status.info(
            "📚 Training: Logistic Regression · Random Forest · Extra Trees · "
            "Gradient Boosting · SVM · KNN · Decision Tree · Naive Bayes · XGBoost (if installed)"
        )
        baseline_results = run_classification_baselines(
            X_train, y_train, X_test, y_test, num_classes
        )
    baseline_status.empty()

    ok_baselines = [r for r in baseline_results if r['status'] == 'ok']
    st.success(f"✅ {len(ok_baselines)} baseline models trained successfully.")

    with st.expander("📋 Individual Baseline Results", expanded=False):
        bl_df = pd.DataFrame(ok_baselines).drop(columns=['status'], errors='ignore')
        st.dataframe(bl_df, use_container_width=True)

    # ── STEP 8: Comparison Chart ──────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>📊 PERFORMANCE COMPARISON CHART</div>",
        unsafe_allow_html=True
    )
    st.pyplot(plot_comparison_classification(baseline_results, nas_metrics, theme=theme))

    # ── STEP 9: Ranked Table ──────────────────────────────────────────────────
    st.markdown(
        "<div class='section-title'>🏅 FULL MODEL RANKING</div>",
        unsafe_allow_html=True
    )
    nas_param_count = final_model.count_params()
    nas_row         = {'model': '🧬 NAS (Ours)', 'predict_time': nas_predict_time,
                       'params': nas_param_count, **nas_metrics}
    all_comparison  = [nas_row] + ok_baselines

    st.pyplot(plot_rank_table(all_comparison, 'classification', theme=theme))

    with st.expander("📋 Raw Comparison DataFrame", expanded=False):
        cmp_df = pd.DataFrame(all_comparison).drop(columns=['status'], errors='ignore')
        st.dataframe(cmp_df, use_container_width=True)

    # ── STEP 10: Winner Banner ────────────────────────────────────────────────
    best_row   = max(all_comparison, key=lambda x: x.get('accuracy') or 0)
    metric_val = best_row.get('accuracy', 0)
    nas_val    = nas_metrics.get('accuracy', 0)

    if '🧬' in best_row['model']:
        st.markdown(f"""<div class="winner-banner">
        <h2 style="color:{WINNER_BC};font-family:'Orbitron',sans-serif;margin:0">🏆 NAS MODEL WINS!</h2>
        <p style="color:{WINNER_TEXT};margin:8px 0 0 0">
        The evolved neural architecture outperformed all {len(ok_baselines)} classical baselines.<br>
        Best Accuracy: <b style="color:{WINNER_BC}">{metric_val:.5f}</b>
        &nbsp;|&nbsp; Inference Time: <b style="color:{WINNER_BC}">{nas_predict_time}s</b>
        </p></div>""", unsafe_allow_html=True)
    else:
        gap = round(abs(metric_val - nas_val), 5)
        st.markdown(f"""<div class="best-card">
        <b style="color:{ACCENT2}">🥇 Best Overall: {best_row['model']}</b><br>
        <small style="color:{TEXT}">
        Accuracy: <b>{metric_val:.5f}</b> &nbsp;|&nbsp;
        NAS scored: <b>{nas_val:.5f}</b> &nbsp;|&nbsp;
        Gap: <b>{gap}</b><br>
        💡 Try: more generations, higher epochs per model, or a larger dataset.
        </small></div>""", unsafe_allow_html=True)

    # ── STEP 11: Save Model ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>💾 SAVE BEST NAS MODEL</div>", unsafe_allow_html=True)
    save_path = os.path.join(os.path.dirname(__file__), "best_model.keras")
    final_model.save(save_path)
    st.markdown(f"""<div class="genome-card">
    ✅ Saved → <code>{save_path}</code><br>
    Reload: <code>from tensorflow import keras; m = keras.models.load_model('best_model.keras')</code>
    </div>""", unsafe_allow_html=True)

    # ── STEP 12: Summary ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📈 RUN SUMMARY</div>", unsafe_allow_html=True)

    gen1  = generation_stats[0]['best_fitness']
    genN  = generation_stats[-1]['best_fitness']
    impro = round(((genN - gen1) / max(abs(gen1), 1e-9)) * 100, 1)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Generations",         generations)
    s2.metric("Total Models Trained", generations * pop_size + len(ok_baselines) + 1)
    s3.metric("Best Genome Layers",  str(global_best_genome['layers']))
    s4.metric("Fitness Improvement", f"{impro}%")

    st.success(
        "🎉 Complete! GA evolution finished, training curves plotted, "
        "all baselines benchmarked, and winner declared."
    )