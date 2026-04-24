"""
plotter.py
----------
All matplotlib visualizations for Auto NAS (classification only).
Supports light and dark themes via the `theme` parameter (default: 'light').
"""

import matplotlib.pyplot as plt
import numpy as np


def _t(theme):
    """Return a dict of theme colours."""
    if theme == 'dark':
        return {
            'bg':        '#0f1117',
            'fig_bg':    '#0a0e1a',
            'text':      'white',
            'grid':      'white',
            'spine':     '#333333',
            'legend_bg': '#1a1a2e',
            'legend_ec': '#333333',
            'row_odd':   '#0f1117',
            'row_even':  '#161b2e',
            'nas_row':   '#0d2137',
            'header':    '#1a1a3e',
            'header_txt':'#00d4ff',
            'nas_txt':   '#00d4ff',
            'cell_txt':  '#c9d1d9',
            'cell_ec':   '#21262d',
            'header_ec': '#333355',
            'title':     'white',
            'acc_col':   '#00d4ff',
            'f1_col':    '#7b2ff7',
            'roc_col':   '#ff6b6b',
            'val_col':   '#7b2ff7',
            'lr_col':    '#ffd166',
            'bar_lbl':   'white',
            'nas_edge':  '#ffffff',
        }
    else:  # light
        return {
            'bg':        '#ffffff',
            'fig_bg':    '#f4f6fb',
            'text':      '#1a1a2e',
            'grid':      '#bbbbbb',
            'spine':     '#cccccc',
            'legend_bg': '#ffffff',
            'legend_ec': '#cccccc',
            'row_odd':   '#ffffff',
            'row_even':  '#f0f4ff',
            'nas_row':   '#ddeeff',
            'header':    '#1a1a3e',
            'header_txt':'#00aadd',
            'nas_txt':   '#0066aa',
            'cell_txt':  '#1a1a2e',
            'cell_ec':   '#cccccc',
            'header_ec': '#334466',
            'title':     '#1a1a2e',
            'acc_col':   '#0077cc',
            'f1_col':    '#6a0dad',
            'roc_col':   '#e05c3a',
            'val_col':   '#6a0dad',
            'lr_col':    '#cc8800',
            'bar_lbl':   '#1a1a2e',
            'nas_edge':  '#0077cc',
        }


# ─────────────────────────────────────────────────────────────────────────────
# FITNESS OVER GENERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitness_over_generations(generation_stats: list, theme: str = 'light') -> plt.Figure:
    c = _t(theme)
    generations  = [s['generation']  for s in generation_stats]
    best_fitness = [s['best_fitness'] for s in generation_stats]
    avg_fitness  = [s['avg_fitness']  for s in generation_stats]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(c['fig_bg'])
    ax.set_facecolor(c['bg'])

    ax.plot(generations, best_fitness, color=c['acc_col'], linewidth=2.5,
            marker='o', markersize=6, label='Best Fitness')
    ax.plot(generations, avg_fitness,  color=c['roc_col'], linewidth=1.5,
            linestyle='--', marker='s', markersize=4, label='Avg Fitness')
    ax.fill_between(generations, avg_fitness, best_fitness, alpha=0.12, color=c['acc_col'])

    ax.set_xlabel('Generation', color=c['text'], fontsize=12)
    ax.set_ylabel('Fitness (Val Accuracy)', color=c['text'], fontsize=12)
    ax.set_title('Fitness Evolution Across Generations', color=c['title'],
                 fontsize=14, fontweight='bold')
    ax.tick_params(colors=c['text'])
    for spine in ax.spines.values(): spine.set_edgecolor(c['spine'])
    ax.legend(facecolor=c['legend_bg'], edgecolor=c['legend_ec'],
              labelcolor=c['text'], fontsize=10)
    ax.grid(True, alpha=0.2, color=c['grid'])
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def plot_architecture(genome: dict, theme: str = 'light') -> plt.Figure:
    c = _t(theme)
    layers     = genome['layers']
    activation = genome['activation']
    dropout    = genome['dropout']
    optimizer  = genome.get('optimizer', 'adam')

    all_labels  = ['Input'] + [str(n) for n in layers] + ['Output']
    n_layers    = len(all_labels)
    colors      = ['#4ecdc4'] + [c['acc_col']] * len(layers) + [c['roc_col']]
    x_positions = np.linspace(0.1, 0.9, n_layers)

    fig, ax = plt.subplots(figsize=(max(9, n_layers * 1.8), 5))
    fig.patch.set_facecolor(c['fig_bg'])
    ax.set_facecolor(c['bg'])

    for i, (x, label, color) in enumerate(zip(x_positions, all_labels, colors)):
        circle = plt.Circle((x, 0.5), 0.06, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, 0.5, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#ffffff', zorder=4)
        if i < n_layers - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.065, 0.5),
                        xytext=(x + 0.065, 0.5),
                        arrowprops=dict(arrowstyle='->', color=c['spine'], lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.axis('off')
    ax.set_title(
        f"Architecture: {layers}  |  Activation: {activation}  |  Dropout: {dropout}  |  Optimizer: {optimizer}",
        color=c['title'], fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(history_dict: dict, problem_type: str = 'classification',
                        theme: str = 'light') -> plt.Figure:
    c = _t(theme)
    has_lr  = 'lr' in history_dict
    n_plots = 3 if has_lr else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    fig.patch.set_facecolor(c['fig_bg'])
    epochs = range(1, len(history_dict['loss']) + 1)

    ax1 = axes[0]
    ax1.set_facecolor(c['bg'])
    ax1.plot(epochs, history_dict['loss'],     color=c['acc_col'], lw=2, label='Train Loss')
    ax1.plot(epochs, history_dict['val_loss'], color=c['roc_col'], lw=2, linestyle='--', label='Val Loss')
    ax1.fill_between(epochs, history_dict['loss'], history_dict['val_loss'], alpha=0.08, color=c['acc_col'])
    ax1.set_title('Loss Curve', color=c['title'], fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', color=c['text'])
    ax1.set_ylabel('Loss', color=c['text'])
    ax1.tick_params(colors=c['text'])
    ax1.legend(facecolor=c['legend_bg'], edgecolor=c['legend_ec'], labelcolor=c['text'])
    ax1.grid(True, alpha=0.15, color=c['grid'])
    for spine in ax1.spines.values(): spine.set_edgecolor(c['spine'])

    ax2 = axes[1]
    ax2.set_facecolor(c['bg'])
    ax2.plot(epochs, history_dict['accuracy'],     color=c['acc_col'], lw=2, label='Train Acc')
    ax2.plot(epochs, history_dict['val_accuracy'], color=c['val_col'], lw=2, linestyle='--', label='Val Acc')
    ax2.set_title('Accuracy Curve', color=c['title'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', color=c['text'])
    ax2.set_xlabel('Epoch', color=c['text'])
    ax2.tick_params(colors=c['text'])
    ax2.legend(facecolor=c['legend_bg'], edgecolor=c['legend_ec'], labelcolor=c['text'])
    ax2.grid(True, alpha=0.15, color=c['grid'])
    for spine in ax2.spines.values(): spine.set_edgecolor(c['spine'])

    if has_lr:
        ax3 = axes[2]
        ax3.set_facecolor(c['bg'])
        ax3.plot(epochs, history_dict['lr'], color=c['lr_col'], lw=2, label='Learning Rate')
        ax3.set_title('Learning Rate Schedule', color=c['title'], fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch', color=c['text'])
        ax3.set_ylabel('LR', color=c['text'])
        ax3.tick_params(colors=c['text'])
        ax3.legend(facecolor=c['legend_bg'], edgecolor=c['legend_ec'], labelcolor=c['text'])
        ax3.grid(True, alpha=0.15, color=c['grid'])
        for spine in ax3.spines.values(): spine.set_edgecolor(c['spine'])

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_classification(baseline_results: list, nas_metrics: dict,
                                   theme: str = 'light') -> plt.Figure:
    c = _t(theme)
    all_results = [{'model': '🧬 NAS (Ours)', **nas_metrics}] + \
                  [r for r in baseline_results if r['status'] == 'ok']

    model_names = [r['model']               for r in all_results]
    accuracies  = [r.get('accuracy') or 0   for r in all_results]
    f1s         = [r.get('f1')       or 0   for r in all_results]
    roc_aucs    = [r.get('roc_auc')  or 0   for r in all_results]

    n = len(model_names)
    x = np.arange(n)
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(12, n * 1.6), 5))
    fig.patch.set_facecolor(c['fig_bg'])
    ax.set_facecolor(c['bg'])

    b_acc = ax.bar(x - w, accuracies, w, label='Accuracy', color=c['acc_col'], alpha=0.85)
    b_f1  = ax.bar(x,     f1s,        w, label='F1 Score', color=c['f1_col'],  alpha=0.85)
    b_roc = ax.bar(x + w, roc_aucs,   w, label='ROC-AUC',  color=c['roc_col'], alpha=0.85)

    for b in [b_acc[0], b_f1[0], b_roc[0]]:
        b.set_edgecolor(c['nas_edge'])
        b.set_linewidth(2)

    for bars in [b_acc, b_f1, b_roc]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom',
                        color=c['bar_lbl'], fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', color=c['text'], fontsize=9)
    ax.set_ylabel('Score', color=c['text'], fontsize=11)
    ax.set_title('Model Comparison — Classification Metrics', color=c['title'],
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=c['text'])
    ax.yaxis.grid(True, alpha=0.18, color=c['grid'])
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_edgecolor(c['spine'])
    ax.legend(facecolor=c['legend_bg'], edgecolor=c['legend_ec'],
              labelcolor=c['text'], fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RANKED TABLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_table(comparison_data: list, problem_type: str = 'classification',
                    theme: str = 'light') -> plt.Figure:
    c = _t(theme)
    data = sorted(comparison_data, key=lambda x: x.get('accuracy') or 0, reverse=True)
    cols = ['Rank', 'Model', 'Accuracy', 'F1 Score', 'ROC-AUC', 'Params', 'Inference Time']
    rows = []

    for i, r in enumerate(data):
        params_val = r.get('params')
        params_str = f"{int(params_val):,}" if params_val is not None else 'N/A'
        t_val      = r.get('predict_time')
        time_str   = f"{float(t_val):.3f}s" if t_val is not None else 'N/A'

        rows.append([
            f"#{i+1}",
            r['model'],
            f"{r['accuracy']:.5f}" if r.get('accuracy') is not None else 'N/A',
            f"{r['f1']:.5f}"       if r.get('f1')       is not None else 'N/A',
            f"{r['roc_auc']:.5f}"  if r.get('roc_auc')  is not None else 'N/A',
            params_str,
            time_str
        ])

    fig_h = max(3, len(rows) * 0.48 + 1.4)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor(c['fig_bg'])
    ax.axis('off')

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)

    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor(c['header'])
        cell.set_text_props(color=c['header_txt'], fontweight='bold')
        cell.set_edgecolor(c['header_ec'])

    for i in range(1, len(rows) + 1):
        is_nas = '🧬' in rows[i-1][1]
        for j in range(len(cols)):
            cell = tbl[i, j]
            if is_nas:
                cell.set_facecolor(c['nas_row'])
                cell.set_text_props(color=c['nas_txt'])
            else:
                cell.set_facecolor(c['row_even'] if i % 2 == 0 else c['row_odd'])
                cell.set_text_props(color=c['cell_txt'])
            cell.set_edgecolor(c['cell_ec'])

    ax.set_title('Full Model Ranking', color=c['title'], fontsize=13,
                 fontweight='bold', pad=15)
    plt.tight_layout()
    return fig