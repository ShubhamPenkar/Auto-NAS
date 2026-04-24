"""
nn_visualizer.py
----------------
Interactive Neural Network Architecture Visualizer for Auto NAS.
Returns an HTML string to be rendered via st.components.v1.html().

Matches the dark cyberpunk theme of the main app (JetBrains Mono, #0a0e1a bg,
#00d4ff / #7b2ff7 accents). Genome-aware: pass a genome dict to pre-configure
the controls to match the best evolved architecture.
"""


def get_nn_visualizer_html(genome: dict = None, input_dim: int = 5, num_classes: int = 3) -> str:
    """
    Returns a self-contained HTML string for the interactive NN visualizer.

    Args:
        genome:      Optional genome dict with keys: layers, activation, dropout, optimizer.
                     If provided, the visualizer is pre-seeded with the genome's architecture.
        input_dim:   Number of input features (shown as input nodes).
        num_classes: Number of output classes (shown as output nodes).
    """
    # Clamp display counts
    input_n  = min(max(input_dim, 1), 8)
    output_n = min(max(num_classes, 2), 6)

    # Pre-seed values from genome if provided
    preset_layers     = genome['layers']    if genome else [128, 64]
    preset_activation = genome['activation'].capitalize() if genome else 'Relu'
    preset_dropout    = genome['dropout']   if genome else 0.2
    preset_optimizer  = genome.get('optimizer', 'adam').capitalize() if genome else 'Adam'

    # Activation display name mapping
    act_map = {'relu': 'ReLU', 'tanh': 'Tanh', 'elu': 'ELU', 'selu': 'SELU'}
    preset_activation_display = act_map.get(
        preset_activation.lower(), preset_activation.upper()
    )

    # Optimizer display name mapping
    opt_map = {'adam': 'Adam', 'rmsprop': 'RMSprop', 'adamw': 'AdamW'}
    preset_optimizer_display = opt_map.get(
        preset_optimizer.lower(), preset_optimizer
    )

    # Build the layer selector options, marking the correct one as selected
    layer_options_data = [
        (1, "[128]",                         [128]),
        (2, "[128, 64]",                     [128, 64]),
        (3, "[256, 128, 64]",                [256, 128, 64]),
        (4, "[512, 256, 128, 64]",           [512, 256, 128, 64]),
        (5, "[512, 256, 128, 64, 32]",       [512, 256, 128, 64, 32]),
        (6, "[1024, 512, 256, 128, 64, 32]", [1024, 512, 256, 128, 64, 32]),
    ]

    # Find closest match for preset_layers
    def layer_match_score(candidate):
        if len(candidate) != len(preset_layers):
            return 999
        return sum(abs(a - b) for a, b in zip(candidate, preset_layers))

    best_layer_n = min(layer_options_data, key=lambda x: layer_match_score(x[2]))[0]

    layer_options_html = ""
    for n, label, _ in layer_options_data:
        sel = 'selected' if n == best_layer_n else ''
        layer_options_html += f'<option value="{n}" {sel}>{n} Layer{"s" if n > 1 else ""} {label}</option>\n'

    act_options_html = ""
    for act in ['ReLU', 'Tanh', 'ELU', 'SELU']:
        sel = 'selected' if act == preset_activation_display else ''
        act_options_html += f'<option value="{act}" {sel}>{act}</option>\n'

    drop_options_html = ""
    for d in [0.0, 0.1, 0.2, 0.3]:
        sel = 'selected' if abs(d - preset_dropout) < 0.01 else ''
        drop_options_html += f'<option value="{d}" {sel}>{d}</option>\n'

    opt_options_html = ""
    for opt in ['Adam', 'RMSprop', 'AdamW']:
        sel = 'selected' if opt == preset_optimizer_display else ''
        opt_options_html += f'<option value="{opt}" {sel}>{opt}</option>\n'

    genome_badge = ""
    if genome:
        genome_badge = f"""
        <div style="margin-bottom:10px;padding:8px 12px;background:#0d1a2e;
                    border:1px solid #00d4ff40;border-left:3px solid #00d4ff;
                    border-radius:6px;font-size:11px;color:#58a6ff;font-family:'JetBrains Mono',monospace;">
            🧬 Genome loaded — Layers: <b style="color:#00d4ff">{preset_layers}</b>
            &nbsp;·&nbsp; Activation: <b style="color:#00d4ff">{preset_activation_display}</b>
            &nbsp;·&nbsp; Dropout: <b style="color:#00d4ff">{preset_dropout}</b>
            &nbsp;·&nbsp; Optimizer: <b style="color:#00d4ff">{preset_optimizer_display}</b>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0e1a; color: #c9d1d9; font-family: 'JetBrains Mono', monospace; padding: 12px; }}

  .viz-title {{
    font-family: 'Orbitron', sans-serif; font-size: 13px; font-weight: 700;
    color: #00d4ff; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 10px;
  }}

  .controls {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; align-items: flex-end;
  }}
  .ctrl-group {{ display: flex; flex-direction: column; gap: 3px; }}
  .ctrl-label {{
    font-size: 9px; color: #58a6ff; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  select {{
    height: 28px; padding: 0 8px;
    background: #161b2e; color: #c9d1d9;
    border: 1px solid #21262d; border-radius: 5px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    cursor: pointer; outline: none;
  }}
  select:focus {{ border-color: #00d4ff; }}

  .play-btn {{
    height: 28px; padding: 0 14px;
    background: linear-gradient(135deg, #00d4ff20, #7b2ff720);
    border: 1px solid #00d4ff; border-radius: 5px;
    color: #00d4ff; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; cursor: pointer; letter-spacing: 1px;
    transition: all 0.2s;
  }}
  .play-btn:hover {{ background: linear-gradient(135deg, #00d4ff40, #7b2ff740); box-shadow: 0 0 10px #00d4ff40; }}
  .play-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}

  .legend {{
    display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; font-size: 10px; color: #8b949e; }}
  .leg-dot  {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-rect {{ width: 14px; height: 7px; border-radius: 2px; flex-shrink: 0; }}

  canvas {{ display: block; width: 100%; border-radius: 6px; }}

  .info-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
  .info-card {{
    flex: 1; min-width: 80px;
    background: #161b2e; border: 1px solid #21262d; border-radius: 6px;
    padding: 7px 10px;
  }}
  .info-val {{ font-family: 'Orbitron', sans-serif; font-size: 13px; color: #00d4ff; line-height: 1.2; }}
  .info-lbl {{ font-size: 9px; color: #8b949e; margin-top: 2px; text-transform: uppercase; letter-spacing: 1px; }}
</style>
</head>
<body>

<div class="viz-title">⬡ Architecture Explorer</div>

{genome_badge}

<div class="controls">
  <div class="ctrl-group">
    <span class="ctrl-label">Hidden Layers</span>
    <select id="sel-layers" onchange="rebuildNet()">
      {layer_options_html}
    </select>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Activation</span>
    <select id="sel-act" onchange="rebuildNet()">
      {act_options_html}
    </select>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Dropout</span>
    <select id="sel-drop" onchange="rebuildNet()">
      {drop_options_html}
    </select>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Optimizer</span>
    <select id="sel-opt" onchange="rebuildNet()">
      {opt_options_html}
    </select>
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">&nbsp;</span>
    <button class="play-btn" id="btn-play" onclick="startForwardPass()">▶ Forward pass</button>
  </div>
</div>

<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#00d4ff;border:1px solid #0099bb"></div> Input</div>
  <div class="leg-item"><div class="leg-dot" style="background:#7b2ff7;border:1px solid #5520aa"></div> Dense (hidden)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#ff4444;border:1px solid #cc2222"></div> Dropped out</div>
  <div class="leg-item"><div class="leg-dot" style="background:#00ff88;border:1px solid #00bb66"></div> Output</div>
  <div class="leg-item"><div class="leg-rect" style="background:#ffd16640;border:1px solid #ffd166"></div> Batch Norm</div>
</div>

<canvas id="c"></canvas>
<div class="info-bar" id="info-bar"></div>

<script>
const C   = document.getElementById('c');
const ctx = C.getContext('2d');

const INPUT_N  = {input_n};
const OUTPUT_N = {output_n};
const MAX_SHOW = 6;

let NET  = null;
let anim = null;

const PAL = {{
  input:   {{ fill:'#001a2e', stroke:'#00d4ff', text:'#00d4ff' }},
  hidden:  {{ fill:'#1a0d2e', stroke:'#7b2ff7', text:'#a070ff' }},
  dropped: {{ fill:'#2e0d0d', stroke:'#ff4444', text:'#ff6666' }},
  output:  {{ fill:'#0d2e1a', stroke:'#00ff88', text:'#00ff88' }},
  bn:      {{ fill:'#2a2010', stroke:'#ffd166' }},
  edge:    '#1e3a5c',
  active:  '#00d4ff',
  activePulse: '#7b2ff7',
}};

function getLayerSizes(n) {{
  return [
    [128],
    [128,64],
    [256,128,64],
    [512,256,128,64],
    [512,256,128,64,32],
    [1024,512,256,128,64,32],
  ][n-1];
}}

function buildNet() {{
  const nLayers   = parseInt(document.getElementById('sel-layers').value);
  const activation = document.getElementById('sel-act').value;
  const dropout   = parseFloat(document.getElementById('sel-drop').value);
  const optimizer = document.getElementById('sel-opt').value;
  const hiddenSizes = getLayerSizes(nLayers);

  const allLayers = [
    {{ type:'input',  size:INPUT_N,  label:'Input',    sublabel:INPUT_N+' features' }},
    ...hiddenSizes.map((s,i) => ({{
      type:'hidden', size:s, label:'Dense '+(i+1),
      sublabel:s+' units', activation, dropout, hasBN:true
    }})),
    {{ type:'output', size:OUTPUT_N, label:'Output', sublabel:'Softmax · '+OUTPUT_N+' cls' }}
  ];

  const DPR  = window.devicePixelRatio || 1;
  const W    = C.parentElement.clientWidth || 680;
  const nC   = allLayers.length;
  const COLW = Math.min(90, (W - 40) / nC);
  const TW   = Math.round(COLW * nC + 40);
  const HC   = 420;

  C.width  = TW * DPR; C.height = HC * DPR;
  C.style.width  = TW + 'px'; C.style.height = HC + 'px';
  ctx.scale(DPR, DPR);

  const NR       = Math.min(14, COLW * 0.2);
  const TOP_PAD  = 32;
  const DRAW_H   = HC - TOP_PAD - 72;

  function colX(i) {{ return 20 + COLW * i + COLW / 2; }}

  function layerNodes(layer, cx) {{
    const n = Math.min(layer.size, MAX_SHOW);
    const spacing = Math.min(NR * 2.8, (DRAW_H - 60) / Math.max(n-1, 1));
    const totalH  = (n-1) * spacing;
    const startY  = TOP_PAD + (DRAW_H - totalH) / 2;
    return Array.from({{length:n}}, (_,j) => ({{ x:cx, y:startY+j*spacing }}));
  }}

  const allCols = allLayers.map((layer,i) => {{
    const cx = colX(i);
    return {{ layer, cx, nodes:layerNodes(layer,cx), showEllipsis:layer.size > MAX_SHOW }};
  }});

  NET = {{ allCols, NR, COLW, TOP_PAD, DRAW_H, TW, HC, activation, dropout, optimizer, hiddenSizes, nLayers }};
  drawAll();
  renderInfoBar(hiddenSizes, activation, dropout, optimizer);
}}

function drawAll(highlightCol, activeEdgeCol) {{
  if (!NET) return;
  const {{ allCols, NR, TW, HC }} = NET;
  ctx.clearRect(0,0,TW,HC);

  // Grid lines background
  ctx.save();
  ctx.strokeStyle = '#ffffff08';
  ctx.lineWidth = 0.5;
  for (let x = 0; x < TW; x += 40) {{
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,HC); ctx.stroke();
  }}
  for (let y = 0; y < HC; y += 40) {{
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(TW,y); ctx.stroke();
  }}
  ctx.restore();

  // Draw edges
  allCols.forEach((col,ci) => {{
    if (ci === allCols.length-1) return;
    const next = allCols[ci+1];
    col.nodes.forEach(n1 => {{
      next.nodes.forEach(n2 => {{
        const isActive = ci === activeEdgeCol;
        ctx.beginPath();
        ctx.moveTo(n1.x + NR, n1.y);
        ctx.lineTo(n2.x - NR, n2.y);
        if (isActive) {{
          ctx.strokeStyle = PAL.active;
          ctx.lineWidth   = 1.2;
          ctx.globalAlpha = 0.6;
        }} else {{
          ctx.strokeStyle = PAL.edge;
          ctx.lineWidth   = 0.4;
          ctx.globalAlpha = 0.3;
        }}
        ctx.stroke();
        ctx.globalAlpha = 1;
      }});
    }});
    if (col.showEllipsis) {{
      const last = col.nodes[col.nodes.length-1];
      ctx.fillStyle = '#58a6ff';
      ctx.font = `bold 12px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText('···', col.cx, last.y + 22);
    }}
  }});

  // Draw nodes
  allCols.forEach((col,ci) => {{
    const {{ layer, cx, nodes }} = col;
    const isHi = highlightCol === ci;

    // BN bar above layer
    if (layer.hasBN && nodes.length > 0) {{
      const barW = NET.COLW * 0.55;
      const barX = cx - barW/2;
      const barY = nodes[0].y - 22;
      ctx.fillStyle   = PAL.bn.fill;
      ctx.strokeStyle = PAL.bn.stroke;
      ctx.lineWidth   = 0.8;
      ctx.beginPath(); ctx.roundRect(barX, barY, barW, 6, 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = PAL.bn.stroke;
      ctx.font = `bold 8px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('BN', cx, barY+3);
    }}

    nodes.forEach((nd,j) => {{
      const isDrop = layer.type==='hidden' && layer.dropout>0 && j%4===2;
      const pal    = isDrop ? PAL.dropped :
                     layer.type==='input'  ? PAL.input  :
                     layer.type==='output' ? PAL.output : PAL.hidden;
      const r = isHi ? NR * 1.25 : NR;

      // Glow for highlighted
      if (isHi && !isDrop) {{
        ctx.save();
        ctx.shadowColor = pal.stroke;
        ctx.shadowBlur  = 12;
        ctx.beginPath(); ctx.arc(nd.x, nd.y, r, 0, Math.PI*2);
        ctx.strokeStyle = pal.stroke; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
      }}

      ctx.beginPath(); ctx.arc(nd.x, nd.y, r, 0, Math.PI*2);
      ctx.fillStyle = pal.fill; ctx.fill();
      ctx.strokeStyle = pal.stroke;
      ctx.lineWidth = isHi ? 1.8 : 0.8;
      ctx.stroke();

      if (isDrop) {{
        ctx.fillStyle = PAL.dropped.text;
        ctx.font = `bold ${{NR*0.75}}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('×', nd.x, nd.y);
      }} else if (layer.type === 'input') {{
        ctx.fillStyle = PAL.input.text;
        ctx.font = `bold ${{Math.max(8, NR*0.65)}}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('x'+(j+1), nd.x, nd.y);
      }}
    }});

    // Column label
    const ly = NET.TOP_PAD + NET.DRAW_H + 14;
    ctx.fillStyle = '#00d4ff';
    ctx.font = `bold ${{Math.max(9, Math.min(11, NET.COLW*0.12))}}px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(layer.label, cx, ly);
    ctx.fillStyle = '#58a6ff';
    ctx.font = `${{Math.max(8, Math.min(10, NET.COLW*0.10))}}px 'JetBrains Mono', monospace`;
    ctx.fillText(layer.sublabel, cx, ly+13);
    if (layer.hasBN) {{
      ctx.fillStyle = PAL.bn.stroke;
      ctx.font = `8px 'JetBrains Mono', monospace`;
      ctx.fillText('BN→Drop('+layer.dropout+')', cx, ly+25);
    }}
    if (layer.type==='output') {{
      ctx.fillStyle = '#7b2ff7';
      ctx.font = `8px 'JetBrains Mono', monospace`;
      ctx.fillText(NET.optimizer+'·lr=0.001', cx, ly+25);
    }}
  }});

  // Title
  ctx.fillStyle = '#00d4ff80';
  ctx.font = `bold 10px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('AUTO NAS · ' + NET.nLayers + ' hidden layer'+(NET.nLayers>1?'s':'') +
    ' · ' + NET.activation + ' · dropout ' + NET.dropout, 20, 10);
}}

function startForwardPass() {{
  if (anim) {{ cancelAnimationFrame(anim); anim=null; }}
  const btn = document.getElementById('btn-play');
  btn.disabled = true;
  let col=0; const total=NET.allCols.length;
  const STEP=300; let last=null;
  function step(ts) {{
    if (!last) last=ts;
    if (ts-last < STEP) {{ anim=requestAnimationFrame(step); return; }}
    last=ts;
    drawAll(col, col-1);
    col++;
    if (col<total) {{ anim=requestAnimationFrame(step); }}
    else {{ setTimeout(() => {{ drawAll(); btn.disabled=false; anim=null; }}, STEP); }}
  }}
  anim=requestAnimationFrame(step);
}}

function renderInfoBar(hiddenSizes, activation, dropout, optimizer) {{
  let params = hiddenSizes.reduce((acc,s,i) => {{
    const inp = i===0 ? INPUT_N : hiddenSizes[i-1];
    return acc + inp*s + s;
  }}, 0);
  const bar = document.getElementById('info-bar');
  bar.innerHTML = [
    {{ val: NET.nLayers+2,               lbl:'Total layers'   }},
    {{ val: hiddenSizes.join('→'),        lbl:'Hidden units'   }},
    {{ val: (params/1000).toFixed(1)+'K', lbl:'Est. params'    }},
    {{ val: activation,                  lbl:'Activation'     }},
    {{ val: dropout,                     lbl:'Dropout'        }},
    {{ val: optimizer,                   lbl:'Optimizer'      }},
  ].map(d => `<div class="info-card"><div class="info-val">${{d.val}}</div><div class="info-lbl">${{d.lbl}}</div></div>`).join('');
}}

function rebuildNet() {{
  if (anim) {{ cancelAnimationFrame(anim); anim=null; }}
  document.getElementById('btn-play').disabled=false;
  buildNet();
}}

window.addEventListener('resize', () => {{ if (NET) buildNet(); }});
buildNet();
</script>
</body>
</html>"""

    return html
