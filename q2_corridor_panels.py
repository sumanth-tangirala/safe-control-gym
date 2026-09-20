'''Render the corridor success-probability slices as one self-contained HTML page.

Reads the slice_*.npz files written by the sweep (each a 43x61 grid of
per-state success probability over (x, z), at fixed theta/x_dot/z_dot/
theta_dot) and emits a single page with no external assets, ready for the
Artifact tool.

The previous generator lived in a session scratchpad and was lost when that
mount went away (2026-08-19); this one lives in the repo so a re-render is
`python q2_corridor_panels.py`.

Encoding: diverging blue-red with a neutral midpoint at p = 0.5. Blue means
the start reliably reaches the goal, red means it reliably does not, and the
neutral band is where the label is a coin flip. That midpoint is the quantity
the whole corridor design optimises (`fraction_interior`), so it gets the
neutral -- a plain sequential ramp would bury it as a mid-tone.

Usage:
  python q2_corridor_panels.py --out corridor_panels.html
'''
import argparse
import glob
import json
import os

import numpy as np

from q2_common import TOL
from q2_corridor_common import BAND, CENTRE, SINE_PERIOD, WIDTH

# Panels in reading order: each row is one flight condition, each column one
# config. The grid is deliberately ragged -- the sweep measured descending
# only under `sharp` and spinning only under `smooth`, so those rows carry two
# panels, not three. Rendering a blank third cell would imply a run that never
# happened.
ROWS = [
    ('Hover', 'Released from rest: no velocity, no tilt, no spin.',
     ['slice_baseline.npz', 'slice_f0.08_a0.06.npz', 'slice_f0.05_a0.09.npz']),
    ('Descending at 0.5 m/s', 'Released already falling (z_dot = -0.5).',
     ['slice_base_zd.npz', 'slice_zd_f0.08_a0.06.npz']),
    ('Spinning at 4 rad/s', 'Released already rotating (theta_dot = 4).',
     ['slice_base_td.npz', 'slice_td_f0.05_a0.09.npz']),
]

CONFIG_NAMES = {(0.0, -1.0): 'baseline', (0.08, 0.06): 'sharp', (0.05, 0.09): 'smooth'}
GOAL_Z = 1.0

# Diverging ramp, 11 stops from p=0 to p=1. Light mode recedes toward the light
# surface at the midpoint; dark mode recedes toward the dark surface, so the
# arms brighten outward rather than darken.
RAMP_LIGHT = ['#7d1a1a', '#a02525', '#c23434', '#dc5c5c', '#eda0a0', '#f0efec',
              '#b7d3f6', '#86b6ef', '#3987e5', '#256abf', '#0d366b']
RAMP_DARK = ['#f0a0a0', '#e88080', '#de6060', '#c04d4d', '#7a4040', '#383835',
             '#35507a', '#2a6ab0', '#3987e5', '#6da7ec', '#9ec5f4']


def load_panel(path):
    d = np.load(path, allow_pickle=True)

    def g(k, default=0.0):
        return float(d[k]) if k in d.files else default

    p = d['p']
    f_max, ambient = g('f_max'), g('ambient', -1.0)
    interior = (p > 0) & (p < 1)
    return dict(
        file=os.path.basename(path),
        config=CONFIG_NAMES.get((round(f_max, 4), round(ambient, 4)), 'custom'),
        model=str(d['model']), f_max=f_max, ambient=ambient, trials=int(d['trials']),
        theta=g('theta'), xd=g('xd'), zd=g('zd'), td=g('td'),
        xs=[round(v, 4) for v in d['xs'].tolist()],
        zs=[round(v, 4) for v in d['zs'].tolist()],
        # 3dp is finer than the 1/20 = 0.05 quantum a 20-trial estimate can
        # actually resolve, so nothing measured is lost to rounding.
        p=[round(v, 3) for v in p.ravel().tolist()],
        nz=int(p.shape[0]), nx=int(p.shape[1]), n=int(p.size),
        mean_p=float(p.mean()),
        n_certain_ok=int((p >= 1).sum()), n_certain_fail=int((p <= 0).sum()),
        n_fuzzy=int(interior.sum()),
    )


def build(panels_by_file, out_path):
    rows = []
    for title, subtitle, files in ROWS:
        present = [panels_by_file[f] for f in files if f in panels_by_file]
        if present:
            rows.append(dict(title=title, subtitle=subtitle, panels=present))
    payload = json.dumps(dict(rows=rows, band=[round(BAND[0], 3), round(BAND[1], 3)],
                              centre=CENTRE, width=WIDTH, period=SINE_PERIOD,
                              goal_z=GOAL_Z, goal_r=TOL,
                              rampLight=RAMP_LIGHT, rampDark=RAMP_DARK),
                         separators=(',', ':'))
    with open(out_path, 'w') as fh:
        fh.write(TEMPLATE.replace('/*DATA*/', payload))
    return out_path


TEMPLATE = r'''<title>Quad2D altitude corridor: where the disturbance breaks the controller</title>
<style>
/* Neutrals carry a slight cool bias, pulled toward the blue arm of the
   diverging scale, so page chrome and data read as one system. Mono is the
   utility face throughout -- coordinates, counts, parameters and eyebrows are
   all numerical, and this is the vernacular they are normally read in. */
:root{
  color-scheme:light;
  --page:#f7f8fa; --surface:#ffffff; --sunk:#eef1f5;
  --ink:#111318; --ink2:#4a4f58; --muted:#7f8794;
  --rule:#e2e6ec; --axis:#c2c8d2; --ring:rgba(17,19,24,0.09);
  --fail:#a02525; --ok:#256abf;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
    --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
    --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
    --fail:#e88080; --ok:#6da7ec;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
  --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
  --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
  --fail:#e88080; --ok:#6da7ec;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 var(--sans);
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:52px 24px 80px;
  display:flex;flex-direction:column;gap:0}
h1{font-size:30px;line-height:1.2;margin:0 0 14px;letter-spacing:-0.021em;
  font-weight:640;text-wrap:balance;max-width:22ch}
h2{font-size:15px;margin:0;letter-spacing:0.09em;text-transform:uppercase;
  font-family:var(--mono);font-weight:600;color:var(--ink)}
h3{font-size:12px;margin:0 0 10px;font-family:var(--mono);font-weight:600;
  letter-spacing:0.09em;text-transform:uppercase;color:var(--muted)}
p{margin:0 0 14px;max-width:68ch;color:var(--ink2)}
p:last-child{margin-bottom:0}
.lede{font-size:17px;line-height:1.55;color:var(--ink);max-width:60ch;margin-bottom:30px}
.sub{color:var(--muted);font-size:13.5px}
code{font:0.875em/1.5 var(--mono);background:var(--sunk);border-radius:3px;padding:1.5px 5px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;
  padding:20px 22px;margin:0 0 30px}
ol.recipe{margin:0;padding-left:19px;color:var(--ink2);max-width:68ch}
ol.recipe li{margin:0 0 7px}
ol.recipe li:last-child{margin:0}
/* Each row is one held-fixed flight condition -- the eyebrow states which
   state variables are pinned, because that is what separates the rows. */
.rowhead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  padding:0 0 12px;border-bottom:1px solid var(--rule);margin:38px 0 0}
.rowhead .sub{margin:0}
.panels{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0 0}
.panel{background:var(--surface);border:1px solid var(--ring);border-radius:8px;
  padding:15px;flex:1 1 296px;min-width:272px;max-width:340px}
.panel .name{font-size:15px;font-weight:620;margin:0 0 3px;letter-spacing:-0.01em}
.panel .cfg{font:11.5px/1.5 var(--mono);color:var(--muted);margin:0 0 12px;
  font-variant-numeric:tabular-nums}
canvas{width:100%;height:auto;display:block;border-radius:3px;cursor:crosshair;
  background:var(--sunk)}
.readout{font:11.5px/1.55 var(--mono);color:var(--ink2);margin:10px 0 0;
  min-height:3.1em;font-variant-numeric:tabular-nums;white-space:pre-line}
.scale{display:flex;align-items:center;gap:12px;margin:0 0 8px;flex-wrap:wrap}
.bar{height:10px;flex:1 1 200px;min-width:170px;border-radius:2px;border:1px solid var(--ring)}
.scale span{font:11.5px/1.4 var(--mono);color:var(--muted)}
.tablewrap{overflow-x:auto;margin:18px 0 10px;-webkit-overflow-scrolling:touch;
  border:1px solid var(--ring);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;font-size:13.5px;min-width:620px;width:100%}
th,td{text-align:right;padding:9px 14px;border-bottom:1px solid var(--rule);
  font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12.5px}
th:first-child,td:first-child,td.cfgcell{text-align:left;font-family:var(--sans);
  font-variant-numeric:normal;font-size:13.5px}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.07em;font-family:var(--mono)}
tbody tr:last-child td{border-bottom:none}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px}
.note{border-left:2px solid var(--axis);padding:1px 0 1px 16px;margin:0 0 18px;
  color:var(--ink2);max-width:68ch}
.notes{margin:22px 0 0;display:flex;flex-direction:column;gap:4px}
.foot{color:var(--muted);font-size:12.5px;line-height:1.7;margin-top:40px;
  border-top:1px solid var(--rule);padding-top:18px;max-width:78ch}
:focus-visible{outline:2px solid var(--ok);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="wrap">
<h1>Quad2D altitude corridor: where the disturbance breaks the controller</h1>
<p class="lede">A corridor is a band of disturbed air the quadrotor must cross, with the goal
in calm air above it. These maps show, for every starting position, how often the controller
still reaches the goal once the corridor is switched on.</p>

<div class="card">
<h3>How each pixel was measured</h3>
<ol class="recipe">
<li>Pick a starting position <code>(x, z)</code> on a 61 &times; 43 grid, holding the other four
state variables fixed at the value named on the row.</li>
<li>Fly the trained <code>safe_explorer_ppo</code> controller from it, with a fresh random
draw of the disturbance, until it reaches the goal or runs out of time.</li>
<li>Repeat 20 times for the disturbed configs, each with a different draw.</li>
<li>The pixel's value is successes divided by attempts. Blue is 20 of 20, red is 0 of 20,
and the neutral band in the middle is roughly a coin flip.</li>
</ol>
</div>

<div class="scale">
  <span>0 &mdash; never reaches the goal</span>
  <div class="bar" id="legendbar"></div>
  <span>1 &mdash; always reaches the goal</span>
</div>
<p class="sub">Dashed lines mark the corridor band, where the sideways gust is at least 1% of its
peak. The gust pushes in +x, to the right. The ellipse is the goal ball at its true radius
of 0.2, drawn as an ellipse only because the two axes are scaled differently. Hover any map
to read exact values.</p>

<div id="rows"></div>

<h2>Reading the numbers</h2>
<div class="tablewrap"><table id="stats">
  <thead><tr><th>Condition</th><th>Config</th><th>Mean success</th>
  <th>Always reaches</th><th>Never reaches</th><th>In between</th></tr></thead>
  <tbody></tbody>
</table></div>
<p class="sub">Counts are grid cells out of 2,623 per map. &ldquo;In between&rdquo; is the
fuzzy zone &mdash; cells that sometimes succeed and sometimes fail.</p>

<div class="notes">
<div class="note">
<p><strong>Not every condition was run under both configs.</strong> The sweep measured the
descending case under <code>sharp</code> only and the spinning case under <code>smooth</code>
only, so those rows carry two maps rather than three. The missing cells are runs that were
never made, not results that came back empty.</p>
</div>

<div class="note">
<p><strong>The baseline&rsquo;s zero fuzzy cells are a definition, not a finding.</strong>
With no disturbance there is nothing to vary, so each baseline start was flown once. A single
flight can only score 0 or 1, so no in-between value is reachable. Baseline is the reference for
<em>which</em> starts work at all, not for how uncertain they are.</p>
</div>

<div class="note">
<p><strong>These maps are far fuzzier than the dataset as a whole, and that is expected.</strong>
Measured over the full collected set, 7.01 fuzzy starts per 100 for sharp and 8.96 for smooth,
across all 489,789 six-dimensional grid states. Most of those sit far from the corridor and never
interact with it. These slices deliberately cut through the region where the corridor acts, so
they concentrate exactly the states the global average dilutes. The two numbers measure different
populations; neither contradicts the other.</p>
</div>

<div class="note">
<p><strong>20 flights per start undercounts fuzziness.</strong> A start that fails only one time
in fifty will usually show 20 successes out of 20 and be painted solid blue. The bias pushes
every config the same way, so comparisons between them hold even though each map understates
the true fuzzy area. A top-up to 50 flights per start is built and verified.</p>
</div>
</div>

<p class="foot">Disturbance: <code>F_x = sigma(z) &middot; (0.5 + 0.5&middot;A&middot;sin(2&pi;t/period + &phi;)) + N(0, ambient)</code>,
a world-frame sideways force at the centre of mass with no torque. A and &phi; are drawn once per
flight, so the gust is coherent for the whole crossing; the ambient term is redrawn every control
step. Grid 61 &times; 43 over x in [-1, 1] m and z in [0.1, 1.5] m.</p>
</div>

<script>
const D = /*DATA*/;
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const isDark = () => matchMedia('(prefers-color-scheme:dark)').matches
  ? document.documentElement.getAttribute('data-theme') !== 'light'
  : document.documentElement.getAttribute('data-theme') === 'dark';

function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function ramp(){ return (isDark() ? D.rampDark : D.rampLight).map(hex2rgb); }
function colorAt(p, r){
  const t = Math.max(0, Math.min(1, p)) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(t)), f = t - i;
  const a = r[i], b = r[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f].map(Math.round);
}
const fmt = (v, d=2) => v.toFixed(d);
const pct = (n, tot) => (100*n/tot).toFixed(1) + '%';

function condLabel(p){
  const b = [];
  if (p.zd) b.push('ż = ' + p.zd);
  if (p.td) b.push('θ̇ = ' + p.td);
  if (p.xd) b.push('ẋ = ' + p.xd);
  return b.length ? b.join(', ') : 'at rest';
}
function cfgLabel(p){
  if (p.config === 'baseline') return 'baseline — no disturbance, 1 flight/start';
  return p.config + ' — F_max ' + p.f_max + ' N, ambient ' + p.ambient + ', ' + p.trials + ' flights/start';
}

const CANVASES = [];
function draw(panel, cv){
  const r = ramp(), dpr = Math.min(2, devicePixelRatio || 1);
  const W = cv.clientWidth || 300, H = Math.round(W * 0.72);
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const PAD_L = 34, PAD_B = 26, PAD_T = 6, PAD_R = 6;
  const pw = W - PAD_L - PAD_R, ph = H - PAD_T - PAD_B;
  const nx = panel.nx, nz = panel.nz;
  const cw = pw / nx, ch = ph / nz;
  // z increases upward on screen, so row 0 (lowest z) draws at the bottom.
  for (let i = 0; i < nz; i++){
    for (let j = 0; j < nx; j++){
      const c = colorAt(panel.p[i*nx + j], r);
      g.fillStyle = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
      g.fillRect(PAD_L + j*cw, PAD_T + (nz-1-i)*ch, Math.ceil(cw)+0.5, Math.ceil(ch)+0.5);
    }
  }
  const zmin = panel.zs[0], zmax = panel.zs[nz-1];
  const xmin = panel.xs[0], xmax = panel.xs[nx-1];
  const zy = z => PAD_T + ph * (1 - (z - zmin)/(zmax - zmin));
  const xX = x => PAD_L + pw * (x - xmin)/(xmax - xmin);
  g.strokeStyle = cssv('--ink'); g.globalAlpha = 0.55; g.lineWidth = 1; g.setLineDash([4,3]);
  for (const z of D.band){ g.beginPath(); g.moveTo(PAD_L, zy(z)); g.lineTo(W-PAD_R, zy(z)); g.stroke(); }
  g.setLineDash([]); g.globalAlpha = 1;
  // The goal ball drawn at its true radius in data units. x and z are scaled
  // independently (the panel is not square), so it renders as an ellipse.
  g.strokeStyle = cssv('--ink'); g.lineWidth = 1.5;
  g.beginPath();
  g.ellipse(xX(0), zy(D.goal_z),
            pw * D.goal_r/(xmax - xmin), ph * D.goal_r/(zmax - zmin), 0, 0, 6.2832);
  g.stroke();
  g.strokeStyle = cssv('--axis'); g.lineWidth = 1;
  g.strokeRect(PAD_L, PAD_T, pw, ph);
  g.fillStyle = cssv('--muted'); g.font = '10px system-ui,sans-serif';
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const z of [0.5, 1.0, 1.5]) g.fillText(z.toFixed(1), PAD_L - 5, zy(z));
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const x of [-1, 0, 1]) g.fillText(x.toFixed(0), xX(x), PAD_T + ph + 5);
  g.fillText('x (m)', PAD_L + pw/2, PAD_T + ph + 15);
  g.save(); g.translate(9, PAD_T + ph/2); g.rotate(-Math.PI/2);
  g.textBaseline = 'middle'; g.fillText('z (m)', 0, 0); g.restore();
  cv._geom = {PAD_L, PAD_T, pw, ph, nx, nz};
}

function build(){
  const host = document.getElementById('rows'); host.innerHTML = '';
  const tb = document.querySelector('#stats tbody'); tb.innerHTML = '';
  for (const row of D.rows){
    const head = document.createElement('div'); head.className = 'rowhead';
    const h = document.createElement('h2'); h.textContent = row.title;
    const s = document.createElement('p'); s.className = 'sub'; s.textContent = row.subtitle;
    head.appendChild(h); head.appendChild(s); host.appendChild(head);
    const box = document.createElement('div'); box.className = 'panels';
    for (const panel of row.panels){
      const d = document.createElement('div'); d.className = 'panel';
      const cv = document.createElement('canvas');
      const ro = document.createElement('p'); ro.className = 'readout';
      ro.textContent = 'Hover the map for exact values.';
      d.innerHTML = '<p class="name">' + panel.config + '</p>' +
                    '<p class="cfg">' + cfgLabel(panel) + '</p>';
      d.appendChild(cv); d.appendChild(ro); box.appendChild(d);
      CANVASES.push([panel, cv]);
      cv.addEventListener('mousemove', e => {
        const g = cv._geom; if (!g) return;
        const r = cv.getBoundingClientRect();
        const px = e.clientX - r.left - g.PAD_L, py = e.clientY - r.top - g.PAD_T;
        if (px < 0 || py < 0 || px > g.pw || py > g.ph){ ro.textContent = 'Hover the map for exact values.'; return; }
        const j = Math.min(g.nx-1, Math.floor(px / (g.pw/g.nx)));
        const i = g.nz - 1 - Math.min(g.nz-1, Math.floor(py / (g.ph/g.nz)));
        const p = panel.p[i*g.nx + j];
        const hits = Math.round(p * panel.trials);
        ro.textContent = 'x = ' + fmt(panel.xs[j]) + ' m, z = ' + fmt(panel.zs[i]) + ' m\n' +
          'reached the goal ' + hits + ' of ' + panel.trials + '  (p = ' + fmt(p) + ')';
      });
      cv.addEventListener('mouseleave', () => { ro.textContent = 'Hover the map for exact values.'; });

      const tr = document.createElement('tr');
      const dot = '<span class="dot" style="background:' + (panel.config === 'baseline'
        ? 'var(--muted)' : panel.config === 'sharp' ? 'var(--fail)' : 'var(--ok)') + '"></span>';
      tr.innerHTML = '<td>' + row.title + '</td><td class="cfgcell">' + dot + panel.config + '</td>' +
        '<td>' + fmt(panel.mean_p, 3) + '</td>' +
        '<td>' + panel.n_certain_ok.toLocaleString() + ' (' + pct(panel.n_certain_ok, panel.n) + ')</td>' +
        '<td>' + panel.n_certain_fail.toLocaleString() + ' (' + pct(panel.n_certain_fail, panel.n) + ')</td>' +
        '<td>' + panel.n_fuzzy.toLocaleString() + ' (' + pct(panel.n_fuzzy, panel.n) + ')</td>';
      tb.appendChild(tr);
    }
    host.appendChild(box);
  }
}

function paintLegend(){
  const r = ramp();
  document.getElementById('legendbar').style.background =
    'linear-gradient(90deg,' + r.map((c,i) =>
      'rgb(' + c.join(',') + ') ' + (100*i/(r.length-1)).toFixed(1) + '%').join(',') + ')';
}
function redrawAll(){ paintLegend(); for (const [p, cv] of CANVASES) draw(p, cv); }

build();
requestAnimationFrame(redrawAll);
addEventListener('resize', redrawAll);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', redrawAll);
new MutationObserver(redrawAll).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});
</script>
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slice_dir', default='.', help='directory holding slice_*.npz')
    ap.add_argument('--out', default='corridor_panels.html')
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.slice_dir, 'slice_*.npz')))
    if not paths:
        raise SystemExit(f'[ERROR] q2_corridor_panels.py: no slice_*.npz under {args.slice_dir!r}')
    panels = {os.path.basename(p): load_panel(p) for p in paths}

    named = {f for _, _, files in ROWS for f in files}
    missing = named - set(panels)
    extra = set(panels) - named
    if missing:
        print(f'[warn] named in ROWS but not on disk, skipped: {sorted(missing)}')
    if extra:
        print(f'[warn] on disk but not placed in any row, NOT rendered: {sorted(extra)}')

    out = build(panels, args.out)
    total = sum(len(r['panels']) for r in
                [dict(panels=[panels[f] for f in files if f in panels]) for _, _, files in ROWS])
    print(f'{out}: {total} panels, {os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
