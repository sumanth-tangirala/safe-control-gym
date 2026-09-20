'''Draw the quad3d twin-curtain disturbance, ambient term excluded.

Explanatory, not a data product: it plots the closed-form law rather than any
collected rollout, so it is exact and costs no simulation. Every constant is
imported from q3_corridor_common, so the figure cannot drift from the code the
way a hand-drawn diagram would.

Deliberately NOT a 3-D surface. The curtain term factorises,

    w(x, t) = sigma(x) * g(t),   g(t) = 0.5 + 0.5*A*sin(2*pi*t/T + phi)

so a surface over (x, t) is the outer product of two 1-D curves and shows
nothing the factors do not. It also buries where sigma crosses the 1%-of-peak
threshold, which is the line that defines the band and drives the reachability
test in q3_corridor_collect.

Usage:
  python q3_corridor_noise_figure.py --out corridor_noise.html
'''
import argparse
import json
import math

import numpy as np

from q3_corridor_common import BAND, SINE_PERIOD, WIDTH, X_C, sigma

# The ladder proposed off the 8-level sweep, plus the calm reference.
LEVELS = [0.0, 0.20, 0.30]
CROSS_SPEED = 1.0     # m/s, a straight constant-speed pass used for the felt-gust panel


def build_data():
    xs = np.linspace(-2.2, 2.2, 441)
    env = {f'{L:g}': [round(float(sigma(x, L)), 5) for x in xs] for L in LEVELS}

    # g(t) for a few per-rollout draws. These are SAMPLES of one process, not
    # separate quantities, so the page renders them as one hue at varying
    # lightness rather than as categorical series.
    rng = np.random.default_rng(20260817)
    ts = np.linspace(0, 2 * SINE_PERIOD, 241)
    draws = []
    for _ in range(5):
        A = float(rng.uniform(0.0, 1.0))
        phi = float(rng.uniform(-math.pi, math.pi))
        draws.append(dict(A=round(A, 3), phi=round(phi, 3),
                          g=[round(0.5 + 0.5 * A * math.sin(2 * math.pi / SINE_PERIOD * t + phi), 5)
                             for t in ts]))

    # The gust as felt on a straight pass at CROSS_SPEED through both curtains,
    # for one draw. This is the only view the controller ever sees.
    A, phi = draws[0]['A'], draws[0]['phi']
    t_cross = np.linspace(0, 4.4 / CROSS_SPEED, 441)
    x_of_t = -2.2 + CROSS_SPEED * t_cross
    felt = {f'{L:g}': [round(float(sigma(x, L))
                             * (0.5 + 0.5 * A * math.sin(2 * math.pi / SINE_PERIOD * t + phi)), 5)
                       for x, t in zip(x_of_t, t_cross)] for L in LEVELS if L > 0}

    return dict(xs=[round(float(v), 4) for v in xs],
                ts=[round(float(v), 4) for v in ts],
                t_cross=[round(float(v), 4) for v in t_cross],
                env=env, draws=draws, felt=felt, levels=LEVELS,
                band=[[round(b[0], 4), round(b[1], 4)] for b in BAND],
                x_c=X_C, width=WIDTH, period=SINE_PERIOD, speed=CROSS_SPEED,
                sigma_peak={f'{L:g}': round(float(sigma(X_C, L)), 5) for L in LEVELS})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='corridor_noise.html')
    args = ap.parse_args(argv)
    payload = json.dumps(build_data(), separators=(',', ':'))
    with open(args.out, 'w') as fh:
        fh.write(TEMPLATE.replace('/*DATA*/', payload))
    print(f'{args.out}: written')


TEMPLATE = r'''<title>The quad3d twin-curtain disturbance</title>
<style>
:root{
  color-scheme:light;
  --page:#f7f8fa; --surface:#ffffff; --sunk:#eef1f5;
  --ink:#111318; --ink2:#4a4f58; --muted:#7f8794;
  --rule:#e2e6ec; --axis:#c2c8d2; --ring:rgba(17,19,24,0.09);
  --c1:#0d366b; --c2:#256abf; --c3:#3987e5; --c4:#86b6ef; --c5:#b7d3f6;
  --warm:#a02525;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
  --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
  --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
  --c1:#9ec5f4; --c2:#6da7ec; --c3:#3987e5; --c4:#2a6ab0; --c5:#1c4a80;
  --warm:#e88080;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
  --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
  --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
  --c1:#9ec5f4; --c2:#6da7ec; --c3:#3987e5; --c4:#2a6ab0; --c5:#1c4a80;
  --warm:#e88080;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 var(--sans);
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:52px 24px 80px}
h1{font-size:29px;line-height:1.2;margin:0 0 14px;letter-spacing:-0.021em;font-weight:640;
  text-wrap:balance;max-width:20ch}
h2{font-size:14px;margin:0;letter-spacing:0.09em;text-transform:uppercase;
  font-family:var(--mono);font-weight:600}
p{margin:0 0 14px;max-width:68ch;color:var(--ink2)}
.lede{font-size:17px;line-height:1.55;color:var(--ink);max-width:62ch;margin-bottom:8px}
.sub{color:var(--muted);font-size:13.5px}
code{font:0.875em/1.5 var(--mono);background:var(--sunk);border-radius:3px;padding:1.5px 5px}
.eq{background:var(--surface);border:1px solid var(--ring);border-radius:8px;
  padding:16px 20px;margin:0 0 26px;font:14px/1.9 var(--mono);color:var(--ink);
  overflow-x:auto;white-space:pre}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  padding:0 0 12px;border-bottom:1px solid var(--rule);margin:36px 0 16px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:16px}
svg{display:block;width:100%;height:auto;overflow:visible}
.key{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 0;font:11.5px/1.5 var(--mono);
  color:var(--muted)}
.key i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
.note{border-left:2px solid var(--axis);padding:1px 0 1px 16px;margin:0 0 16px;
  color:var(--ink2);max-width:68ch}
.foot{color:var(--muted);font-size:12.5px;line-height:1.7;margin-top:40px;
  border-top:1px solid var(--rule);padding-top:18px;max-width:78ch}
</style>
<div class="wrap">
<h1>The quad3d twin-curtain disturbance</h1>
<p class="lede">Two walls of disturbed air standing at x = &plusmn;0.9 m. The ambient term is
left out here, so this is the curtain alone, plotted from its closed form rather than from any
rollout.</p>

<div class="eq">w(x, t) = sigma(x) &middot; g(t)

sigma(x) = F_max &middot; [ exp(-(x-0.9)&sup2;/2&middot;0.25&sup2;) + exp(-(x+0.9)&sup2;/2&middot;0.25&sup2;) ]
g(t)     = 0.5 + 0.5&middot;A&middot;sin(2&pi;t/2.0 + phi),   A ~ U(0,1), phi ~ U(-&pi;,&pi;)</div>

<div class="note">
<p>A and phi are drawn once per flight and then held. That is what makes the gust coherent
across a crossing instead of averaging itself away, which is the failure the per-step uniform
draw ran into.</p>
</div>

<div class="head"><h2>Where the force lives</h2><p class="sub">sigma(x), the envelope</p></div>
<div class="card"><svg id="s1" viewBox="0 0 900 300" role="img"
  aria-label="Envelope sigma of x, twin Gaussian peaks at plus and minus 0.9 metres"></svg>
<div class="key" id="k1"></div></div>
<p class="sub" style="margin-top:12px">Dashed verticals mark where the envelope falls to 1% of a
single curtain's peak. Those four crossings are the band, and they are why the reachability test in
<code>q3_corridor_collect.py</code> checks two intervals rather than one.</p>

<div class="head"><h2>How it varies in time</h2><p class="sub">g(t), five draws</p></div>
<div class="card"><svg id="s2" viewBox="0 0 900 260" role="img"
  aria-label="Time factor g of t for five per-rollout draws"></svg>
<div class="key" id="k2"></div></div>
<p class="sub" style="margin-top:12px">Each line is one flight. g stays in [0, 1] and averages 0.5
over a period, so the mean force is set by sigma and the draw only moves the swing around it.</p>

<div class="head"><h2>The product</h2><p class="sub">w(x, t) for one draw, F_max 0.30 N</p></div>
<div class="card"><svg id="s3" viewBox="0 0 900 300" role="img"
  aria-label="Heat map of the disturbance over position and time"></svg>
<div class="key" id="k3"></div></div>
<p class="sub" style="margin-top:12px">Vertical banding is g, horizontal banding is sigma. Nothing
else is in there, which is the argument against drawing this as a 3-D surface.</p>

<div class="head"><h2>What the drone feels</h2><p class="sub">w along a straight pass at 1 m/s</p></div>
<div class="card"><svg id="s4" viewBox="0 0 900 280" role="img"
  aria-label="Force felt over time on a straight crossing through both curtains"></svg>
<div class="key" id="k4"></div></div>
<p class="sub" style="margin-top:12px">The only view the controller ever gets. Two pulses, one per
curtain, each shaped by wherever the sinusoid happens to be during that pass.</p>

<p class="foot">Body weight is 0.027 kg &times; 9.81 = 0.265 N, so F_max 0.30 N is a sideways push
of about 113% of weight at a curtain's peak. The two curtains draw independently, so the mean force
at x is sigma(x)/2 rather than sigma(x); the envelope plotted here is the sum of both peaks.
Constants are imported live from <code>q3_corridor_common.py</code>.</p>
</div>

<script>
const D = /*DATA*/;
const cv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const RAMP = ['--c1','--c2','--c3','--c4','--c5'];

function frame(svg, W, H, pad, xdom, ydom, xlab, ylab, xticks, yticks){
  const [L,R,T,B] = pad;
  const pw = W-L-R, ph = H-T-B;
  const sx = v => L + pw*(v-xdom[0])/(xdom[1]-xdom[0]);
  const sy = v => T + ph*(1-(v-ydom[0])/(ydom[1]-ydom[0]));
  let g = '';
  for (const t of yticks) g += `<line x1="${L}" y1="${sy(t)}" x2="${W-R}" y2="${sy(t)}"
      stroke="${cv('--rule')}" stroke-width="1"/>
      <text x="${L-8}" y="${sy(t)}" text-anchor="end" dominant-baseline="middle"
      font-size="11" font-family="${cv('--mono')}" fill="${cv('--muted')}">${t}</text>`;
  for (const t of xticks) g += `<text x="${sx(t)}" y="${H-B+18}" text-anchor="middle"
      font-size="11" font-family="${cv('--mono')}" fill="${cv('--muted')}">${t}</text>`;
  g += `<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="${cv('--axis')}"/>`;
  g += `<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="${cv('--axis')}"/>`;
  g += `<text x="${L+pw/2}" y="${H-B+38}" text-anchor="middle" font-size="11.5"
      font-family="${cv('--mono')}" fill="${cv('--muted')}">${xlab}</text>`;
  g += `<text transform="translate(${L-42},${T+ph/2}) rotate(-90)" text-anchor="middle"
      font-size="11.5" font-family="${cv('--mono')}" fill="${cv('--muted')}">${ylab}</text>`;
  return {g, sx, sy, L, R, T, B, pw, ph};
}
const path = (xs, ys, sx, sy) =>
  xs.map((x,i)=>`${i?'L':'M'}${sx(x).toFixed(1)} ${sy(ys[i]).toFixed(1)}`).join('');

function envelope(){
  const W=900,H=300, ymax = Math.max(...D.env['0.3'])*1.12;
  const f = frame(document.getElementById('s1'), W,H,[62,20,16,52], [-2.2,2.2],[0,ymax],
    'x (m)','force (N)', [-2,-1,0,1,2], [0,0.2,0.4,0.6]);
  let g = f.g;
  for (const [lo,hi] of D.band) for (const e of [lo,hi])
    g += `<line x1="${f.sx(e)}" y1="${f.T}" x2="${f.sx(e)}" y2="${H-f.B}"
      stroke="${cv('--ink')}" stroke-width="1" stroke-dasharray="4 3" opacity="0.45"/>`;
  const order = ['0.3','0.2','0'];
  order.forEach((k,i) => { if (!D.env[k]) return;
    g += `<path d="${path(D.xs, D.env[k], f.sx, f.sy)}" fill="none"
      stroke="${cv(RAMP[i])}" stroke-width="2"/>`; });
  document.getElementById('s1').innerHTML = g;
  document.getElementById('k1').innerHTML = order.map((k,i) =>
    `<span><i style="background:${cv(RAMP[i])}"></i>F_max ${k} N</span>`).join('')
    + `<span><i style="background:${cv('--ink')};opacity:.45"></i>1% band edge</span>`;
}

function timefactor(){
  const W=900,H=260;
  const f = frame(document.getElementById('s2'), W,H,[62,20,16,52], [0,D.ts[D.ts.length-1]],[0,1],
    't (s)','g(t)', [0,1,2,3,4], [0,0.5,1]);
  let g = f.g;
  g += `<line x1="${f.L}" y1="${f.sy(0.5)}" x2="${W-20}" y2="${f.sy(0.5)}"
    stroke="${cv('--warm')}" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>`;
  D.draws.forEach((d,i) => {
    g += `<path d="${path(D.ts, d.g, f.sx, f.sy)}" fill="none"
      stroke="${cv(RAMP[i])}" stroke-width="1.8"/>`; });
  document.getElementById('s2').innerHTML = g;
  document.getElementById('k2').innerHTML = D.draws.map((d,i) =>
    `<span><i style="background:${cv(RAMP[i])}"></i>A=${d.A} phi=${d.phi}</span>`).join('')
    + `<span><i style="background:${cv('--warm')};opacity:.7"></i>mean 0.5</span>`;
}

function heat(){
  const W=900,H=300, L=62,R=20,T=16,B=52;
  const nx=D.xs.length, nt=120;
  const d0 = D.draws[0];
  const pw=W-L-R, ph=H-T-B, cw=pw/nt, ch=ph/nx;
  const fmax=0.30, peak=Math.max(...D.env['0.3']);
  let g='';
  for (let j=0;j<nt;j++){
    const t = j/(nt-1)*2*D.period;
    const gt = 0.5+0.5*d0.A*Math.sin(2*Math.PI/D.period*t + d0.phi);
    for (let i=0;i<nx;i+=3){
      const v = D.env['0.3'][i]*gt/peak;
      const c = Math.round(255-v*205), b=Math.round(255-v*120);
      g += `<rect x="${(L+j*cw).toFixed(1)}" y="${(T+ (nx-1-i)*ch).toFixed(1)}"
        width="${(cw+0.6).toFixed(1)}" height="${(ch*3+0.6).toFixed(1)}"
        fill="rgb(${c},${c},${b})"/>`;
    }
  }
  g += `<rect x="${L}" y="${T}" width="${pw}" height="${ph}" fill="none" stroke="${cv('--axis')}"/>`;
  for (const t of [0,1,2,3,4]) g += `<text x="${L+pw*t/(2*D.period)}" y="${H-B+18}"
    text-anchor="middle" font-size="11" font-family="${cv('--mono')}"
    fill="${cv('--muted')}">${t}</text>`;
  for (const x of [-2,-1,0,1,2]) g += `<text x="${L-8}" y="${T+ph*(1-(x+2.2)/4.4)}"
    text-anchor="end" dominant-baseline="middle" font-size="11" font-family="${cv('--mono')}"
    fill="${cv('--muted')}">${x}</text>`;
  g += `<text x="${L+pw/2}" y="${H-B+38}" text-anchor="middle" font-size="11.5"
    font-family="${cv('--mono')}" fill="${cv('--muted')}">t (s)</text>`;
  g += `<text transform="translate(${L-42},${T+ph/2}) rotate(-90)" text-anchor="middle"
    font-size="11.5" font-family="${cv('--mono')}" fill="${cv('--muted')}">x (m)</text>`;
  document.getElementById('s3').innerHTML = g;
  document.getElementById('k3').innerHTML =
    `<span>pale = no force &middot; dark = ${peak.toFixed(2)} N &middot; draw A=${d0.A} phi=${d0.phi}</span>`;
}

function felt(){
  const W=900,H=280, keys=Object.keys(D.felt);
  const ymax = Math.max(...keys.flatMap(k=>D.felt[k]))*1.15;
  const f = frame(document.getElementById('s4'), W,H,[62,20,16,52],
    [0,D.t_cross[D.t_cross.length-1]],[0,ymax],'t (s)','force felt (N)',[0,1,2,3,4],[0,0.2,0.4]);
  let g = f.g;
  keys.sort((a,b)=>b-a).forEach((k,i) => {
    g += `<path d="${path(D.t_cross, D.felt[k], f.sx, f.sy)}" fill="none"
      stroke="${cv(RAMP[i])}" stroke-width="2"/>`; });
  document.getElementById('s4').innerHTML = g;
  document.getElementById('k4').innerHTML = keys.map((k,i) =>
    `<span><i style="background:${cv(RAMP[i])}"></i>F_max ${k} N</span>`).join('')
    + `<span>pass at ${D.speed} m/s from x=-2.2 to +2.2</span>`;
}

function draw(){ envelope(); timefactor(); heat(); felt(); }
draw();
addEventListener('resize', draw);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
'''

if __name__ == '__main__':
    main()
