'''Render the quad3d slice grid as one self-contained HTML page.

Reads the slice_*.npz files q3_corridor_slice.py writes and lays them out as a
matrix: one row per axis, one column per (F_max, ambient) config. Each row
carries its own deterministic reference, so every disturbed panel can be
compared against the no-disturbance boundary in the same slice.

That comparison is the point, and it is what the quad2d panels could not do.
Per cell it reports which of four things happened:

  rescued  the deterministic run failed here, and the disturbance sometimes
           gets it to the goal
  broken   the deterministic run succeeded, and the disturbance sometimes
           does not
  fuzzy    p strictly between 0 and 1, whichever way the deterministic run went
  settled  the disturbance changed nothing

The quad2d campaign measured 70.7% of deterministically-succeeding starts broken
against 2.0% of failing starts rescued, because only 6.2% of that eval set
succeeded at all so the two denominators differed 15-fold. quad3d's
deterministic success is 22.0%, so the split should come out closer to even.

Usage:
  python q3_corridor_panels.py --slice_dir /scratch/dm1487/q3slices \\
      --out q3_panels.html
'''
import argparse
import glob
import json
import os
import re

import numpy as np

AXIS_LABEL = {
    'qw': 'qw   (1 = level, 0 = inverted)',
    'r': 'r   yaw rate (rad/s)',
    'z_dot': 'z_dot   fall rate (m/s)',
}
AXIS_BLURB = {
    'qw': 'Steepest gradient of any coordinate, 0.08 to 0.67 across the shipped '
          'set. The only axis reaching past 0.5, so the only one where rescued '
          'and broken can both be populous.',
    'r': 'Success peaks near zero yaw rate and falls off both ways, so the panel '
         'holds two boundaries. Its deterministic gate came back at 0.967 '
         'though, so the benign background may leave little to rescue.',
    'z_dot': 'Bottom decile of the shipped set succeeds only 2% of the time, so '
             'falling hard is close to fatal regardless of the curtain.',
}
NAME_RE = re.compile(r'^slice_(?P<axis>qw|r|z_dot)_f(?P<f>[0-9.]+)_'
                     r'a(?P<a>none|[0-9.]+)_k(?P<k>\d+)\.npz$')


def load(path):
    m = NAME_RE.match(os.path.basename(path))
    if m is None:
        return None
    d = np.load(path, allow_pickle=True)
    p = d['p']
    return dict(axis=m['axis'], f_max=float(m['f']),
                ambient=None if m['a'] == 'none' else float(m['a']),
                trials=int(m['k']), p=p,
                xs=[round(float(v), 4) for v in d['xs']],
                avals=[round(float(v), 4) for v in d['avals']],
                nx=int(p.shape[1]), na=int(p.shape[0]))


def compare(p, det):
    '''-> per-panel counts against that axis's deterministic reference.'''
    ok = det >= 1
    resc = int(((~ok) & (p > 0)).sum())
    brok = int((ok & (p < 1)).sum())
    fz = int(((p > 0) & (p < 1)).sum())
    return dict(rescued=resc, broken=brok, fuzzy=fz, n=int(p.size),
                det_ok=int(ok.sum()),
                mean_p=round(float(p.mean()), 4),
                det_mean=round(float(det.mean()), 4))


def build(slice_dir):
    panels = [q for q in (load(f) for f in sorted(glob.glob(
        os.path.join(slice_dir, 'slice_*.npz')))) if q is not None]
    if not panels:
        raise SystemExit(f'[ERROR] q3_corridor_panels.py: no slice_*.npz in {slice_dir!r}')

    rows = []
    for axis in ('qw', 'r', 'z_dot'):
        mine = [q for q in panels if q['axis'] == axis]
        if not mine:
            continue
        det = next((q for q in mine if q['f_max'] == 0 and q['ambient'] is None), None)
        if det is None:
            print(f'[warn] {axis}: no deterministic reference (f0_anone_k1); '
                  f'rescued/broken cannot be computed for this row')
        others = sorted((q for q in mine if q is not det),
                        key=lambda q: (q['f_max'], q['ambient'] or 0))
        out = []
        for q in [det] + others if det else others:
            e = dict(f_max=q['f_max'], ambient=q['ambient'], trials=q['trials'],
                     is_det=(q is det),
                     p=[round(float(v), 3) for v in q['p'].ravel()],
                     nx=q['nx'], na=q['na'])
            if det is not None:
                e.update(compare(q['p'], det['p']))
            out.append(e)
        rows.append(dict(axis=axis, label=AXIS_LABEL[axis], blurb=AXIS_BLURB[axis],
                         xs=mine[0]['xs'], avals=mine[0]['avals'], panels=out))
    return dict(rows=rows, x_lo=-1.8, x_hi=1.8, curtains=[-0.9, 0.9],
                band=[[0.141, 1.659], [-1.659, -0.141]])


TEMPLATE = r'''<title>quad3d twin curtain: where the boundary moves</title>
<style>
:root{
  color-scheme:light;
  --page:#f7f8fa; --surface:#ffffff; --sunk:#eef1f5;
  --ink:#111318; --ink2:#4a4f58; --muted:#7f8794;
  --rule:#e2e6ec; --axis:#c2c8d2; --ring:rgba(17,19,24,0.09);
  --fail:#a02525; --ok:#256abf; --neutral:#f0efec;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
  --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
  --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
  --fail:#e88080; --ok:#6da7ec; --neutral:#383835;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0e1014; --surface:#171a21; --sunk:#1e222b;
  --ink:#f2f4f7; --ink2:#b6bcc7; --muted:#7f8794;
  --rule:#262a33; --axis:#39404c; --ring:rgba(242,244,247,0.10);
  --fail:#e88080; --ok:#6da7ec; --neutral:#383835;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 var(--sans);
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1400px;margin:0 auto;padding:52px 24px 80px}
h1{font-size:29px;line-height:1.2;margin:0 0 14px;letter-spacing:-0.021em;font-weight:640;
  text-wrap:balance;max-width:24ch}
h2{font-size:14px;margin:0;letter-spacing:0.09em;text-transform:uppercase;
  font-family:var(--mono);font-weight:600}
p{margin:0 0 14px;max-width:70ch;color:var(--ink2)}
.lede{font-size:17px;line-height:1.55;color:var(--ink);max-width:64ch}
.sub{color:var(--muted);font-size:13.5px}
code{font:0.875em/1.5 var(--mono);background:var(--sunk);border-radius:3px;padding:1.5px 5px}
.rowhead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  padding:0 0 10px;border-bottom:1px solid var(--rule);margin:38px 0 6px}
.strip{display:flex;gap:12px;overflow-x:auto;padding:14px 2px 6px;-webkit-overflow-scrolling:touch}
.cell{background:var(--surface);border:1px solid var(--ring);border-radius:8px;
  padding:11px;flex:0 0 210px}
.cell.det{border-color:var(--axis)}
.cell .t{font:11.5px/1.4 var(--mono);color:var(--ink);margin:0 0 2px;font-weight:600}
.cell .s{font:10.5px/1.4 var(--mono);color:var(--muted);margin:0 0 8px}
canvas{width:100%;height:auto;display:block;border-radius:3px;background:var(--sunk)}
.stat{font:10.5px/1.55 var(--mono);color:var(--ink2);margin:8px 0 0;
  font-variant-numeric:tabular-nums}
.stat b{color:var(--ink);font-weight:600}
.scale{display:flex;align-items:center;gap:12px;margin:6px 0 22px;flex-wrap:wrap}
.bar{height:10px;flex:0 0 220px;border-radius:2px;border:1px solid var(--ring)}
.scale span{font:11.5px/1.4 var(--mono);color:var(--muted)}
.note{border-left:2px solid var(--axis);padding:1px 0 1px 16px;margin:0 0 16px;
  color:var(--ink2);max-width:70ch}
.foot{color:var(--muted);font-size:12.5px;line-height:1.7;margin-top:40px;
  border-top:1px solid var(--rule);padding-top:18px;max-width:80ch}
</style>
<div class="wrap">
<h1>quad3d twin curtain: where the boundary moves</h1>
<p class="lede">Two walls of disturbed air at x = &plusmn;0.9 m. Each row sweeps x against a
coordinate that actually decides success, and each panel is one disturbance setting. The first
panel in every row is the no-disturbance reference.</p>

<div class="scale">
  <span>p = 0</span><div class="bar" id="legendbar"></div><span>p = 1</span>
  <span>dashed = curtain centres</span>
</div>

<div class="note">
<p>Position does not decide quad3d outcomes, which is why none of these plot x against y or z.
Measured over 300,000 rows of the shipped deterministic set, success by decile spans 0.13 to 0.27
across x, y and z, against 0.08 to 0.67 across <code>qw</code>. An (x, z) panel comes out
near-uniform and says nothing about how the curtain moves the boundary.</p>
</div>

<div id="rows"></div>

<p class="foot">Grid 61 &times; 43, generated rather than filtered: the shipped eval set is
1,000,000 randomly drawn 13-D points, so no two share the other eleven coordinates. Off-axis
coordinates are held benign, centred in y, 1.5 m up, level, still. Rescued counts cells the
deterministic run failed and the disturbance sometimes carries to the goal; broken counts the
reverse. Body weight is 0.265 N, so F_max 0.30 is a sideways push of about 113% of weight at a
curtain peak.</p>
</div>

<script>
const D = /*DATA*/;
const cv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const RAMP_L = ['#7d1a1a','#a02525','#c23434','#dc5c5c','#eda0a0','#f0efec',
                '#b7d3f6','#86b6ef','#3987e5','#256abf','#0d366b'];
const RAMP_D = ['#f0a0a0','#e88080','#de6060','#c04d4d','#7a4040','#383835',
                '#35507a','#2a6ab0','#3987e5','#6da7ec','#9ec5f4'];
const isDark = () => matchMedia('(prefers-color-scheme:dark)').matches
  ? document.documentElement.getAttribute('data-theme') !== 'light'
  : document.documentElement.getAttribute('data-theme') === 'dark';
const h2r = h => [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
const ramp = () => (isDark()?RAMP_D:RAMP_L).map(h2r);
function col(p, r){
  const t = Math.max(0,Math.min(1,p))*(r.length-1), i=Math.min(r.length-2,Math.floor(t)), f=t-i;
  const a=r[i], b=r[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f].map(Math.round);
}
const pct = (a,b) => b ? (100*a/b).toFixed(1)+'%' : 'n/a';
const CANV = [];

function draw(pan, row, cvs){
  const r = ramp(), dpr = Math.min(2, devicePixelRatio||1);
  const W = cvs.clientWidth||188, H = Math.round(W*0.72);
  cvs.width = W*dpr; cvs.height = H*dpr;
  const g = cvs.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  const PL=22, PB=16, pw=W-PL-2, ph=H-PB-2;
  const nx=pan.nx, na=pan.na, cw=pw/nx, ch=ph/na;
  for (let a=0;a<na;a++) for (let x=0;x<nx;x++){
    const c = col(pan.p[a*nx+x], r);
    g.fillStyle = 'rgb('+c.join(',')+')';
    g.fillRect(PL+x*cw, 2+(na-1-a)*ch, Math.ceil(cw)+0.5, Math.ceil(ch)+0.5);
  }
  g.strokeStyle = cv('--ink'); g.globalAlpha=0.5; g.lineWidth=1; g.setLineDash([3,3]);
  for (const xc of D.curtains){
    const px = PL + pw*(xc-D.x_lo)/(D.x_hi-D.x_lo);
    g.beginPath(); g.moveTo(px,2); g.lineTo(px,2+ph); g.stroke();
  }
  g.setLineDash([]); g.globalAlpha=1;
  g.strokeStyle = cv('--axis'); g.strokeRect(PL,2,pw,ph);
  g.fillStyle = cv('--muted'); g.font='9px '+cv('--mono');
  g.textAlign='center'; g.textBaseline='top';
  g.fillText('-1.8', PL+6, 2+ph+3); g.fillText('0', PL+pw/2, 2+ph+3);
  g.fillText('1.8', PL+pw-6, 2+ph+3);
  g.save(); g.translate(8, 2+ph/2); g.rotate(-Math.PI/2); g.textBaseline='middle';
  g.fillText(row.axis, 0, 0); g.restore();
}

function build(){
  const host = document.getElementById('rows'); host.innerHTML='';
  for (const row of D.rows){
    const hd = document.createElement('div'); hd.className='rowhead';
    hd.innerHTML = '<h2>x vs '+row.axis+'</h2><p class="sub">'+row.label+'</p>';
    host.appendChild(hd);
    const bl = document.createElement('p'); bl.className='sub';
    bl.style.margin='6px 0 0'; bl.textContent=row.blurb; host.appendChild(bl);
    const strip = document.createElement('div'); strip.className='strip';
    for (const pan of row.panels){
      const d = document.createElement('div');
      d.className = 'cell' + (pan.is_det ? ' det' : '');
      const title = pan.is_det ? 'no disturbance'
        : 'F_max '+pan.f_max+' N';
      const sub = pan.is_det ? 'K=1, the reference'
        : 'ambient '+(pan.ambient===null?'0':pan.ambient)+', K='+pan.trials;
      d.innerHTML = '<p class="t">'+title+'</p><p class="s">'+sub+'</p>';
      const c = document.createElement('canvas'); d.appendChild(c);
      const s = document.createElement('p'); s.className='stat';
      if (pan.rescued === undefined){
        s.textContent = 'mean p '+pan.mean_p;
      } else {
        s.innerHTML = 'mean p <b>'+pan.mean_p+'</b><br>'
          + 'fuzzy <b>'+pct(pan.fuzzy,pan.n)+'</b><br>'
          + 'rescued <b>'+pan.rescued+'</b> ('+pct(pan.rescued, pan.n-pan.det_ok)+' of det-fails)<br>'
          + 'broken <b>'+pan.broken+'</b> ('+pct(pan.broken, pan.det_ok)+' of det-oks)';
      }
      d.appendChild(s); strip.appendChild(d);
      CANV.push([pan, row, c]);
    }
    host.appendChild(strip);
  }
}
function legend(){
  const r = ramp();
  document.getElementById('legendbar').style.background =
    'linear-gradient(90deg,'+r.map((c,i)=>'rgb('+c.join(',')+') '+(100*i/(r.length-1)).toFixed(1)+'%').join(',')+')';
}
function all(){ legend(); for (const [p,row,c] of CANV) draw(p,row,c); }
build(); requestAnimationFrame(all);
addEventListener('resize', all);
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', all);
new MutationObserver(all).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slice_dir', default='.')
    ap.add_argument('--out', default='q3_panels.html')
    args = ap.parse_args(argv)
    data = build(args.slice_dir)
    payload = json.dumps(data, separators=(',', ':'))
    with open(args.out, 'w') as fh:
        fh.write(TEMPLATE.replace('/*DATA*/', payload))
    n = sum(len(r['panels']) for r in data['rows'])
    print(f'{args.out}: {len(data["rows"])} rows, {n} panels, '
          f'{os.path.getsize(args.out) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
