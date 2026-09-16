'''Paper figures: one pictorial panel per system, sized for a column of three.

Three files, quad2d, quad3d and cartpole, all rendered at the same figure size
and aspect so they sit in one ICRA column without per-panel scaling in LaTeX.
Scaling a panel in LaTeX would change its line weights against its neighbours,
which is the thing this file exists to prevent.

The two quadrotor panels draw the gate envelope sigma / f_max from its closed
form, not from rollouts, so they are exact and one panel covers every level of
a family: the levels differ only in f_max, never in where the band sits. Centre
and width come from q2_corridor_common and q3_corridor_common, the modules the
collectors fly, so a panel cannot drift from the data.

The cartpole panel carries no field, because the family it illustrates has no
noise. It is a proportioned illustration rather than a scale drawing: the real
rail is +-6 m against a 0.65 m pole, which at this panel size puts the cart
under 2 mm wide [user, 2026-09-16].

Every panel carries no text at all: no ticks, axis labels, titles or legend.
The caption has to say what the marks mean:

- Wind is flat black arrows along the push direction, +x on quad2d and +y on
  quad3d. An arrow's length, shaft width and head size all grow with sigma at
  its position.
- The drone icon is the goal hover pose, (x, z) = (0, 1) on quad2d and
  (0, 0, 1) on quad3d, where the quad3d one has a dotted drop line to the floor.
  The cartpole's goal is the dashed cart, upright at the rail centre; the solid
  cart is a start.
- quad2d shades the envelope behind its arrows. quad3d draws each curtain as
  nested translucent slabs, one per contour of the envelope at 0.2, 0.4, 0.6
  and 0.8 of peak, so the number of slabs a point sits inside tracks sigma
  there. Its arrows come in combs of five laid across one curtain, at 0, 0.8
  and 1.6 curtain widths either side of its centre, spread by farthest-point
  sampling on screen so none overlap.
- Thin gray rules are the arena: the published kill box on the quadrotors, the
  rail and its end stops on the cartpole.

--field picks the field colour, gray or light blue. It changes nothing but the
envelope shading and the slab fill; arrows and icons stay black either way.

The ungated ambient term is not drawn. It acts everywhere in the arena and has
no band to show.

Arena bounds are the published families' kill boxes, copied from their
dataset_description.json: quad2d rl/* x in [-1, 1], z in [0.1, 1.5]; quad3d
ppo*/* x, y in [-1.8, 1.8], z in [0.1, 3.0]. The quad3d box is not
q3_corridor_common.STATE_BOUNDS, which is the older LQR family's.

Usage:
  python viz_system_panels.py --field blue --out figures/paper/sys_blue
      # writes sys_blue_{2d,3d,cp}.{pdf,png}, each 1.13 x 0.94 in
'''
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import q2_corridor_common as q2  # noqa: E402
import q3_corridor_common as q3  # noqa: E402

Q2_X = (-1.0, 1.0)
Q2_Z = (0.1, 1.5)
Q2_GOAL = (0.0, 1.0)

Q3_XY = (-1.8, 1.8)
Q3_Z = (0.1, 3.0)
Q3_GOAL = (0.0, 0.0, 1.0)

INK = '#0b0b0b'
INK2 = '#52514e'
GHOST = '#8f8d88'
RULE = '#b9b8b0'
ARROW = '#000000'

# Field colours. White at zero and light enough at the peak that the black
# arrows still carry the strength.
#
# The blue build is NOT a luminance match for the gray one. Converted to
# grayscale, the darkest 2% of the field reads 191/255 against the gray build's
# 183 on quad2d, and 177 against 141 on quad3d. So a reviewer printing in black
# and white sees a weaker field, most of it on quad3d. What the blue build does
# buy is agreement between its own two panels, 191 against 177, where the gray
# build's panels sit 42 levels apart. Measured 2026-09-16 off the rendered PNGs.
FIELDS = {
    'gray': dict(band=['#ffffff', '#e8e8e8', '#d0d0d0', '#b6b6b6'],
                 shell='#6e6e6e', edge='#8c8c8c'),
    'blue': dict(band=['#ffffff', '#e6f0fa', '#cbe1f5', '#abd0ee'],
                 shell='#4f93cc', edge='#9cc4e4'),
}
NORM = Normalize(0.0, 1.0)

# One panel size for all three, so LaTeX places them without \includegraphics
# scaling. Three across an ICRA column (3.5 in) with 0.05 in gutters.
PANEL_W = 1.13
ASPECT = 1.2
# Breathing room past the arena box on every side, as a share of the fitted
# window. Without it a box drawn exactly at the data limits loses half its
# rule width off the edge of the panel.
MARGIN = 0.035

# The quad3d arena spans 3.6 m where quad2d's spans 2 m, and the 3-D projection
# shrinks it further, so its arrows are drawn this much larger in metres.
ARROW_SCALE_3D = 2.5
# Azimuth -68 rather than matplotlib's -60: the two slabs overlap less on
# screen, so the calm gap around the goal stays visibly calm.
ELEV, AZIM = 20, -68
PROJ = 'ortho'
ZOOM_3D = 1.12
N_COMBS_3D = 12
# Arrows of a comb, in curtain widths from its centre. All inside the outer slab.
COMB_OFFSETS = (-1.6, -0.8, 0.0, 0.8, 1.6)
# quad3d volume: one slab per envelope level. Seen straight through the core a
# ray crosses eight faces, so this alpha lands the peak near quad2d's shading.
SHELL_LEVELS = (0.2, 0.4, 0.6, 0.8)
SHELL_ALPHA = 0.055

# Cartpole illustration units: the rail half-length is CP_RAIL. Proportions are
# chosen to read at panel size, not copied from the URDF.
# The rail half-length is set so the scene fills a panel of this aspect without
# padding: 2 * CP_RAIL is ASPECT times the rail-to-pole-tip height.
CP_RAIL = 0.70
CP_CART_W, CP_CART_H = 0.34, 0.20
CP_WHEEL = 0.052
CP_POLE = 0.67
CP_START = (-0.43, 0.45)    # cart x, pole angle in rad, leaning back toward the goal


def g2(z):
    '''quad2d envelope over its peak, in [0, 1].'''
    return np.exp(-0.5 * ((np.asarray(z) - q2.CENTRE) / q2.WIDTH) ** 2)


def g3(x):
    '''quad3d twin-curtain envelope over one curtain's peak. The far curtain
    adds exp(-0.5 * (1.8 / 0.25)**2), about 6e-12, at a peak, so the max is 1.'''
    x = np.asarray(x)
    return (np.exp(-0.5 * ((x - q3.X_C) / q3.WIDTH) ** 2)
            + np.exp(-0.5 * ((x + q3.X_C) / q3.WIDTH) ** 2))


def arrow(s):
    '''Outline of a flat arrow of strength s in [0, 1] as (u, v), tip at the
    origin, pointing +u, in metres of the quad2d arena. Length, shaft width and
    head size all grow with s.'''
    length = 0.12 + 0.24 * s
    head = 0.034 + 0.034 * s
    shaft = 0.003 + 0.008 * s
    barb = 0.011 + 0.017 * s
    return np.array([(-length, -shaft), (-head, -shaft), (-head, -barb), (0.0, 0.0),
                     (-head, barb), (-head, shaft), (-length, shaft)])


def style():
    plt.rcParams.update({
        'axes.linewidth': 0.5,
        'axes.edgecolor': INK2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 400,
    })


def window(x0, x1, y0, y1, low=0.4):
    '''Expand the shorter side of a data box to the panel aspect, so a panel
    drawn at equal aspect fills its figure with no savefig cropping. `low` is
    the share of added height that goes below the box.'''
    w, h = x1 - x0, y1 - y0
    if w / h < ASPECT:
        pad = (ASPECT * h - w) / 2
        x0, x1 = x0 - pad, x1 + pad
    else:
        pad = w / ASPECT - h
        y0, y1 = y0 - low * pad, y1 + (1 - low) * pad
    mx, my = MARGIN * (x1 - x0), MARGIN * (y1 - y0)
    return (x0 - mx, x1 + mx), (y0 - my, y1 + my)


def frame(ax, x0, x1, y0, y1):
    '''The arena box as a thin gray rule, drawn rather than left to the spine,
    because the view is padded out past the box to reach the panel aspect.'''
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec=RULE, lw=0.5, zorder=1))


def goal_2d(ax):
    '''Drone icon at the quad2d goal, side view: arms, body, rotor posts and
    edge-on props.'''
    gx, gz = Q2_GOAL
    ax.plot([gx - 0.13, gx + 0.13], [gz, gz], color=INK, lw=1.0, solid_capstyle='butt', zorder=5)
    ax.add_patch(Rectangle((gx - 0.04, gz - 0.018), 0.08, 0.036, fc=INK, ec='none', zorder=6))
    for sx in (-1, 1):
        px = gx + sx * 0.13
        ax.plot([px, px], [gz, gz + 0.03], color=INK, lw=0.8, zorder=5)
        ax.plot([px - 0.055, px + 0.055], [gz + 0.032] * 2, color=INK, lw=1.2,
                solid_capstyle='round', zorder=5)


def goal_3d(ax, zorder):
    '''Drone icon at the quad3d goal, an X frame with four rotor rings in the
    hover plane, and a dotted drop line to the floor.'''
    gx, gy, gz = Q3_GOAL
    z0 = Q3_Z[0]
    ax.plot([gx, gx], [gy, gy], [z0, gz], color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=zorder)
    ax.plot([gx], [gy], [z0], marker='o', ms=1.8, mfc=INK2, mec='none', ls='none', zorder=zorder)
    t = np.linspace(0, 2 * np.pi, 60)
    arm = 0.36 / np.sqrt(2)
    for sx, sy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        ex, ey = gx + sx * arm, gy + sy * arm
        ax.plot([gx, ex], [gy, ey], [gz, gz], color=INK, lw=1.0, zorder=zorder + 0.001)
        ax.plot(ex + 0.14 * np.cos(t), ey + 0.14 * np.sin(t), np.full_like(t, gz + 0.03),
                color=INK, lw=0.7, zorder=zorder + 0.002)


def tips_2d(rng):
    '''Arrow tip positions on quad2d: staggered rows through the band, jittered
    so the arrows read as moving air rather than a table.'''
    out = []
    for k in range(-3, 4):
        z = q2.CENTRE + 0.085 * k
        for x in np.arange(-0.58 + (0.21 if k % 2 else 0.0), 0.99, 0.42):
            out.append((x + rng.uniform(-0.04, 0.04), z + rng.uniform(-0.01, 0.01)))
    return out


def plot_2d(ax, rng, pal):
    cmap = LinearSegmentedColormap.from_list('band', pal['band'])
    zs = np.linspace(*Q2_Z, 400)
    ax.imshow(np.tile(g2(zs)[:, None], (1, 2)), extent=(*Q2_X, *Q2_Z), origin='lower',
              aspect='auto', cmap=cmap, norm=NORM, interpolation='bilinear', zorder=0)

    for x, z in tips_2d(rng):
        s = float(g2(z))
        if s >= 0.05:
            ax.add_patch(Polygon(arrow(s) + (x, z), closed=True, fc=ARROW, ec='none', zorder=3))

    goal_2d(ax)
    frame(ax, Q2_X[0], Q2_X[1], Q2_Z[0], Q2_Z[1])

    xlim, ylim = window(*Q2_X, *Q2_Z)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_axis_off()


def toward_camera():
    '''Unit vector from the scene toward the quad3d camera.'''
    e, a = np.radians(ELEV), np.radians(AZIM)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def slab_faces(xa, xb):
    '''The six faces of the slab x in [xa, xb], spanning the arena in y and z,
    keyed by outward normal.'''
    (y0, y1), (z0, z1) = Q3_XY, Q3_Z
    return {
        (-1, 0, 0): [(xa, y0, z0), (xa, y1, z0), (xa, y1, z1), (xa, y0, z1)],
        (1, 0, 0): [(xb, y0, z0), (xb, y1, z0), (xb, y1, z1), (xb, y0, z1)],
        (0, -1, 0): [(xa, y0, z0), (xb, y0, z0), (xb, y0, z1), (xa, y0, z1)],
        (0, 1, 0): [(xa, y1, z0), (xb, y1, z0), (xb, y1, z1), (xa, y1, z1)],
        (0, 0, -1): [(xa, y0, z0), (xb, y0, z0), (xb, y1, z0), (xa, y1, z0)],
        (0, 0, 1): [(xa, y0, z1), (xb, y0, z1), (xb, y1, z1), (xa, y1, z1)],
    }


def draw_curtain(ax, c, arrows, base, pal):
    '''One curtain as nested slabs with its arrows inside. Back faces go under
    the arrows and front faces over them, so arrows in the core sit in the haze.
    Everything lands in zorder [base, base + 1).'''
    cam = toward_camera()
    for t in SHELL_LEVELS:
        half = q3.WIDTH * np.sqrt(-2 * np.log(t))
        for n, quad in slab_faces(c - half, c + half).items():
            front = np.dot(n, cam) > 0
            ax.add_collection3d(Poly3DCollection([quad], facecolors=pal['shell'], edgecolors='none',
                                                 alpha=SHELL_ALPHA, zorder=base + (0.8 if front else 0.1)))

    for pts in arrows:
        ax.add_collection3d(Poly3DCollection([pts], facecolors=ARROW, edgecolors='none',
                                             zorder=base + 0.5))

    # Outline the outermost slab: edges touching a front face solid, the rest faint.
    half = q3.WIDTH * np.sqrt(-2 * np.log(SHELL_LEVELS[0]))
    faces = slab_faces(c - half, c + half)
    normals = list(faces)
    for i, n1 in enumerate(normals):
        for n2 in normals[i + 1:]:
            if np.dot(n1, n2) != 0:
                continue
            p, q = sorted(set(faces[n1]) & set(faces[n2]))
            shown = np.dot(n1, cam) > 0 or np.dot(n2, cam) > 0
            ax.plot(*zip(p, q), color=pal['edge'], lw=0.45 if shown else 0.3,
                    alpha=1.0 if shown else 0.45, zorder=base + 0.9)


def comb(c, y, z, rng):
    '''The five arrows of one comb on quad3d, at COMB_OFFSETS curtain widths
    from centre c, tips near (y, z), each lying in its plane x = const and
    pointing +y. Returns the 3-D outline of each.'''
    k = ARROW_SCALE_3D
    out = []
    for ux in COMB_OFFSETS:
        x = c + ux * q3.WIDTH
        uv = arrow(float(g3(x)))
        yy, zz = y + rng.uniform(-0.04, 0.04), z + rng.uniform(-0.02, 0.02)
        out.append(np.column_stack([np.full(len(uv), x), yy + k * uv[:, 0], zz + k * uv[:, 1]]))
    return out


def screen_box(ax, pts, pad=0.0):
    '''Bounding box of 3-D points in the axes' projected coordinates.'''
    u, v, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax.get_proj())
    return np.array([u.min() - pad, u.max() + pad, v.min() - pad, v.max() + pad])


def arena_corners():
    return np.array([(a, b, c) for a in Q3_XY for b in Q3_XY for c in Q3_Z])


def place_combs(ax, rng, n_combs):
    '''Pick combs from a lattice of candidates by farthest-point sampling on
    screen: each pick is the candidate farthest from everything kept so far
    whose projected bounding box overlaps none of them. The drone icon's
    footprint is kept from the start, so no comb covers it. Evenly spread for
    any view angle.'''
    def centre(b):
        return np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2])

    arena = screen_box(ax, arena_corners())
    pad = 0.01 * (arena[1] - arena[0])

    cands = []
    for c in (-q3.X_C, q3.X_C):
        for y in np.linspace(-0.85, 1.72, 8):
            for z in np.linspace(0.5, 2.75, 10):
                arrows = comb(c, y, z, rng)
                cands.append((screen_box(ax, np.vstack(arrows), pad), arrows))

    gx, gy, gz = Q3_GOAL
    drone = np.array([(gx + a, gy + b, gz + c) for a in (-0.45, 0.45) for b in (-0.45, 0.45)
                      for c in (-0.1, 0.15)])
    kept = [screen_box(ax, drone, pad)]
    picks = []
    while len(picks) < n_combs:
        best, best_d = None, -1.0
        for i, (b, _) in enumerate(cands):
            if any(b[0] < o[1] and o[0] < b[1] and b[2] < o[3] and o[2] < b[3] for o in kept):
                continue
            d = min(np.linalg.norm(centre(b) - centre(o)) for o in kept)
            if d > best_d:
                best, best_d = i, d
        if best is None:
            break
        kept.append(cands[best][0])
        picks.append(cands.pop(best)[1])
    return picks


def plot_3d(ax, rng, pal):
    x0, x1 = Q3_XY
    z0, z1 = Q3_Z
    # View first: comb placement reads the projection.
    ax.set_proj_type(PROJ)
    ax.set_xlim(x0, x1)
    ax.set_ylim(x0, x1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, x1 - x0, z1 - z0), zoom=ZOOM_3D)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()

    edges = [((x0, x0), (x1, x0)), ((x1, x0), (x1, x1)), ((x1, x1), (x0, x1)), ((x0, x1), (x0, x0))]
    for zz in (z0, z1):
        for (a, b), (c, d) in edges:
            ax.plot([a, c], [b, d], [zz, zz], color=RULE, lw=0.5, zorder=1)
    for a in (x0, x1):
        for b in (x0, x1):
            ax.plot([a, a], [b, b], [z0, z1], color=RULE, lw=0.5, zorder=1)

    # Painter's order, since computed_zorder is off: the farther curtain first.
    # Where the two overlap on screen the nearer one is in front.
    combs = place_combs(ax, rng, N_COMBS_3D)
    cam = toward_camera()
    centres = sorted((-q3.X_C, q3.X_C), key=lambda c: np.dot((c, 0.0, 0.0), cam))
    for i, c in enumerate(centres):
        mine = [pts for arrows in combs for pts in arrows if np.sign(pts[0, 0]) == np.sign(c)]
        draw_curtain(ax, c, mine, 2 + i, pal)

    # The goal sits in the gap: behind the nearer curtain, in front of the farther
    # one. From any view that shows +y arrows side-on, the nearer curtain's front
    # corner covers the goal on screen, so drawing it in depth order puts that
    # curtain's haze over it, which is what places it in the gap for the eye.
    goal_3d(ax, zorder=2.95)


def cart(ax, x, theta, ghost=False):
    '''Cart, wheels and pole at cart position x and pole angle theta, theta 0
    upright and positive leaning +x. Ghost draws the goal pose as a dashed
    outline.'''
    col = GHOST if ghost else INK
    ls = (0, (2.0, 1.4)) if ghost else '-'
    lw = 0.8 if ghost else 1.1
    y0 = CP_WHEEL * 1.2
    ax.add_patch(Rectangle((x - CP_CART_W / 2, y0), CP_CART_W, CP_CART_H, fc='white', ec=col,
                           lw=lw, ls=ls, zorder=4))
    for dx in (-CP_CART_W / 3, CP_CART_W / 3):
        ax.add_patch(Circle((x + dx, CP_WHEEL), CP_WHEEL, fc='white', ec=col, lw=lw * 0.85,
                            zorder=5))
    py = y0 + CP_CART_H
    ax.plot([x, x + CP_POLE * np.sin(theta)], [py, py + CP_POLE * np.cos(theta)], color=col,
            lw=1.6 if not ghost else 1.1, ls=ls, solid_capstyle='round', zorder=6)
    ax.add_patch(Circle((x, py), 0.028, fc=col, ec='none', zorder=7))


def plot_cp(ax, rng, pal):
    '''No field: this family has no noise. The rail and its end stops are the
    arena, the dashed cart is the goal, the solid one is a start.'''
    del rng, pal
    ax.plot([-CP_RAIL, CP_RAIL], [0, 0], color=RULE, lw=0.7, zorder=1,
            solid_capstyle='butt')
    for sx in (-1, 1):
        ax.plot([sx * CP_RAIL] * 2, [-0.10, 0.17], color=RULE, lw=0.7, zorder=1)

    cart(ax, 0.0, 0.0, ghost=True)
    cart(ax, *CP_START)

    # Headroom past the upright pole, so the tip does not sit on the panel edge.
    top = CP_WHEEL * 1.2 + CP_CART_H + CP_POLE + 0.07
    xlim, ylim = window(-CP_RAIL, CP_RAIL, -0.13, top, low=0.35)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_axis_off()


# Fixed seeds: the jitter is layout, and must not change between builds of the
# paper. One per panel so adding a panel cannot move another one's arrows.
PANELS = (('2d', plot_2d, 3, False), ('3d', plot_3d, 5, True), ('cp', plot_cp, 7, False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='system_panels',
                    help='path stem; writes <out>_{2d,3d,cp}, each as .pdf and .png')
    ap.add_argument('--field', default='blue', choices=sorted(FIELDS),
                    help='envelope and slab colour; arrows and icons stay black')
    ap.add_argument('--width', type=float, default=PANEL_W,
                    help='panel width in inches, height follows the fixed aspect')
    args = ap.parse_args(argv)
    style()
    pal = FIELDS[args.field]
    size = (args.width, args.width / ASPECT)

    for tag, fn, seed, is_3d in PANELS:
        fig = plt.figure(figsize=size)
        kw = dict(projection='3d', computed_zorder=False) if is_3d else {}
        ax = fig.add_axes([0, 0, 1, 1], **kw)
        fn(ax, np.random.default_rng(seed), pal)
        for ext in ('pdf', 'png'):
            fig.savefig(f'{args.out}_{tag}.{ext}')
        plt.close(fig)

    w, h = size
    print(f'{args.out}_{{2d,3d,cp}}.{{pdf,png}}: written, {w:.2f} x {h:.2f} in, field {args.field}')


if __name__ == '__main__':
    main()
