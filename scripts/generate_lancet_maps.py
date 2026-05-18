"""
scripts/generate_lancet_maps.py
Regenerate the four publication-quality choropleth maps in lancet/
using the latest DGHS divisional data.

Run from project root:
    python3 scripts/generate_lancet_maps.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHP_ADM1  = os.path.join(
    ROOT,
    'data/bgd_adm_bbs_20201113_shp/bgd_adm_bbs_20201113_SHP',
    'bgd_admbnda_adm1_bbs_20201113.shp',
)
LANCET_DIR = os.path.join(ROOT, 'lancet')
DPI = 300

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "DENV", ["cyan", "blue", "green", "red"]
)

# ── Data: DGHS bulletin 18 May 2026, cumulative since 15 March 2026 ──
SUSPECTED = {
    'Dhaka': 26821, 'Rajshahi': 5159, 'Chattogram': 8875,
    'Barisal': 4583, 'Sylhet': 2758, 'Mymensingh': 1319,
    'Khulna': 4239, 'Rangpur': 1157,
}
SUSP_DEATHS = {
    'Dhaka': 155, 'Rajshahi': 78, 'Chattogram': 36,
    'Barisal': 29, 'Sylhet': 35, 'Mymensingh': 33,
    'Khulna': 19, 'Rangpur': 4,
}
VACC_COV = {
    'Dhaka': 102, 'Chattogram': 102, 'Rajshahi': 103,
    'Khulna': 100, 'Rangpur': 103, 'Mymensingh': 101,
    'Sylhet': 98, 'Barisal': 100,
}
DIV_POP_2022 = {
    'Dhaka': 44_310_000, 'Chattogram': 33_580_000,
    'Rajshahi': 20_340_000, 'Khulna': 17_280_000,
    'Rangpur': 17_900_000, 'Mymensingh': 13_360_000,
    'Sylhet': 11_370_000, 'Barisal': 9_380_000,
}

# ── Shapefile ─────────────────────────────────────────────────────
shp = gpd.read_file(SHP_ADM1)
shp['Division'] = shp['ADM1_EN'].replace({'Chittagong': 'Chattogram'})
shp['centroid'] = shp.geometry.centroid


# ── Map helpers ───────────────────────────────────────────────────
def add_compass_rose(ax, size_inch=0.82, loc='upper right'):
    cax = inset_axes(ax, width=size_inch, height=size_inch, loc=loc,
                     bbox_to_anchor=(0.97, 0.97),
                     bbox_transform=ax.transAxes, borderpad=0)
    cax.set_xlim(-1.4, 1.4)
    cax.set_ylim(-1.4, 1.4)
    cax.set_aspect('equal')
    cax.axis('off')
    cax.add_patch(mpatches.Circle((0, 0), 1.15, fc='white', ec='#333333',
                                  lw=1.1, zorder=1))
    card = [
        [(0, 1.0), (-0.22, 0.12), (0, 0), (0.22, 0.12)],
        [(0, -1.0), (-0.22, -0.12), (0, 0), (0.22, -0.12)],
        [(1.0, 0), (0.12, 0.22), (0, 0), (0.12, -0.22)],
        [(-1.0, 0), (-0.12, 0.22), (0, 0), (-0.12, -0.22)],
    ]
    fills = ['#111111', 'white', '#111111', 'white']
    for pts, fc in zip(card, fills):
        cax.add_patch(mpatches.Polygon(pts, closed=True, fc=fc,
                                       ec='#333333', lw=0.7, zorder=3))
    for dx, dy in [(0.62, 0.62), (-0.62, 0.62), (0.62, -0.62), (-0.62, -0.62)]:
        ang = np.arctan2(dy, dx)
        perp = ang + np.pi / 2
        tip = (dx, dy)
        bl = (0.14 * np.cos(perp), 0.14 * np.sin(perp))
        br = (-0.14 * np.cos(perp), -0.14 * np.sin(perp))
        cax.add_patch(mpatches.Polygon([tip, bl, (0, 0), br], closed=True,
                                       fc='#888888', ec='#333333', lw=0.5, zorder=2))
    cax.add_patch(mpatches.Circle((0, 0), 0.11, fc='white', ec='#333333',
                                  lw=0.9, zorder=4))
    cax.text(0,  1.22, 'N', ha='center', va='bottom', fontsize=9,
             fontweight='bold', color='#111111', zorder=5)
    cax.text(0, -1.22, 'S', ha='center', va='top',    fontsize=7,
             color='#333333', zorder=5)
    cax.text( 1.22, 0, 'E', ha='left',   va='center', fontsize=7,
             color='#333333', zorder=5)
    cax.text(-1.22, 0, 'W', ha='right',  va='center', fontsize=7,
             color='#333333', zorder=5)


def add_scale_bar(ax, gdf, bar_km=200, n_seg=4, x0_frac=0.05, y0_frac=0.05,
                  fig_width_in=7.0):
    bounds = gdf.total_bounds
    mw = bounds[2] - bounds[0]
    mh = bounds[3] - bounds[1]
    x0 = bounds[0] + x0_frac * mw
    yb = bounds[1] + y0_frac * mh
    km_per_deg = 111.32 * np.cos(np.radians(23.5))
    total_deg = bar_km / km_per_deg
    seg_deg = total_deg / n_seg
    seg_km = bar_km / n_seg
    bh = mh * 0.013
    for i in range(n_seg):
        fc = '#111111' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * seg_deg, yb - bh / 2), seg_deg, bh,
            fc=fc, ec='#111111', lw=0.8, zorder=6,
        ))
        label = '0' if i == 0 else f'{int(i * seg_km)}'
        ax.text(x0 + i * seg_deg, yb - bh / 2 - mh * 0.012, label,
                ha='center', va='top', fontsize=7.5, color='#111111', zorder=7)
    ax.text(x0 + total_deg, yb - bh / 2 - mh * 0.012, f'{bar_km} km',
            ha='center', va='top', fontsize=7.5, color='#111111', zorder=7)
    # Scale denominator: bar_km / bar length on paper
    bar_frac = total_deg / mw          # fraction of map width the bar covers
    bar_paper_m = bar_frac * fig_width_in * 0.0254 * 0.82  # ~82% usable axis
    scale_denom = int(round(bar_km * 1000 / bar_paper_m))
    ax.text(x0, yb + bh, f'Scale 1:{scale_denom:,}',
            ha='left', va='bottom', fontsize=7, color='#111111', zorder=7)


def label_divisions(ax, gdf, val_col, fmt, text_color='white'):
    for _, row in gdf.iterrows():
        cx, cy = row['centroid'].x, row['centroid'].y
        ax.text(cx, cy + 0.06, row['Division'],
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=text_color,
                path_effects=[pe.withStroke(linewidth=2, foreground='#00000055')])
        ax.text(cx, cy - 0.08, fmt(row[val_col]),
                ha='center', va='top', fontsize=8.5,
                color=text_color,
                path_effects=[pe.withStroke(linewidth=2, foreground='#00000055')])


def make_map(gdf, col, title, fmt, norm, out_name):
    fig, ax = plt.subplots(figsize=(7, 9))
    gdf.plot(column=col, cmap=CMAP, norm=norm,
             edgecolor='#444444', linewidth=0.8, ax=ax, alpha=0.95)
    label_divisions(ax, gdf, col, fmt)
    add_scale_bar(ax, gdf)
    add_compass_rose(ax)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.04, pad=0.02, shrink=0.85, aspect=30)
    cbar.ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    out = os.path.join(LANCET_DIR, out_name)
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out}')


# ── Build GeoDataFrames ───────────────────────────────────────────
import pandas as pd

df_main = pd.DataFrame({
    'Division':    list(SUSPECTED.keys()),
    'Suspected':   list(SUSPECTED.values()),
    'Susp_Deaths': list(SUSP_DEATHS.values()),
    'Population':  [DIV_POP_2022[d] for d in SUSPECTED],
    'Vacc_Pct':    [VACC_COV[d] for d in SUSPECTED],
})
df_main['CFR_pct']        = df_main['Susp_Deaths'] / df_main['Suspected'] * 100
df_main['Incidence_100k'] = df_main['Suspected'] / df_main['Population'] * 100_000

gdf = shp.merge(df_main, on='Division', how='left')

# ── Generate maps ─────────────────────────────────────────────────
make_map(
    gdf, 'Suspected',
    title='Suspected Cases by Division — Bangladesh',
    fmt=lambda v: f'{int(v):,}',
    norm=mcolors.Normalize(vmin=0, vmax=gdf['Suspected'].max()),
    out_name='fig01_2026_division_cases_map.png',
)

make_map(
    gdf, 'CFR_pct',
    title='Case Fatality Rate (%) by Division — Bangladesh',
    fmt=lambda v: f'{v:.2f}%',
    norm=mcolors.Normalize(vmin=0, vmax=gdf['CFR_pct'].max()),
    out_name='fig02_2026_division_cfr_map.png',
)

make_map(
    gdf, 'Incidence_100k',
    title='Incidence Rate per 100,000 by Division — Bangladesh',
    fmt=lambda v: f'{v:.1f}',
    norm=mcolors.Normalize(vmin=0, vmax=gdf['Incidence_100k'].max()),
    out_name='fig03_2026_division_incidence_map.png',
)

make_map(
    gdf, 'Vacc_Pct',
    title='Vaccination Coverage by Division — Bangladesh',
    fmt=lambda v: f'{int(v)}%',
    norm=mcolors.Normalize(vmin=gdf['Vacc_Pct'].min() - 1,
                           vmax=gdf['Vacc_Pct'].max() + 1),
    out_name='fig04_2026_division_vaccination_map.png',
)

print('\nAll lancet maps regenerated.')
