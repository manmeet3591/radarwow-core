# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 00 · Free radar data for S, C and X band, and what is wrong with each
#
# **radar-wow core.** Every algorithm in the radar-wow repos runs on the same free data:
#
# | name | band | radar | source |
# |---|---|---|---|
# | `nexrad_ktlx_s` | S | NWS WSR-88D KTLX, 20 May 2013 (Moore tornado) | NEXRAD Level II |
# | `arm_csapr2_c` | C | ARM CSAPR2, Argentina, 11 Nov 2018 | ARM CfRadial |
# | `imd_jaipur_c` | C | IMD Jaipur DWR, 22 Aug 2022 | IMD IRIS netCDF |
# | `arm_xprecip_x` | X | ARM X-band precip radar, Colorado, 14 Mar 2022 | ARM CfRadial |
# | `arm_xsapr_x` | X | ARM X-SAPR, Oklahoma, 20 May 2012 | CfRadial (Py-ART example) |
#
# All files come from the openradar `open-radar-data` collection; `nexrad_aws()` fetches any other
# NEXRAD volume from AWS. **Each radar arrives with its own problems**, so before any algorithm we
# run `diagnose()`, which measures them and names the radar-wow stage that fixes each one.

# %%
# Colab: uncomment to install
# !pip install -q "git+https://github.com/manmeet3591/radarwow-core"

# %%
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from radarwow.core import SAMPLES, diagnose, load_sample
from radarwow.core.noise import signal_mask
from radarwow.core.plotting import panel

FIG = "../docs/figures"

# %% [markdown]
# ## 1. Known problems per source (from our checks)

# %%
for s in SAMPLES.values():
    display(Markdown(f"**{s.name}** · band {s.band} · {s.radar} · {s.when}\n\n" + "\n".join(f"- {k}" for k in s.known_issues)))

# %% [markdown]
# ## 2. Look at each radar: raw moments, noise masked
# Gates that are only receiver noise are hidden using `signal_mask` (SNR and NCP if the file has
# them, otherwise the reflectivity noise floor). Note how different the five look.

# %%
volumes = {name: load_sample(name) for name in SAMPLES}
reports = {}
for name, dt in volumes.items():
    rep = diagnose(dt)
    reports[name] = rep
    ds = dt[rep.sweep].to_dataset()
    sig = signal_mask(ds)
    shown = ds.copy()
    for m in ("DBZH", "ZDR", "PHIDP", "RHOHV"):
        if m in shown:
            shown[m] = shown[m].where(sig)
    span = float(ds["range"].values[-1]) / 1e3
    fig = panel(shown, [m for m in ("DBZH", "ZDR", "PHIDP", "RHOHV") if m in shown] or ["DBZH"],
                max_km=min(span, 150), suptitle=f"{name} (band {SAMPLES[name].band}), {rep.sweep}", size=3.6)
    fig.savefig(f"{FIG}/00_{name}.png", dpi=110)
    plt.show()

# %% [markdown]
# ## 3. Health checks: the problems, measured
# Severity: **blocker** (an algorithm family cannot run), **fix** (must be corrected first),
# **check** (inspect), **info**.

# %%
for name, rep in reports.items():
    display(Markdown(rep.to_markdown()))

# %% [markdown]
# ## 4. Summary table

# %%
cols = ["frac_signal", "phidp_system_offset", "rays_with_folding", "zdr_median_light_rain", "rhohv_gt_1.02", "interference_rays"]
rows = ["| sample | band | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
for name, rep in reports.items():
    rows.append(f"| {name} | {SAMPLES[name].band} | " + " | ".join(str(rep.numbers.get(c, "-")) for c in cols) + " |")
display(Markdown("\n".join(rows)))

# %% [markdown]
# ## What this tells us
# - **No two sources share the same problems.** NEXRAD stores "below threshold" as a number;
#   ARM files store noise and ship corrected copies next to the raw fields; the CSAPR2 SNR field
#   is biased; PHIDP offsets range from 0 to ~200 deg and some sources wrap at 360 deg; one ZDR
#   field is clipped at 0; the IMD file has no PHIDP or rhoHV at all.
# - So every radar-wow pipeline starts with `open_volume` (format + names + codes) and
#   `diagnose` (measured problems), and each algorithm declares which moments and bands it needs.
