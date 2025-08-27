import os, re, json, numpy as np
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
# Setup LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

RUN_DIR   = "./"
BASE_DIR  = os.path.join(RUN_DIR, "checkpoint0")
CHECK_52_DIR = os.path.join(RUN_DIR, "checkpoint52")
CHECK_104_DIR = os.path.join(RUN_DIR, "checkpoint104")
CHECK_153_DIR = os.path.join(RUN_DIR, "checkpoint153")

# The cluster you gave (Language Identification)
cluster = [
    {"tool":"What's Language","api_name":"LanguageDetection"},
    {"tool":"Quick Language Detector","api_name":"Detect Language"},
    {"tool":"Text Language by API-Ninjas","api_name":"/v1/textlanguage"},
    {"tool":"Translate_v3","api_name":"Fast Language Detection"},
    {"tool":"Translate All Languages","api_name":"Detect"},
]

OUT_PDF = "selection_rates_base_vs_adapted.pdf"
OUT_PNG = "selection_rates_base_vs_adapted.png"
# ------------------------------------------

# LaTeX special chars:  # $ % & ~ _ ^ \ { }
TEX_ESCAPES = {
    '&':  r'\&',
    '%':  r'\%',
    '$':  r'\$',
    '#':  r'\#',
    '_':  r'\_',
    '{':  r'\{',
    '}':  r'\}',
    '~':  r'\textasciitilde{}',
    '^':  r'\^{}',
    '\\': r'\textbackslash{}',
}

def escape_tex(s):
    return ''.join(TEX_ESCAPES.get(ch, ch) for ch in s)

def standardize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).lower()
    s = re.sub(r"_+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = "get_" + s
    return s

# Map each endpoint to the function name form: {api}_for_{tool}
tool_labels = [escape_tex(c["tool"]) for c in cluster]
fn_names = [f"{standardize(c['api_name'])}_for_{standardize(c['tool'])}" for c in cluster]
fn_tools = [f"{standardize(c['tool'])}" for c in cluster]

fn_total = fn_names + fn_tools

# For parsing: find first allowed tool AFTER the first "Action"
allowed_re = re.compile("|".join(re.escape(n) for n in fn_total))
def parse_action(text):
    m = re.search(r"(^|\n)\s*Action\s*:?", text, flags=re.IGNORECASE)
    start_idx = m.end() if m else 0
    tail = text[start_idx:]
    m2 = allowed_re.search(tail)
    return m2.group(0) if m2 else None

def tally_folder(folder):
    counts = {fn: 0 for fn in fn_names}
    files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    for f in files:
        with open(os.path.join(folder, f), "r", encoding="utf-8") as fh:
            txt = fh.read()
        tool = parse_action(txt)
        for fn in counts.keys():
            if tool and tool in fn:
                counts[fn] += 1
    total = sum(counts.values())
    rates = {fn: (counts[fn] / total if total else 0.0) for fn in fn_names}
    return counts, rates, total

counts_base, rates_base, total_base   = tally_folder(BASE_DIR)
counts_52,   rates_52,   total_52     = tally_folder(CHECK_52_DIR)
counts_104,  rates_104,  total_104    = tally_folder(CHECK_104_DIR)
counts_153,  rates_153,  total_153    = tally_folder(CHECK_153_DIR)

print("Base total counted (recognized tools):    ", total_base)
print("Step 52 total counted (recognized tools):", total_52)
print("Step 104 total counted (recognized tools):", total_104)
print("Step 153 total counted (recognized tools):", total_153)
for lbl, fn in zip(tool_labels, fn_names):
    print(f"{lbl:32s} | base {rates_base[fn]:.3f} | s52 {rates_52[fn]:.3f} | s104 {rates_104[fn]:.3f} | s153 {rates_153[fn]:.3f}")

# --- plotting (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
x = np.arange(len(tool_labels))

def plot_panel(ax, rates, title):
    vals = [rates[fn] for fn in fn_names]
    bars = ax.bar(x, vals, color="#cfcfcf")
    ax.set_xticks(x)
    ax.set_xticklabels(tool_labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(r'\textbf{' + title + '}', fontsize=14)
    for y in [0.2, 0.4, 0.6, 0.8]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, zorder=0)

    ninjas_idx = tool_labels.index("Text Language by API-Ninjas")
    b = bars[ninjas_idx]
    b.set_color("#1f77b4")
    b.set_edgecolor("black")
    b.set_linewidth(2.0)
    b.set_hatch("//")
    b.set_zorder(3)

    yv = vals[ninjas_idx]
    ax.annotate(f"{yv:.3f}",
                xy=(ninjas_idx, yv),
                xytext=(ninjas_idx, min(1.0, yv + 0.12)),
                ha="center", va="bottom",
                fontsize=12,
                arrowprops=dict(arrowstyle="->", lw=1.2))

plot_panel(axes[0,0], rates_base, "Base Model")
plot_panel(axes[0,1], rates_52,   "CPT Model (1/3 epoch)")
plot_panel(axes[1,0], rates_104,  "CPT Model (2/3 epoch)")
plot_panel(axes[1,1], rates_153,  "CPT Model (1 epoch)")

axes[0,0].set_ylabel("Selection Rate", fontsize=13)
axes[1,0].set_ylabel("Selection Rate", fontsize=13)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
plt.show()