"""One PDF with a drawing of every NFA found by ceiling_hunt.py (6-state NFAs whose minimized DFA has all 2^6 = 64 states).

Reads data/experiment_ceiling_hunt_6_states/hits_nfas.json and writes ceiling_nfas.pdf next to it.
"""
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from core.automata import EPSILON, minimize_dfa, nfa_from_dict, nfa_to_dfa
from experiments.ceiling_hunt import OUTPUT_DIR
from visualization.visualize_dfa import render_dfa
from visualization.visualize_nfa import render_nfa

PAGE = (11, 8.5)


def use_georgia():
    for name in ["Georgia.ttf", "Georgia Bold.ttf", "Georgia Italic.ttf"]:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    plt.rcParams["font.family"] = ["Georgia", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42


def get_facts(nfa):
    pairs = [(state, symbol, sorted(destinations))
             for state, symbol_dict in sorted(nfa.transitions.items())
             for symbol, destinations in sorted(symbol_dict.items())
             if symbol != EPSILON and len(destinations) > 1]
    edges = sum(len(d) for sd in nfa.transitions.values() for d in sd.values())
    return pairs, edges


def format_transitions(nfa):
    # One line per state: "q0: 0 → q1   1 → q5"
    rows = []

    for state in sorted(nfa.states):
        parts = []
        for symbol, destinations in sorted(nfa.transitions.get(state, {}).items()):
            targets = ",".join(f"q{t}" for t in sorted(destinations))
            parts.append(f"{symbol} → {targets}")

        rows.append(f"q{state}:  " + ("     ".join(parts) if parts else "(none)"))

    return rows


def title_page(pdf, hits, summary):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.90, "6-state NFAs that reach the 2$^n$ ceiling", fontsize=26, fontweight="bold")
    fig.text(0.06, 0.855, "Ten distinct random NFAs whose minimized DFA has all 2$^6$ = 64 states", fontsize=14, color="#555555")

    lines = [
        f"Found by ceiling_hunt.py: {summary['new_distinct_nfas_scanned']:,} new distinct random NFAs scanned, "
        f"{summary['new_hits']} hits (about {summary['hit_rate_per_million']:.1f} per million),",
        "plus 1 head start from the earlier 1M run.",
        "A hit has 64 states after subset construction and still 64 after Moore minimization, the most a 6-state NFA can give.",
        "NFAs that differ only by swapping the symbols 0 and 1 count as the same NFA.",
        "",
        "How to read the drawings:  double circle = accepting state;  red edge = one of several choices for the same (state, symbol);",
        "plain edges are deterministic. None of these NFAs uses ε edges. Each NFA page is followed by the page for its 64-state DFA",
        "(letters A, B, C ... name the DFA states in breadth-first order from the start state A; double circles accept).",
    ]
    for index, text in enumerate(lines):
        fig.text(0.06, 0.79 - index * 0.036, text, fontsize=11.5)

    header = ["#", "NFA id", "accepting", "choices", "symbol edges", "found"]
    rows = []
    for index, hit in enumerate(hits, 1):
        nfa = nfa_from_dict(hit["nfa"])
        pairs, edges = get_facts(nfa)
        rows.append([str(index), hit["id"][:12], ", ".join(f"q{s}" for s in sorted(nfa.accept_states)), str(len(pairs)), str(edges),
                     "head start" if hit["found_in"].startswith("earlier") else "this run"])

    axis = fig.add_axes([0.06, 0.07, 0.88, 0.38])
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="left",
                       colWidths=[0.05, 0.22, 0.2, 0.12, 0.15, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)

    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#A4C2F4")
        if row == 0:
            cell.set_facecolor("#C9DAF8")
            cell.set_text_props(fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def nfa_page(pdf, index, hit, image_path):
    nfa = nfa_from_dict(hit["nfa"])
    pairs, edges = get_facts(nfa)
    sub = nfa_to_dfa(nfa)
    minimized = minimize_dfa(sub)
    assert len(sub.states) == len(minimized.states) == 64

    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.93, f"NFA {index} of {len(HITS_TOTAL)}", fontsize=22, fontweight="bold")
    fig.text(0.05, 0.895, f"{hit['id'][:16]}   ({hit['found_in']})", fontsize=11, color="#555555")

    axis = fig.add_axes([0.04, 0.34, 0.92, 0.52])
    axis.axis("off")
    axis.imshow(mpimg.imread(image_path), aspect="equal")

    choices = "; ".join(f"q{s} on {a} → {', '.join(f'q{t}' for t in d)}" for s, a, d in pairs)
    stats = [
        f"Subset-construction DFA: {len(sub.states)} states     Minimized DFA: {len(minimized.states)} states     Ceiling 2$^6$ = 64   (100%)",
        f"Accepting states: {', '.join(f'q{s}' for s in sorted(nfa.accept_states))}     Symbol edges: {edges}     "
        f"Non-deterministic choices: {len(pairs)}",
        f"Choices:  {choices}",
    ]
    for line_index, text in enumerate(stats):
        fig.text(0.05, 0.285 - line_index * 0.035, text, fontsize=11.5)

    fig.text(0.05, 0.165, "Transitions", fontsize=12, fontweight="bold")
    for line_index, row in enumerate(format_transitions(nfa)):
        column, line = divmod(line_index, 3)
        fig.text(0.05 + column * 0.34, 0.135 - line * 0.035, row, fontsize=11, family="DejaVu Sans Mono")

    pdf.savefig(fig)
    plt.close(fig)


def dfa_page(pdf, index, hit, image_path, dfa, minimized):
    # The subset-construction DFA; it is already minimal, so the minimized DFA is the same machine
    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.93, f"DFA for NFA {index} of {len(HITS_TOTAL)}", fontsize=22, fontweight="bold")
    accepting = len(dfa.accept_states)
    fig.text(0.05, 0.895, f"{hit['id'][:16]}   subset construction: {len(dfa.states)} states ({accepting} accepting)   "
                          f"minimized: {len(minimized.states)} states, so no two states merge", fontsize=11, color="#555555")

    axis = fig.add_axes([0.02, 0.03, 0.96, 0.84])
    axis.axis("off")
    axis.imshow(mpimg.imread(image_path), aspect="equal")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def main():
    global HITS_TOTAL
    hits = json.loads((OUTPUT_DIR / "hits_nfas.json").read_text())
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    table = {row["nfa_key"]: row["found_in"] for row in __import__("csv").DictReader(open(OUTPUT_DIR / "hits.csv"))}

    for hit in hits:
        hit["nfa"] = {key: value for key, value in hit.items() if key != "id"}
        hit["found_in"] = table[hit["id"]]

    HITS_TOTAL = hits
    use_georgia()
    output = OUTPUT_DIR / "ceiling_nfas.pdf"

    with tempfile.TemporaryDirectory() as directory, PdfPages(output) as pdf:
        title_page(pdf, hits, summary)

        for index, hit in enumerate(hits, 1):
            image_path = Path(directory) / f"nfa_{index}.png"
            render_nfa(nfa_from_dict(hit["nfa"]), image_path, dpi=170)
            nfa_page(pdf, index, hit, image_path)

            nfa = nfa_from_dict(hit["nfa"])
            dfa = nfa_to_dfa(nfa)
            dfa_path = Path(directory) / f"dfa_{index}.png"
            # 64 states: letter names, no sets, force-directed layout (dot is too slow and too wide at this size)
            render_dfa(dfa, dfa_path, names="letters", layout="neato", dpi=150)
            dfa_page(pdf, index, hit, dfa_path, dfa, minimize_dfa(dfa))

    print(f"Wrote {output} ({2 * len(hits) + 1} pages)")


if __name__ == "__main__":
    main()
