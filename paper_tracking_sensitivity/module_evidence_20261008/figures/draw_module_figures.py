"""Draw the five implementation-grounded joint residual model schematics.

Run with the existing ClearSAR Python. Outputs stay beside this script.
This script draws diagrams only; it does not import or execute model code.
"""

from pathlib import Path
import json

import matplotlib as mpl
mpl.use("Agg")
# Academic Figure Skill typography baseline.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"
# Academic Figure Skill export baseline.
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
import fitz
from PIL import Image


OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN, PURPLE = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[4]
INK, MUTED, RULE = BLACK, "#565656", "#D3D3D3"
MM = 1 / 25.4


class Diagram:
    """Millimetre layout with explicit text bounds for dense scientific nodes."""

    def __init__(self, slug, number, title, subtitle, height=243):
        self.slug, self.height = slug, height
        self.fig = plt.figure(figsize=(183 * MM, height * MM), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1], xlim=(0, 183), ylim=(height, 0))
        self.ax.set_axis_off()
        self.texts, self.nodes = [], []
        self.label(6, 6, f"{number}  {title}", size=12.5, weight="bold")
        self.label(6, 15, subtitle, size=8.4, color=MUTED)
        self.ax.plot([6, 177], [23, 23], color=RULE, lw=0.7)

    def label(self, x, y, s, size=8.2, color=INK, weight="normal", ha="left", bounds=None):
        t = self.ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                         ha=ha, va="top", linespacing=1.32, zorder=5)
        self.texts.append((t, bounds))
        return t

    def box(self, x, y, w, h, title, body, color=BLUE, size=8.2):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                                   edgecolor=color, linewidth=1.0, zorder=2))
        self.ax.plot([x + 0.3, x + 0.3], [y + 0.3, y + h - 0.3], lw=2.3, color=color, zorder=3)
        bounds = (x + 2, y + 1.5, x + w - 2, y + h - 1.5)
        self.label(x + 3, y + 3, title, 9.0, color, "bold", bounds=bounds)
        if body:
            self.label(x + 3, y + 10, body, size, bounds=bounds)
        self.nodes.append((x, y, w, h))

    def arrow(self, points, color=MUTED, dashed=False, width=1.0):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        patch = FancyArrowPatch(path=MplPath(points, codes), arrowstyle="-|>",
                                mutation_scale=8, lw=width, color=color,
                                linestyle=(0, (3, 2)) if dashed else "-", zorder=1)
        self.ax.add_patch(patch)

    def section(self, y, text, color=INK):
        self.label(6, y, text, 9.2, color, "bold")

    def note(self, y, text, color=MUTED, size=8.0):
        self.label(6, y, text, size, color)

    def validate_layout(self):
        # Detect clipped node labels and text collisions; a failure needs a layout edit.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        extents = []
        issues = []
        for text, bounds in self.texts:
            extent = text.get_window_extent(renderer)
            extents.append((text, extent))
            data = extent.transformed(self.ax.transData.inverted())
            x0, x1 = sorted((data.x0, data.x1))
            y0, y1 = sorted((data.y0, data.y1))
            limit = bounds or (2, 1, 181, self.height - 1)
            if x0 < limit[0] or y0 < limit[1] or x1 > limit[2] or y1 > limit[3]:
                issues.append(f"Text outside bounds: {text.get_text()!r}; {(x0, y0, x1, y1)} vs {limit}")
        for i, (a, ea) in enumerate(extents):
            for b, eb in extents[i + 1:]:
                if ea.overlaps(eb):
                    issues.append(f"Overlapping text: {a.get_text()!r} / {b.get_text()!r}")
        if issues:
            raise ValueError(self.slug + "\n" + "\n".join(issues))
        return {"text_elements": len(self.texts), "node_count": len(self.nodes),
                "text_out_of_bounds": 0, "text_collisions": 0,
                "minimum_font_pt": min(t.get_fontsize() for t, _ in self.texts)}


def overall():
    d = Diagram("01_overall_pipeline", "1", "Joint residual model: three operating phases",
                "M1: source-anchored adaptation   M2: expert geometry   M3: alignment   M4: handoff", height=247)
    d.section(29, "A  Offline source fitting and representation training", BLUE)
    d.box(6, 38, 50, 34, "Source SAR cohorts",
          "Images + class labels\nPort-based source folds\nIdentity-purged fitting\nNo AIS / facility inputs")
    d.box(66, 38, 50, 34, "Source knowledge",
          "PCA + normalization\nRidge-1 base/expert heads\nSource OOF LR gate (M4)\nFixed parameter artifacts")
    d.box(126, 38, 51, 34, "M3 | alignment",
          "Source support 64 + query\nM2 + actual ridge-8 solve\nBalanced query CE\nUpdate only R18 layer4", RED)
    d.arrow([(56, 55), (66, 55)])
    d.arrow([(116, 55), (126, 55)])
    d.box(6, 83, 81, 28, "Paired VV/VH ResNet-50",
          "2048-D pooled features; fully frozen\nSame fixed base branch in all phases")
    d.box(97, 83, 80, 28, "VH grayscale SimCLR-pretrained R18",
          "VH repeated to 3 channels; 512-D features\nPrefix frozen; layer4 trainable only in M3", RED, size=8.0)
    d.arrow([(151, 72), (151, 83)], RED)
    d.section(122, "B  Target adaptation: exactly 64 shared labels", GREEN)
    d.box(6, 132, 44, 35, "Target support S",
          "64 labeled SAR rows\nSame S for both heads\nEncoder fixed\nNo query labels", GREEN)
    d.box(60, 132, 56, 35, "M2 | expert features",
          "Retained PCA + complement\nFixed source transformations\nBase feature map: 129-D\nJoint expert map: 2753-D", GREEN)
    d.box(126, 132, 51, 35, "M1 | residual fitting",
          "Source prior + correction\nClass-balanced targets\nTwo ridge-8 solves\nStore base/expert updates", GREEN)
    d.arrow([(50, 149), (60, 149)], GREEN)
    d.arrow([(116, 149), (126, 149)], GREEN)
    d.arrow([(46, 111), (46, 116), (87, 116), (87, 132)], BLUE)
    d.arrow([(137, 111), (137, 116), (91, 116), (91, 132)], BLUE)
    d.section(178, "C  Query inference: fixed encoders, heads and gate", PURPLE)
    d.box(6, 188, 50, 32, "Query SAR batch Q",
          "Frozen R50 + updated R18\nCompute both feature maps\nAll query rows processed", BLUE)
    d.box(66, 188, 50, 32, "Adapted class scores",
          "Base64 and joint expert64\nM1 prior + fitted residual\nBoth scores available", GREEN)
    d.box(126, 188, 51, 32, "M4 | batch handoff",
          "Existing frozen LR gate\nEligible top 5% at most\nExpert if selected; else base", PURPLE, size=8.0)
    d.arrow([(56, 204), (66, 204)])
    d.arrow([(116, 204), (126, 204)], PURPLE)
    d.arrow([(151, 167), (151, 172), (91, 172), (91, 188)], GREEN)
    d.note(228, "Source knowledge = fitted feature statistics, heads and gate; no external port knowledge.")
    d.note(234, "All expert features precede M4 selection. The 5% cap limits handoffs, not computation.")
    d.note(240, "R18 pretraining: historical unlabeled F0921 pool; deployment drops the projection head.")
    return d


def m1():
    d = Diagram("02_m1_source_prior_adaptation", "2", "M1 | Source-prior-anchored residual adaptation",
                "The same 64 target labels update the base and the M2-enabled expert.", height=239)
    d.box(6, 32, 171, 29, "Source-only fitting cohort",
          "Purge overlapping source identities; exclude the held source port for episode / OOF fits.\n"
          "Fit PCA, standardization and class-balanced source RidgeClassifier heads (alpha = 1).")
    d.box(6, 73, 80, 42, "Base branch | R50",
          "2048-D features -> PCA128 -> source z-score\n"
          "Fixed source head gives prior P_b\n"
          "Residual map: [standardized128 / sqrt(128), 1]\n"
          "129 coordinates; source-fitted statistics", BLUE, size=8.0)
    d.box(97, 73, 80, 42, "Expert branch | R50 + R18",
          "PCA128 + PCA64 -> standardized192\n"
          "Fixed ridge-1 head gives prior P_e\n"
          "M2 adds source-normalized complement\n"
          "Residual map phi_e: 2753 coordinates", BLUE, size=8.0)
    d.arrow([(46, 61), (46, 73)], BLUE)
    d.arrow([(137, 61), (137, 73)], BLUE)
    d.box(48, 127, 87, 26, "One shared support set S | n = 64",
          "T_ic = +1 for c = y_i, otherwise -1\n"
          "w_i = 64 / (C_S n_{y_i}); W = diag(w_i)", GREEN)
    d.box(6, 164, 80, 30, "Base correction | Delta_b",
          "Support residual: T - P_b,S\n"
          "Use phi_b,S; solve a 64 x 64 system\n"
          "Keep base64 fixed during M3", GREEN)
    d.box(97, 164, 80, 30, "Joint expert correction | Delta_e",
          "Support residual: T - P_e,S\n"
          "Use M2 phi_e,S; same 64 x 64 ridge solve\n"
          "Same solver in M3 and deployment", GREEN)
    d.arrow([(20, 115), (20, 164)], BLUE)
    d.arrow([(163, 115), (163, 164)], BLUE)
    d.arrow([(48, 141), (40, 141), (40, 164)], GREEN)
    d.arrow([(135, 141), (143, 141), (143, 164)], GREEN)
    d.label(23, 139, "prior + map", size=7.5, color=BLUE)
    d.label(147, 139, "prior + map", size=7.5, color=BLUE)
    d.section(204, "Actual support-balanced residual ridge | alpha = 8", GREEN)
    d.label(7, 214,
            r"$A=W^{1/2}\Phi_S,\quad \Delta=A^\top(AA^\top+8I_{64})^{-1}W^{1/2}(T-P_S)$",
            size=9.0)
    d.label(7, 226, r"$s_Q=P_Q+\Phi_Q\Delta$", size=9.0, color=GREEN)
    d.label(62, 227, "One penalized bias; C_S counts support-observed classes.", size=8.0)
    return d


def m2():
    d = Diagram("03_m2_retained_complement_geometry", "3", "M2 | Retained and orthogonal-complement geometry",
                "Source-fitted projections and normalization; fixed half-kernel mixing.", height=242)
    d.box(6, 33, 80, 25, "R50 feature f_50 | 2048-D",
          "Paired VV/VH; frozen pooled representation")
    d.box(97, 33, 80, 25, "SimCLR-pretrained R18 | 512-D",
          "VH grayscale; trunk only; M3 updates layer4")
    d.box(6, 70, 171, 30, "Center separately in each encoder's source coordinates",
          "c_j = f_j - mu_j; PCA rows U_j are source-fitted orthonormal directions.\n"
          "z_j = c_j U_j^T; d_j = c_j - (c_j U_j^T) U_j, for j in {50, 18}.\n"
          "Retain 128 R50 directions and 64 R18 directions; keep residuals in ambient coordinates.", size=8.1)
    d.arrow([(46, 58), (46, 70)], BLUE)
    d.arrow([(137, 58), (137, 70)], BLUE)
    d.box(6, 112, 80, 41, "Retained branch | 192-D",
          "R50 PCA128 source-standardized first\n"
          "Concatenate with R18 PCA64\n"
          "Source-standardize all 192 coordinates\n"
          "p = standardized192 / sqrt(192)", BLUE)
    d.box(97, 112, 80, 41, "Complement branch | 2560-D",
          "Concatenate [d_50, d_18]: 2048 + 512\n"
          "Subtract source complement mean\n"
          "Divide by source complement scale\n"
          "d = normalized2560 / sqrt(2560)", GREEN)
    d.arrow([(46, 100), (46, 112)], BLUE)
    d.arrow([(137, 100), (137, 112)], GREEN)
    d.box(6, 165, 80, 23, "Fixed branch weight sqrt(0.5)",
          "Retained coordinates also feed the prior", BLUE)
    d.box(97, 165, 80, 23, "Fixed branch weight sqrt(0.5)",
          "Explicit projection; no learned attention", GREEN)
    d.arrow([(46, 153), (46, 165)], BLUE)
    d.arrow([(137, 153), (137, 165)], GREEN)
    d.box(6, 201, 171, 22, "Joint map | 192 + 2560 + 1 = 2753 coordinates",
          "phi = [sqrt(0.5) p, sqrt(0.5) d, 1]     ->     M1 residual adaptation / M3 ridge solve", GREEN)
    d.arrow([(46, 188), (46, 201)], BLUE)
    d.arrow([(137, 188), (137, 201)], GREEN)
    d.label(6, 230, r"$K=\Phi\Phi^{\prime\top}=0.5K_p+0.5K_d+1$", size=9.5, color=GREEN)
    d.label(99, 231, "One shared, penalized bias coordinate.", size=8.0)
    return d


def m3():
    d = Diagram("04_m3_deployment_aligned_training", "4", "M3 | Source training through the deployed solver",
                "Solid arrows: forward computation. Dashed red arrows: differentiation.", height=243)
    d.box(6, 33, 80, 32, "Held source-port episode",
          "64 support rows + query minibatches of 128\n"
          "Support/query disjoint within the held port\n"
          "Source labels supervise the episode", BLUE)
    d.box(97, 33, 80, 32, "Fit on other source ports",
          "Exclude held port + overlapping identities\n"
          "Freeze source PCA, statistics and heads\n"
          "No post-training head/statistic refit", BLUE)
    d.box(6, 82, 47, 38, "R18 layer4",
          "Frozen layer3 prefix\n"
          "Support + query features\n"
          "512-D output\n"
          "Only trainable module", RED, size=8.0)
    d.box(66, 82, 49, 38, "M2 geometry",
          "PCA + complement\n"
          "Differentiable R18 path\n"
          "Source buffers fixed\n"
          "phi_S and phi_Q", BLUE)
    d.box(128, 82, 49, 38, "Dynamic prior",
          "Fixed head parameters\n"
          "Retained features in\n"
          "P_S and P_Q change\n"
          "as layer4 changes", BLUE)
    d.arrow([(29, 65), (29, 82)])
    d.arrow([(120, 65), (120, 73), (89, 73), (89, 82)], BLUE)
    d.arrow([(152, 65), (152, 82)], BLUE)
    d.arrow([(53, 97), (66, 97)])
    d.arrow([(115, 97), (128, 97)])
    d.arrow([(66, 111), (53, 111)], RED, True)
    d.arrow([(128, 111), (115, 111)], RED, True)
    d.box(6, 143, 47, 36, "Frozen R50 cache",
          "PCA128 + complement\n"
          "Same source geometry\n"
          "No R50 gradient\n"
          "Combined inside M2", BLUE, size=8.0)
    d.box(66, 143, 111, 36, "Actual 64-support ridge-8 adaptation (M1)",
          "Support labels -> +/-1 residual targets and class weights\n"
          "Delta = A^T (A A^T + 8 I_64)^(-1) W^(1/2) (T - P_S)\n"
          "Adapted query scores = P_Q + phi_Q Delta\n"
          "Use the final complement-enabled 2753-D map", GREEN, size=8.2)
    d.arrow([(53, 151), (59, 151), (59, 128), (75, 128), (75, 120)], BLUE)
    d.arrow([(85, 120), (85, 143)])
    d.arrow([(96, 143), (96, 120)], RED, True)
    d.arrow([(146, 120), (146, 143)])
    d.arrow([(158, 143), (158, 120)], RED, True)
    d.box(6, 195, 47, 24, "Update layer4 only",
          "Backpropagate through\nfeatures, prior and solve", RED, size=8.0)
    d.box(66, 195, 111, 24, "Source-class-balanced query cross-entropy",
          "All episode query minibatches contribute; frozen BN running statistics.\n"
          "No target labels train the encoder; no additional 5%-handoff CE.", RED, size=8.0)
    d.arrow([(113, 179), (113, 195)])
    d.arrow([(159, 195), (159, 179)], RED, True)
    d.arrow([(66, 207), (53, 207)], RED, True)
    d.arrow([(6, 207), (2.5, 207), (2.5, 101), (6, 101)], RED, True)
    d.note(228, "20 source-port episodes; six epochs; keep the final checkpoint (no target selection).")
    d.note(234, "AdamW: lr = 1e-5, weight decay = 1e-4; gradient clip = 5. Base64 and M4 stay fixed.")
    return d


def m4():
    d = Diagram("05_m4_oof_reliability_handoff", "5", "M4 | Source-learned reliability and budgeted handoff",
                "The inherited LR parameters read the actual receiving expert's adapted scores.", height=245)
    d.section(29, "A  Offline source OOF learning", BLUE)
    d.box(6, 38, 171, 22, "Out-of-port source base/expert scores + source truth",
          "Held port and shared identities excluded from head fitting; OOF is conditional on the encoder.")
    d.box(6, 72, 80, 37, "4-D confidence + weighted scaling",
          "x = [expert margin, base margin,\n"
          "       expert entropy, base entropy]\n"
          "Equal mass per port, then per present class\n"
          "Source counts and scaling before filtering", BLUE, size=8.0)
    d.box(97, 72, 80, 37, "Logistic regression | C = 1",
          "delta = 1[expert correct] - 1[base correct]\n"
          "Recovery (+1) vs harm (-1) on disagreements\n"
          "Discard zero-utility rows; use source weights\n"
          "Freeze center, scale, coefficients, intercept", PURPLE, size=8.0)
    d.arrow([(46, 60), (46, 72)], BLUE)
    d.arrow([(86, 91), (97, 91)], PURPLE)
    d.note(114, "Margin = top1 - top2 score; entropy = H(softmax(scores)), using all 8 class scores.", size=8.0)
    d.section(124, "B  Target query selection: fixed gate, adapted score inputs", PURPLE)
    d.box(6, 134, 80, 28, "Compute both adapted score matrices",
          "Base64 + actual joint expert64 scores\n"
          "Same 64 support labels; form 4-D x\n"
          "All expert features already computed", GREEN)
    d.box(97, 134, 80, 28, "Fixed receiving-expert gate",
          "g = ((x - center) / scale) dot beta + b\n"
          "Parameters inherited from source OOF\n"
          "No target gate refit or threshold tuning", PURPLE, size=8.0)
    d.arrow([(86, 148), (97, 148)], PURPLE)
    d.arrow([(177, 91), (180, 91), (180, 126), (137, 126), (137, 134)], PURPLE)
    d.box(6, 174, 171, 25, "Eligible query rows only",
          "Exclude all 64 supports; require base prediction != expert prediction and g > 0.\n"
          "The expert-predicted class must appear among the 64 support labels.", PURPLE)
    d.arrow([(137, 162), (137, 174)], PURPLE)
    d.box(6, 211, 80, 23, "Stable batch ranking",
          "Descending g; ties by ascending row ID\n"
          "Select at most floor(0.05 N_query) eligible", PURPLE, size=8.0)
    d.box(97, 211, 80, 23, "Final prediction",
          "Selected -> expert argmax\n"
          "Otherwise -> base argmax", PURPLE)
    d.arrow([(46, 199), (46, 211)], PURPLE)
    d.arrow([(86, 222), (97, 222)], PURPLE)
    d.note(238, "N_query excludes support rows. The cap counts changed decisions, not compute savings.", size=8.0)
    return d


def main():
    diagrams = [overall(), m1(), m2(), m3(), m4()]
    qa = []
    preview_dir = OUT / "qa_previews"
    preview_dir.mkdir(exist_ok=True)
    # Preserve exact manuscript dimensions rather than applying a tight crop.
    with mpl.rc_context({"savefig.bbox": None}):
        with PdfPages(OUT / "module_pipeline_diagrams.pdf") as merged:
            metadata = merged.infodict()
            metadata["Title"] = "Joint residual model: architecture and module processes"
            metadata["Subject"] = "Implementation-grounded M1-M4 schematics; no performance claims"
            for diagram in diagrams:
                record = diagram.validate_layout()
                stem = OUT / diagram.slug
                diagram.fig.savefig(stem.with_suffix(".pdf"))
                diagram.fig.savefig(stem.with_suffix(".svg"))
                diagram.fig.savefig(stem.with_suffix(".png"), dpi=300)
                merged.savefig(diagram.fig)
                with fitz.open(stem.with_suffix(".pdf")) as doc:
                    page = doc[0]
                    page.get_pixmap(dpi=150, alpha=False).save(preview_dir / f"{diagram.slug}.png")
                    record.update(pdf_pages=len(doc), pdf_raster_images=len(page.get_images()),
                                  pdf_width_mm=round(page.rect.width * 25.4 / 72, 2),
                                  pdf_height_mm=round(page.rect.height * 25.4 / 72, 2))
                with Image.open(stem.with_suffix(".png")) as png:
                    record.update(png_pixels=list(png.size), png_dpi=list(png.info["dpi"]))
                record["name"] = diagram.slug
                qa.append(record)
                print(f"Rendered {diagram.slug}: {record['text_elements']} labels; zero text collisions", flush=True)
                plt.close(diagram.fig)
    with fitz.open(OUT / "module_pipeline_diagrams.pdf") as doc:
        if len(doc) != 5:
            raise ValueError("Combined PDF must contain all five diagrams")
    (OUT / "figure_export_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print("Complete: 5 vector PDFs, 5 SVGs, 5 PNGs at 300 dpi, and the 5-page merged PDF.", flush=True)


if __name__ == "__main__":
    main()
