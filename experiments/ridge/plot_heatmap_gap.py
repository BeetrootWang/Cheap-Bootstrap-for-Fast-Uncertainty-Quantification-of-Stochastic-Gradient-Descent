import argparse
import csv
import os
from PIL import Image, ImageDraw, ImageFont


def fmt_val(val):
    return f"{val:g}"


def blend(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_for_gap(gap, max_abs):
    if gap is None:
        return "#dddddd"
    if max_abs <= 0:
        return "#ffffff"
    t = max(-1.0, min(1.0, gap / max_abs))
    white = (255, 255, 255)
    blue = (33, 102, 172)
    red = (178, 24, 43)
    if t < 0:
        rgb = blend(white, blue, abs(t))
    else:
        rgb = blend(white, red, t)
    return rgb_to_hex(rgb)


def text_color_for_bg(hex_color):
    if hex_color == "#dddddd":
        return "#000000"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 140 else "#ffffff"


def load_gap_table(csv_path, target):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kappa = float(row["kappa"])
                lam = float(row["lambda_reg"])
                mean_cov = float(row["mean_coverage"])
            except (KeyError, ValueError):
                continue
            gap = mean_cov - target
            rows.append((kappa, lam, gap))
    kappas = sorted({k for k, _, _ in rows})
    lambdas = sorted({l for _, l, _ in rows})
    grid = {(k, l): g for k, l, g in rows}
    return kappas, lambdas, grid


def load_summary_table(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kappa = float(row["kappa"])
                lam = float(row["lambda_reg"])
                mean_cov = float(row["mean_coverage"])
                avg_len = float(row["avg_len"])
            except (KeyError, ValueError):
                continue
            rows.append((kappa, lam, mean_cov, avg_len))
    kappas = sorted({k for k, _, _, _ in rows})
    lambdas = sorted({l for _, l, _, _ in rows})
    grid = {(k, l): (c, a) for k, l, c, a in rows}
    return kappas, lambdas, grid


def write_latex_table(path, kappas, lambdas, grid, caption=None, label=None):
    col_spec = "l" + "c" * len(lambdas)
    lines = []
    if caption or label:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header = "$\kappa$ \\backslash $\lambda$"
    header += " & " + " & ".join(fmt_val(l) for l in lambdas) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for k in kappas:
        row = [fmt_val(k)]
        for l in lambdas:
            val = grid.get((k, l))
            if val is None:
                row.append("--")
            else:
                cov, avg_len = val
                row.append(f"{cov:.3f} ({avg_len:.2f})")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    if caption or label:
        lines.append("\\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_matrix_csv(path, kappas, lambdas, grid):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kappa"] + [fmt_val(l) for l in lambdas])
        for k in kappas:
            row = [fmt_val(k)]
            for l in lambdas:
                gap = grid.get((k, l))
                row.append("" if gap is None else f"{gap:+.6f}")
            writer.writerow(row)


def draw_png_heatmap(path, title, kappas, lambdas, grid, target, max_abs):
    font = ImageFont.load_default()
    def text_size(text):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    cell_w = 90
    cell_h = 40
    header_w = 110
    header_h = 40
    title_h = 30 if title else 0
    width = header_w + cell_w * len(lambdas)
    height = title_h + header_h + cell_h * len(kappas)
    img = Image.new("RGB", (width + 1, height + 1), "white")
    draw = ImageDraw.Draw(img)

    if title:
        draw.text((10, 5), title, fill="#000000", font=font)

    y_offset = title_h
    draw.rectangle([0, y_offset, width, y_offset + header_h], fill="#f0f0f0", outline="#999999")
    draw.text((5, y_offset + 12), f"gap=coverage-{target:.2f}", fill="#000000", font=font)

    for j, lam in enumerate(lambdas):
        x0 = header_w + j * cell_w
        y0 = y_offset
        x1 = x0 + cell_w
        y1 = y0 + header_h
        draw.rectangle([x0, y0, x1, y1], fill="#f0f0f0", outline="#999999")
        text = fmt_val(lam)
        tw, th = text_size(text)
        draw.text((x0 + (cell_w - tw) / 2, y0 + (header_h - th) / 2), text, fill="#000000", font=font)

    for i, kappa in enumerate(kappas):
        x0 = 0
        y0 = y_offset + header_h + i * cell_h
        x1 = header_w
        y1 = y0 + cell_h
        draw.rectangle([x0, y0, x1, y1], fill="#f0f0f0", outline="#999999")
        text = fmt_val(kappa)
        tw, th = text_size(text)
        draw.text((x0 + (header_w - tw) / 2, y0 + (cell_h - th) / 2), text, fill="#000000", font=font)

        for j, lam in enumerate(lambdas):
            gap = grid.get((kappa, lam))
            color = color_for_gap(gap, max_abs)
            tcolor = text_color_for_bg(color)
            x0 = header_w + j * cell_w
            y0 = y_offset + header_h + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="#999999")
            text = "" if gap is None else f"{gap:+.3f}"
            tw, th = text_size(text)
            draw.text((x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2), text, fill=tcolor, font=font)

    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Plot heatmap of coverage gap (coverage - target).")
    parser.add_argument("--csv", required=True, help="Path to summary CSV.")
    parser.add_argument("--target", type=float, default=0.95, help="Target coverage (default 0.95).")
    parser.add_argument("--vmax", type=float, default=None, help="Max abs gap for color scale.")
    parser.add_argument("--out", default=None, help="Output file path.")
    parser.add_argument("--matrix-csv", default=None, help="Output pivot matrix CSV path.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--latex", action="store_true", help="Write a LaTeX table instead of a heatmap.")
    parser.add_argument("--caption", default=None, help="LaTeX caption text.")
    parser.add_argument("--label", default=None, help="LaTeX label text.")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.csv))[0]
    if args.latex:
        kappas, lambdas, grid = load_summary_table(args.csv)
        out_tex = args.out or os.path.join(os.path.dirname(args.csv), f"table_{base}.tex")
        caption = args.caption
        label = args.label
        write_latex_table(out_tex, kappas, lambdas, grid, caption=caption, label=label)
        print(f"Wrote {out_tex}")
        return

    kappas, lambdas, grid = load_gap_table(args.csv, args.target)
    gaps = [abs(g) for g in grid.values() if g is not None]
    max_abs = args.vmax if args.vmax is not None else (max(gaps) if gaps else 0.0)
    out_png = args.out or os.path.join(os.path.dirname(args.csv), f"heatmap_gap_{base}.png")
    out_csv = args.matrix_csv or os.path.join(os.path.dirname(args.csv), f"gap_matrix_{base}.csv")
    title = args.title or f"Coverage gap heatmap ({base})"

    write_matrix_csv(out_csv, kappas, lambdas, grid)
    draw_png_heatmap(out_png, title, kappas, lambdas, grid, args.target, max_abs)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
