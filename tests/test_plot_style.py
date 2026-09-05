# the style module has to apply its rcParams and write both formats where it says it does.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.plot_style import PALETTE, RC, save, use_style


def test_use_style_applies_the_serif_rcparams():
    use_style()
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["axes.spines.top"] is False


def test_palette_has_five_distinct_hex_colours():
    assert len(PALETTE) == len(set(PALETTE)) == 5
    assert all(c.startswith("#") and len(c) == 7 for c in PALETTE)


def test_rc_block_is_a_plain_mapping_of_known_keys():
    assert "figure.dpi" in RC and RC["figure.dpi"] == 300


def test_save_writes_png_and_pdf(tmp_path):
    use_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save(fig, "unit_test_figure", output_dir=tmp_path)
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
