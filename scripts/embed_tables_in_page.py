# injects the generated readout tables into the synthesis page, converting markdown to html so nothing is retyped.
# uv run -m scripts.embed_tables_in_page --page results/synthesis/260905_diffing_recovers_midtraining.html --tables_dir results/tables/260910_readout_tables_v1 --agreement_path results/tables/260910_toolkit_table_v1/method_agreement.md --output_dir results/tables/260910_page_embed_v1 --seed 42

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

START = "<!-- generated-tables-start -->"
END = "<!-- generated-tables-end -->"


@dataclass
class EmbedConfig(Config):
    page: str = ""
    tables_dir: str = ""
    agreement_path: str = ""


def md_inline(text: str) -> str:
    """Backticks, bold and italics only; everything else is escaped so token strings cannot inject markup."""
    out = html.escape(text)
    out = re.sub(r"`([^`]*)`", r'<span class="mono">\1</span>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)


def md_to_html(markdown: str) -> str:
    """Handles the subset the table scripts emit: h3 headings, pipe tables, fenced blocks, and paragraphs."""
    out, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            out.append(f"<h3>{md_inline(line[4:])}</h3>")
            i += 1
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(html.escape(lines[i]))
                i += 1
            out.append("<pre class='mono'>" + "\n".join(block) + "</pre>")
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("- :")]
            cells = "".join(f"<th>{md_inline(c)}</th>" for c in header)
            table = [f"<tr>{cells}</tr>"]
            for row in body:
                table.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in row) + "</tr>")
            out.append('<div class="scroll"><table>' + "".join(table) + "</table></div>")
        elif line.strip():
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "```")):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{md_inline(' '.join(para))}</p>")
        else:
            i += 1
    return "\n".join(out)


def sections(markdown: str) -> dict[str, str]:
    """Splits a generated tables file on its own h3 headings, keyed by the heading text."""
    parts = re.split(r"^### ", markdown, flags=re.MULTILINE)
    found = {}
    for part in parts[1:]:
        title = part.splitlines()[0].strip()
        found[title] = "### " + part.rstrip()
    return found


def build_block(config: EmbedConfig) -> tuple[str, str]:
    tables = sections(Path(f"{config.tables_dir}/readout_tables.md").read_text())
    agreement = Path(config.agreement_path).read_text()

    summary_keys = [k for k in tables if k.startswith("Table 1")]
    appendix_keys = [k for k in tables if k.startswith("Table 2")]

    pieces = [
        START,
        '<div class="card">',
        '<span class="tag t-real">Raw readouts</span>',
        "<h2>What the readouts actually say, before any grading</h2>",
        "<p>Every token below is copied from a committed results file by "
        '<span class="mono">scripts/build_readout_tables.py</span>; none is hand-picked or retyped. '
        "Layer 22 is shown because it is the depth the write-up plan asked for, and layer 31 beside it because that "
        "is where readable content sits at all &mdash; at 22 the diff returns tokenizer debris in every row, "
        "including the value rows.</p>",
    ]
    pieces += [md_to_html(tables[k]) for k in summary_keys]
    pieces += [
        "<details><summary>Raw agent behaviour, chosen by a stated rule</summary>",
        md_to_html(tables["Table 3. Raw agent behaviour"]),
        "</details>",
        "</div>",
        '<div class="card">',
        '<span class="tag t-ctl">Appendix</span>',
        "<h2>Every readout cell, with the published toolkit's beside ours</h2>",
        "<details><summary>Open the full tables</summary>",
    ]
    pieces += [md_to_html(tables[k]) for k in appendix_keys]
    pieces += ["</details>", "</div>", END]
    return "\n".join(pieces), md_to_html(agreement)


def main():
    config = simple_parsing.parse(EmbedConfig, add_config_path_arg=True)
    assert config.page and config.tables_dir and config.agreement_path, "page, tables_dir and agreement_path required"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    block, agreement_html = build_block(config)
    page = Path(config.page)
    text = page.read_text()

    # re-running replaces the previous injection rather than stacking a second copy
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.DOTALL)
    else:
        anchor = "<h2>Seven readings that did not survive their own tests</h2>"
        assert text.count(anchor) == 1
        text = text.replace(anchor, block + "\n\n" + anchor, 1)

    marker = "<p><strong>The control cell is the result.</strong>"
    assert text.count(marker) == 1, "the published-method card is where the agreement table belongs"
    if "Method agreement: four readouts" not in text:
        text = text.replace(marker, agreement_html + "\n" + marker, 1)

    page.write_text(text)
    (output_dir / "injected_block.html").write_text(block + "\n\n" + agreement_html + "\n")
    print(f"Page updated: {config.page}")
    print(f"Injected block saved to {output_dir / 'injected_block.html'}")


main()
