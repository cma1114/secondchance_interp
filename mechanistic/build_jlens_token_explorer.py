from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHOR_LABELS = {
    "first_question_end": "First question end",
    "first_answer_decision": "First-answer decision",
    "historical_answer_end": "Historical assistant content end",
    "feedback_subject_end": "Feedback subject token",
    "condition_keyword_end": 'Game "incorrect" / Neutral "lost"',
    "user_different": 'Game "different" (Game only)',
    "action_keyword_end": "Feedback action token",
    "feedback_end": 'Feedback end "."',
    "instruction_letter": 'Answer-only instruction "letter"',
    "instruction_choice": 'Answer-only instruction "choice"',
    "instruction_end": 'Answer-only instruction end "."',
    "repeated_choice": 'After repeated question: "choice"',
    "second_user_end": "Second user prompt final token",
    "decision": "Final decision position",
    "evaluation_period": 'Evaluation-closing period ["." after evaluation]',
    "action_period": 'Shared action-closing period ["." after "again"]',
}


def display_token(text: str) -> str:
    """Keep readable Unicode visible while spelling out control characters."""
    visible = []
    for character in text:
        if character == "\n":
            visible.append(r"\n")
        elif character == "\r":
            visible.append(r"\r")
        elif character == "\t":
            visible.append(r"\t")
        elif character.isprintable():
            visible.append(character)
        else:
            codepoint = ord(character)
            escape = f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            visible.append(escape)
    return "".join(visible)


def display_anchor_token(text: str) -> str:
    """Make whitespace in a decoded prompt token explicit in selector labels."""
    return display_token(text).replace(" ", "␠")


def labels_with_audited_tokens(audit: dict | None) -> dict[str, str]:
    labels = dict(ANCHOR_LABELS)
    if not audit:
        return labels
    anchors = list(audit["anchors"])
    tokens: dict[str, dict[str, set[str]]] = {
        anchor: {"incorrect": set(), "neutral": set()} for anchor in anchors
    }
    for key, trial in audit["trials"].items():
        condition = key.split("/", 1)[0]
        if condition not in {"incorrect", "neutral"}:
            continue
        for anchor, token in zip(anchors, trial["tokens"]):
            if token is not None:
                tokens[anchor][condition].add(display_anchor_token(token))
    for anchor in anchors:
        game = sorted(tokens[anchor]["incorrect"])
        neutral = sorted(tokens[anchor]["neutral"])
        if len(game) == 1 and game == neutral:
            suffix = f' [token "{game[0]}"]'
        elif len(game) == 1 and len(neutral) == 1:
            suffix = f' [Game "{game[0]}" / Neutral "{neutral[0]}"]'
        elif len(game) == 1:
            suffix = f' [Game "{game[0]}"]'
        elif len(neutral) == 1:
            suffix = f' [Neutral "{neutral[0]}"]'
        else:
            suffix = " [question-dependent token]"
        labels[anchor] = labels.get(anchor, anchor) + suffix
    return labels


def build(
    source: Path,
    output: Path,
    top_n: int = 10,
    exclude_system: bool = False,
    glossary_path: Path | None = None,
    position_audit_path: Path | None = None,
) -> None:
    document = json.loads(source.read_text())
    glossary = json.loads(glossary_path.read_text()) if glossary_path else {}
    raw = document["positions"]

    def select_rows(rows: list[dict], *, descending: bool) -> list[dict]:
        ordinary = [row for row in rows if not row.get("tracked")][:top_n]
        tracked = [row for row in rows if row.get("tracked")]
        return sorted(ordinary + tracked, key=lambda row: float(row["score"]), reverse=descending)

    def serialize_row(item: dict) -> list[object]:
        # Positional rows keep the self-contained explorer below the 2 MB inline
        # limit without dropping any layers, anchors, or displayed tokens.
        serialized: list[object] = [
            display_token(item["token"]),
            round(float(item["score"]), 3),
        ]
        if item.get("tracked"):
            serialized.append(1)
        return serialized

    audited_labels = labels_with_audited_tokens(
        json.loads(position_audit_path.read_text()) if position_audit_path else None
    )
    present_anchors = {
        key.split("/", 2)[1]
        for key in raw
        if key.count("/") >= 2
    }
    anchor_labels = {
        key: value
        for key, value in audited_labels.items()
        if key in present_anchors
        and (not exclude_system or not key.startswith("system_"))
    }
    payload: dict[str, dict[str, list[dict[str, list[dict[str, object]]]] | None]] = {}
    for mode in ("game_minus_neutral", "incorrect", "neutral"):
        payload[mode] = {}
        for anchor in anchor_labels:
            if f"{mode}/{anchor}/L0" not in raw:
                payload[mode][anchor] = None
                continue
            layers = []
            for layer in range(64):
                row = raw[f"{mode}/{anchor}/L{layer}"]
                layers.append({
                    "top": [serialize_row(item) for item in select_rows(row["top"], descending=True)],
                    "bottom": [serialize_row(item) for item in select_rows(row["bottom"], descending=False)],
                })
            payload[mode][anchor] = layers

    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    labels = json.dumps(anchor_labels, ensure_ascii=True, separators=(",", ":"))
    glosses = json.dumps(
        {display_token(token): gloss for token, gloss in glossary.items()},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    fragment = f'''<div id="jlens-unrestricted-explorer">
  <div class="viz-controls">
    <label class="form-label">Vocabulary view
      <select id="jue-mode" class="form-select">
        <option value="game_minus_neutral" selected>Game minus Neutral</option>
        <option value="incorrect">Game</option>
        <option value="neutral">Neutral</option>
      </select>
    </label>
    <label class="form-label">Prompt position
      <select id="jue-anchor" class="form-select"></select>
    </label>
    <label class="form-label">Residual readout <span id="jue-layer-label">47</span>
      <input id="jue-layer" class="form-range" type="range" min="1" max="64" value="47" step="1">
    </label>
  </div>
  <div class="overview-wrap">
    <svg id="jue-overview" role="img" aria-label="Largest unrestricted Game minus Neutral JLens vocabulary contrasts across layers"></svg>
  </div>
  <div class="lists">
    <section>
      <h3 id="jue-top-heading">Game-pointing tokens</h3>
      <div id="jue-top" class="token-list"></div>
    </section>
    <section>
      <h3 id="jue-bottom-heading">Neutral-pointing tokens</h3>
      <div id="jue-bottom" class="token-list"></div>
    </section>
  </div>
  <div id="jue-status" class="text-small" aria-live="polite"></div>
</div>
<style>
  #jlens-unrestricted-explorer {{ width: 100%; color: var(--foreground); }}
  #jlens-unrestricted-explorer .viz-controls {{ margin-bottom: .75rem; }}
  #jlens-unrestricted-explorer .overview-wrap {{ width: 100%; }}
  #jlens-unrestricted-explorer svg {{ display: block; width: 100%; height: 235px; overflow: visible; }}
  #jlens-unrestricted-explorer .lists {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top: .5rem; }}
  #jlens-unrestricted-explorer h3 {{ margin: 0 0 .4rem; }}
  #jlens-unrestricted-explorer .token-list {{ display: grid; gap: .22rem; }}
  #jlens-unrestricted-explorer .token-row {{ display: grid; grid-template-columns: minmax(9rem, 1.7fr) 2.3fr 3.4rem; align-items: center; gap: .45rem; }}
  #jlens-unrestricted-explorer .token-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  #jlens-unrestricted-explorer .bar-track {{ height: .55rem; background: color-mix(in srgb, var(--muted) 70%, transparent); }}
  #jlens-unrestricted-explorer .bar {{ height: 100%; background: var(--viz-series-1); }}
  #jlens-unrestricted-explorer .negative .bar {{ background: var(--viz-series-2); }}
  #jlens-unrestricted-explorer .tracked .token-name {{ font-weight: 500; }}
  #jlens-unrestricted-explorer .score {{ text-align: right; font-variant-numeric: tabular-nums; }}
  #jlens-unrestricted-explorer #jue-status {{ color: var(--muted-foreground); margin-top: .65rem; }}
  @media (max-width: 560px) {{
    #jlens-unrestricted-explorer .lists {{ grid-template-columns: 1fr; }}
    #jlens-unrestricted-explorer svg {{ height: 205px; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('jlens-unrestricted-explorer');
  const data = {data};
  const labels = {labels};
  const glosses = {glosses};
  const mode = root.querySelector('#jue-mode');
  const anchor = root.querySelector('#jue-anchor');
  const slider = root.querySelector('#jue-layer');
  const layerLabel = root.querySelector('#jue-layer-label');
  const svg = root.querySelector('#jue-overview');
  const topList = root.querySelector('#jue-top');
  const bottomList = root.querySelector('#jue-bottom');
  const status = root.querySelector('#jue-status');
  const topHeading = root.querySelector('#jue-top-heading');
  const bottomHeading = root.querySelector('#jue-bottom-heading');
  Object.entries(labels).forEach(([value, label]) => {{
    const option = document.createElement('option'); option.value = value; option.textContent = label;
    if (value === 'decision') option.selected = true;
    anchor.appendChild(option);
  }});
  const NS = 'http://www.w3.org/2000/svg';
  const make = (name, attrs={{}}) => {{
    const el = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    return el;
  }};
  const rowToken = row => row[0];
  const rowScore = row => row[1];
  const rowTracked = row => Boolean(row[2]);
  const tokenLabel = row => glosses[rowToken(row)] ? `${{glosses[rowToken(row)]}} · ${{rowToken(row)}}` : rowToken(row);
  const renderList = (node, rows, negative=false) => {{
    node.replaceChildren();
    const ceiling = Math.max(...rows.map(row => Math.abs(rowScore(row))), .001);
    rows.forEach(row => {{
      const label = tokenLabel(row);
      const line = document.createElement('div'); line.className = 'token-row' + (negative ? ' negative' : '') + (rowTracked(row) ? ' tracked' : '');
      line.setAttribute('aria-label', `${{label}}: ${{rowScore(row).toFixed(3)}}`);
      const name = document.createElement('span'); name.className = 'token-name'; name.textContent = label || '(empty)'; name.dataset.tooltip = label;
      const track = document.createElement('span'); track.className = 'bar-track';
      const bar = document.createElement('span'); bar.className = 'bar'; bar.style.width = `${{100 * Math.abs(rowScore(row)) / ceiling}}%`; track.appendChild(bar);
      const score = document.createElement('span'); score.className = 'score'; score.textContent = rowScore(row).toFixed(2);
      line.append(name, track, score); node.appendChild(line);
    }});
  }};
  const drawOverview = () => {{
    const rows = data[mode.value][anchor.value];
    svg.replaceChildren();
    if (!rows) {{
      svg.setAttribute('viewBox', '0 0 720 120');
      const note = make('text', {{x:360, y:62, 'text-anchor':'middle', fill:'var(--muted-foreground)', 'font-size':12}});
      note.textContent = 'This token position is absent from the selected condition.';
      svg.appendChild(note);
      return;
    }}
    const width = 720, height = 235, left = 42, right = 12, top = 12, bottom = 31;
    svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
    const positives = rows.map(row => row.top[0]);
    const negatives = rows.map(row => row.bottom[0]);
    const limit = Math.max(...positives.map(x => Math.abs(rowScore(x))), ...negatives.map(x => Math.abs(rowScore(x))), 1);
    const x = i => left + i * (width-left-right)/63;
    const y = v => top + (limit-v) * (height-top-bottom)/(2*limit);
    const zero = y(0);
    const axis = make('line', {{x1:left, x2:width-right, y1:zero, y2:zero, stroke:'var(--border)', 'stroke-width':1}}); svg.appendChild(axis);
    [-limit, 0, limit].forEach(v => {{
      const text = make('text', {{x:left-7, y:y(v)+4, 'text-anchor':'end', fill:'var(--muted-foreground)', 'font-size':11}}); text.textContent = v.toFixed(1); svg.appendChild(text);
    }});
    [1,8,16,24,32,40,48,56,64].forEach(layer => {{
      const text = make('text', {{x:x(layer-1), y:height-8, 'text-anchor':'middle', fill:'var(--muted-foreground)', 'font-size':11}}); text.textContent = layer; svg.appendChild(text);
    }});
    const linePath = series => series.map((item,i) => `${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(rowScore(item)).toFixed(1)}}`).join(' ');
    svg.appendChild(make('path', {{d:linePath(positives), fill:'none', stroke:'var(--viz-series-1)', 'stroke-width':1.8}}));
    svg.appendChild(make('path', {{d:linePath(negatives), fill:'none', stroke:'var(--viz-series-2)', 'stroke-width':1.8}}));
    const selected = Number(slider.value)-1;
    svg.appendChild(make('line', {{x1:x(selected), x2:x(selected), y1:top, y2:height-bottom, stroke:'var(--foreground)', 'stroke-width':1, opacity:.55}}));
    rows.forEach((row,i) => {{
      [[row.top[0], 'var(--viz-series-1)'], [row.bottom[0], 'var(--viz-series-2)']].forEach(([item,color]) => {{
        const dot = make('circle', {{cx:x(i), cy:y(rowScore(item)), r:i===selected?3.2:2, fill:color}});
        const title = make('title'); title.textContent = `L${{i+1}}  ${{tokenLabel(item)}}  ${{rowScore(item).toFixed(3)}}`; dot.appendChild(title); svg.appendChild(dot);
      }});
    }});
    const ylabel = make('text', {{x:12, y:height/2, transform:`rotate(-90 12 ${{height/2}})`, 'text-anchor':'middle', fill:'var(--muted-foreground)', 'font-size':11}}); ylabel.textContent=mode.value === 'game_minus_neutral' ? 'Game − Neutral JLens score' : 'JLens vocabulary score'; svg.appendChild(ylabel);
    const xlabel = make('text', {{x:(left+width-right)/2, y:height-8, 'text-anchor':'middle', fill:'var(--muted-foreground)', 'font-size':11}}); xlabel.textContent='Residual readout'; svg.appendChild(xlabel);
  }};
  const update = () => {{
    const layer = Number(slider.value); layerLabel.textContent = String(layer);
    const rows = data[mode.value][anchor.value];
    if (!rows) {{
      topList.replaceChildren(); bottomList.replaceChildren();
      topHeading.textContent = 'Highest-scoring tokens'; bottomHeading.textContent = 'Lowest-scoring tokens';
      drawOverview();
      status.textContent = `${{labels[anchor.value]}} is not present in ${{mode.options[mode.selectedIndex].text}}.`;
      return;
    }}
    const row = rows[layer-1];
    const paired = mode.value === 'game_minus_neutral';
    topHeading.textContent = paired ? 'Game-pointing tokens' : 'Highest-scoring tokens';
    bottomHeading.textContent = paired ? 'Neutral-pointing tokens' : 'Lowest-scoring tokens';
    renderList(topList, row.top, false); renderList(bottomList, row.bottom, true); drawOverview();
    const view = mode.options[mode.selectedIndex].text;
    status.textContent = `${{view}} at ${{labels[anchor.value]}}, readout ${{layer}}. Lines track the single most extreme unrestricted token at each layer; token identity may change between layers.`;
  }};
  mode.addEventListener('change', update); anchor.addEventListener('change', update); slider.addEventListener('input', update); update();
}})();
</script>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--exclude-system", action="store_true")
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--position-audit", type=Path)
    args = parser.parse_args()
    build(
        args.source,
        args.output,
        args.top_n,
        args.exclude_system,
        args.glossary,
        args.position_audit,
    )


if __name__ == "__main__":
    main()
