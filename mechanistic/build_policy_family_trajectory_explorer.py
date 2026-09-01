from __future__ import annotations

import argparse
import json
from pathlib import Path


POSITION_LABELS = {
    **{
        f"R{rank}_{kind}": f"2P R{rank}: {label}"
        for rank in range(1, 5)
        for kind, label in (
            ("letter", "option letter"),
            ("semantic", "semantic wordpieces (mean)"),
            ("newline", "closing newline"),
        )
    },
    "choice_cue_space": "Post-list answer-cue space",
    "final_decision": "Final decision position",
}


def compact(payload: dict) -> dict:
    condition_names = ("Game", "Neutral")
    family_names = (
        "incorrect / failed / mistake / wrong",
        "lost / again / resend / repeat",
    )
    series = {}
    for lens_index, lens in enumerate(("J-lens", "R-lens")):
        series[lens] = {}
        for position_index, position in enumerate(payload["position_names"]):
            rows = []
            for condition_index, condition in enumerate(condition_names):
                for family_index, family in enumerate(family_names):
                    rows.append(
                        {
                            "condition": condition,
                            "family": family,
                            "mean": [
                                round(
                                    float(
                                        payload["mean"][condition_index][lens_index][layer][position_index][family_index]
                                    ),
                                    4,
                                )
                                for layer in range(64)
                            ],
                            "lower": [
                                round(
                                    float(
                                        payload["lower"][condition_index][lens_index][layer][position_index][family_index]
                                    ),
                                    4,
                                )
                                for layer in range(64)
                            ],
                            "upper": [
                                round(
                                    float(
                                        payload["upper"][condition_index][lens_index][layer][position_index][family_index]
                                    ),
                                    4,
                                )
                                for layer in range(64)
                            ],
                        }
                    )
            series[lens][position] = rows
    all_bounds = [
        value
        for lens in series.values()
        for position in lens.values()
        for row in position
        for key in ("lower", "upper")
        for value in row[key]
    ]
    return {
        "layers": payload["layers"],
        "positions": payload["position_names"],
        "labels": POSITION_LABELS,
        "heldoutQuestions": payload["heldout_questions"],
        "metric": payload["metric"],
        "interval": payload["interval"],
        "families": payload["family_token_inventory"],
        "globalDomain": [min(all_bounds), max(all_bounds)],
        "series": series,
    }


def build_fragment(data: dict) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"""<div id="policy-family-trajectories">
  <div class="viz-controls" aria-label="Trajectory controls">
    <label class="form-label">Lens
      <select class="form-select" id="pft-lens">
        <option value="J-lens">J-lens</option>
        <option value="R-lens">R-lens</option>
      </select>
    </label>
    <label class="form-label">Destination
      <select class="form-select" id="pft-position"></select>
    </label>
  </div>
  <div class="pft-legend text-small" aria-label="Series legend">
    <span><i class="pft-swatch pft-error"></i>incorrect/failed/mistake/wrong · Game</span>
    <span><i class="pft-swatch pft-error pft-dashed"></i>incorrect/failed/mistake/wrong · Neutral</span>
    <span><i class="pft-swatch pft-loss"></i>lost/again/resend/repeat · Game</span>
    <span><i class="pft-swatch pft-loss pft-dashed"></i>lost/again/resend/repeat · Neutral</span>
  </div>
  <div class="pft-chart-wrap">
    <svg id="pft-chart" viewBox="0 0 720 390" role="img" aria-labelledby="pft-svg-title pft-svg-desc">
      <title id="pft-svg-title">Layerwise activation of the two requested word sets</title>
      <desc id="pft-svg-desc">Game and Neutral activation for morphological variants of the eight requested words over all 64 layers.</desc>
    </svg>
    <div class="tooltip" id="pft-tooltip" hidden></div>
  </div>
  <div class="pft-meta text-small text-muted" id="pft-meta"></div>
  <p class="sr-only" id="pft-summary"></p>
</div>
<style>
  #policy-family-trajectories {{ width: 100%; color: var(--foreground); }}
  #policy-family-trajectories .pft-legend {{ display: flex; flex-wrap: wrap; gap: 0.55rem 1rem; margin: 0.8rem 0 0.35rem; }}
  #policy-family-trajectories .pft-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
  #policy-family-trajectories .pft-swatch {{ display: inline-block; width: 1.8rem; border-top: 3px solid var(--viz-series-1); }}
  #policy-family-trajectories .pft-swatch.pft-loss {{ border-color: var(--viz-series-2); }}
  #policy-family-trajectories .pft-swatch.pft-dashed {{ border-top-style: dashed; }}
  #policy-family-trajectories .pft-chart-wrap {{ position: relative; width: 100%; }}
  #policy-family-trajectories #pft-chart {{ display: block; width: 100%; height: auto; color: var(--foreground); }}
  #policy-family-trajectories .pft-grid {{ stroke: var(--border); stroke-width: 1; }}
  #policy-family-trajectories .pft-axis {{ stroke: var(--muted-foreground); stroke-width: 1; }}
  #policy-family-trajectories .pft-label {{ fill: var(--muted-foreground); font-size: 12px; }}
  #policy-family-trajectories .pft-line {{ fill: none; stroke-width: 2.4; stroke-linejoin: round; stroke-linecap: round; }}
  #policy-family-trajectories .pft-ribbon {{ opacity: 0.09; }}
  #policy-family-trajectories .pft-game {{ stroke-dasharray: none; }}
  #policy-family-trajectories .pft-neutral {{ stroke-dasharray: 6 4; }}
  #policy-family-trajectories .pft-error-stroke {{ stroke: var(--viz-series-1); }}
  #policy-family-trajectories .pft-error-fill {{ fill: var(--viz-series-1); }}
  #policy-family-trajectories .pft-loss-stroke {{ stroke: var(--viz-series-2); }}
  #policy-family-trajectories .pft-loss-fill {{ fill: var(--viz-series-2); }}
  #policy-family-trajectories .pft-hit {{ fill: transparent; cursor: crosshair; }}
  #policy-family-trajectories .pft-guide {{ stroke: var(--muted-foreground); stroke-width: 1; }}
  #policy-family-trajectories .pft-meta {{ margin-top: 0.4rem; }}
  #policy-family-trajectories #pft-tooltip {{ position: absolute; pointer-events: none; max-width: 19rem; }}
  @media (max-width: 520px) {{
    #policy-family-trajectories .pft-legend {{ display: grid; grid-template-columns: 1fr; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    #policy-family-trajectories .pft-line {{ transition: none; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('policy-family-trajectories');
  const DATA = {encoded};
  const lens = root.querySelector('#pft-lens');
  const position = root.querySelector('#pft-position');
  const svg = root.querySelector('#pft-chart');
  const tooltip = root.querySelector('#pft-tooltip');
  const meta = root.querySelector('#pft-meta');
  const summary = root.querySelector('#pft-summary');
  const NS = 'http://www.w3.org/2000/svg';
  const W = 720, H = 390, M = {{ left: 64, right: 22, top: 18, bottom: 48 }};
  const PW = W - M.left - M.right, PH = H - M.top - M.bottom;

  DATA.positions.forEach(key => {{
    const option = document.createElement('option');
    option.value = key;
    option.textContent = DATA.labels[key];
    position.appendChild(option);
  }});
  position.value = 'R1_semantic';

  function el(name, attrs = {{}}, text = '') {{
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  }}
  const x = layer => M.left + (layer - 1) * PW / 63;
  const domain = DATA.globalDomain;
  const pad = Math.max(0.15, (domain[1] - domain[0]) * 0.05);
  const yMin = domain[0] - pad, yMax = domain[1] + pad;
  const y = value => M.top + (yMax - value) * PH / (yMax - yMin);
  const path = values => values.map((value, index) => `${{index ? 'L' : 'M'}}${{x(index + 1).toFixed(2)}},${{y(value).toFixed(2)}}`).join(' ');
  const ribbon = row => {{
    const upper = row.upper.map((value, index) => `${{index ? 'L' : 'M'}}${{x(index + 1).toFixed(2)}},${{y(value).toFixed(2)}}`).join(' ');
    const lower = row.lower.map((value, index) => `L${{x(64 - index).toFixed(2)}},${{y(row.lower[63 - index]).toFixed(2)}}`).join(' ');
    return `${{upper}}${{lower}}Z`;
  }};

  function draw() {{
    svg.replaceChildren(
      el('title', {{ id: 'pft-svg-title' }}, 'Layerwise activation of the two requested word sets'),
      el('desc', {{ id: 'pft-svg-desc' }}, `Raw ${{lens.value}} trajectories at ${{DATA.labels[position.value]}}.`)
    );
    for (let tick = 0; tick <= 4; tick++) {{
      const value = yMin + tick * (yMax - yMin) / 4;
      const py = y(value);
      svg.appendChild(el('line', {{ x1: M.left, y1: py, x2: W - M.right, y2: py, class: 'pft-grid' }}));
      svg.appendChild(el('text', {{ x: M.left - 9, y: py + 4, 'text-anchor': 'end', class: 'pft-label' }}, value.toFixed(1)));
    }}
    [1, 8, 16, 24, 32, 40, 48, 56, 64].forEach(layerNumber => {{
      const px = x(layerNumber);
      svg.appendChild(el('line', {{ x1: px, y1: M.top, x2: px, y2: H - M.bottom, class: 'pft-grid' }}));
      svg.appendChild(el('text', {{ x: px, y: H - M.bottom + 20, 'text-anchor': 'middle', class: 'pft-label' }}, String(layerNumber)));
    }});
    svg.appendChild(el('line', {{ x1: M.left, y1: H - M.bottom, x2: W - M.right, y2: H - M.bottom, class: 'pft-axis' }}));
    svg.appendChild(el('line', {{ x1: M.left, y1: M.top, x2: M.left, y2: H - M.bottom, class: 'pft-axis' }}));
    if (yMin < 0 && yMax > 0) svg.appendChild(el('line', {{ x1: M.left, y1: y(0), x2: W - M.right, y2: y(0), class: 'pft-axis' }}));
    svg.appendChild(el('text', {{ x: M.left + PW / 2, y: H - 9, 'text-anchor': 'middle', class: 'pft-label' }}, 'Post-layer residual'));
    svg.appendChild(el('text', {{ x: 16, y: M.top + PH / 2, transform: `rotate(-90 16 ${{M.top + PH / 2}})`, 'text-anchor': 'middle', class: 'pft-label' }}, 'Mean requested-word score'));

    const rows = DATA.series[lens.value][position.value];
    rows.forEach(row => {{
      const familyClass = row.family.startsWith('incorrect') ? 'pft-error' : 'pft-loss';
      const taskClass = row.condition === 'Game' ? 'pft-game' : 'pft-neutral';
      svg.appendChild(el('path', {{ d: ribbon(row), class: `pft-ribbon ${{familyClass}}-fill` }}));
      svg.appendChild(el('path', {{ d: path(row.mean), class: `pft-line ${{familyClass}}-stroke ${{taskClass}}` }}));
    }});

    const guide = el('line', {{ x1: x(1), y1: M.top, x2: x(1), y2: H - M.bottom, class: 'pft-guide', visibility: 'hidden' }});
    svg.appendChild(guide);
    const hit = el('rect', {{ x: M.left, y: M.top, width: PW, height: PH, class: 'pft-hit' }});
    hit.addEventListener('pointermove', event => {{
      const box = svg.getBoundingClientRect();
      const sx = (event.clientX - box.left) * W / box.width;
      const selectedLayer = Math.max(1, Math.min(64, Math.round(1 + (sx - M.left) * 63 / PW)));
      const px = x(selectedLayer);
      guide.setAttribute('x1', px); guide.setAttribute('x2', px); guide.setAttribute('visibility', 'visible');
      tooltip.hidden = false;
      tooltip.innerHTML = `<strong>Layer ${{selectedLayer}}</strong><br>${{rows.map(row => `${{row.family}}, ${{row.condition}}: ${{row.mean[selectedLayer - 1].toFixed(3)}}`).join('<br>')}}`;
      const tx = Math.max(4, Math.min(box.width - 210, px * box.width / W + 8));
      tooltip.style.left = `${{tx}}px`; tooltip.style.top = `${{Math.max(4, M.top * box.height / H)}}px`;
    }});
    hit.addEventListener('pointerleave', () => {{ guide.setAttribute('visibility', 'hidden'); tooltip.hidden = true; }});
    svg.appendChild(hit);
    meta.textContent = `${{DATA.labels[position.value]}} · ${{lens.value}} · ${{DATA.heldoutQuestions}} held-out questions · raw Game and Neutral means · global shared y-scale`;
    summary.textContent = `Layerwise ${{lens.value}} activation for morphological variants of the eight requested words at ${{DATA.labels[position.value]}}, shown separately for Game and Neutral.`;
  }}

  lens.addEventListener('input', draw);
  position.addEventListener('input', draw);
  draw();
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_fragment(compact(payload)))


if __name__ == "__main__":
    main()
