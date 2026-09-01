from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(payload: dict, top_k: int) -> dict:
    all_source_names = payload["source_position_names"]
    all_source_tokens = payload["source_tokens"]
    # The first three feedback tokens ("Your answer was") are textually
    # identical and precede the only task-defining word. They are retained in
    # the machine-readable audit but omitted from the user-facing explorer.
    source_names = all_source_names[3:]
    source_tokens = all_source_tokens[3:]
    positions = source_names + payload["destination_position_names"]
    labels = {
        name: f"Feedback token {token.split(':', 1)[0]}: {token.split(':', 1)[1]}"
        for name, token in zip(source_names, source_tokens)
    }
    for rank in range(1, 5):
        labels[f"R{rank}_letter"] = f"2P R{rank}: option letter"
        labels[f"R{rank}_semantic"] = f"2P R{rank}: semantic wordpieces (mean)"
        labels[f"R{rank}_newline"] = f"2P R{rank}: closing newline"
    labels["choice_cue_space"] = "Post-list answer-cue space"
    labels["final_decision"] = "Final decision position"
    output = {
        "positions": positions,
        "labels": labels,
        "sourceCount": len(source_names),
        "layers": payload["layers"],
        "heldoutQuestions": payload["heldout_questions"],
        "vocabularyFilter": payload["vocabulary_filter"],
        "semanticDefinition": payload["semantic_definition"],
        "data": {},
    }
    condition_keys = {"game": "incorrect_again", "neutral": "lost_again"}
    for lens_name, lens_key in (("J", "J-lens"), ("R", "R-lens")):
        output["data"][lens_name] = {}
        for layer in payload["layers"]:
            layer_rows = payload["readouts"][lens_key][str(layer)]
            compact_layer = {}
            for position in positions:
                compact_layer[position] = {
                    short: [
                        [item["token"], round(float(item["score"]), 4)]
                        for item in layer_rows[position][condition][:top_k]
                    ]
                    for short, condition in condition_keys.items()
                }
            output["data"][lens_name][str(layer)] = compact_layer
    return output


def build_fragment(data: dict) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"""<div id="qwen-policy-state-explorer">
  <div class="viz-controls" aria-label="State explorer controls">
    <label class="form-label">Lens
      <select class="form-select" id="qpse-lens">
        <option value="J">J-lens</option>
        <option value="R">R-lens</option>
      </select>
    </label>
    <label class="form-label">Position
      <select class="form-select" id="qpse-position"></select>
    </label>
    <label class="form-label qpse-layer-control">Post-layer residual: <output id="qpse-layer-value">36</output>
      <input class="form-range" id="qpse-layer" type="range" min="1" max="64" step="1" value="36" />
    </label>
  </div>
  <div class="qpse-meta text-small text-muted" id="qpse-meta"></div>
  <div class="qpse-panels" aria-live="polite">
    <section aria-labelledby="qpse-game-heading">
      <h3 id="qpse-game-heading">Game</h3>
      <div class="qpse-bars" id="qpse-game"></div>
    </section>
    <section aria-labelledby="qpse-neutral-heading">
      <h3 id="qpse-neutral-heading">Neutral</h3>
      <div class="qpse-bars" id="qpse-neutral"></div>
    </section>
  </div>
  <p class="sr-only" id="qpse-summary"></p>
</div>
<style>
  #qwen-policy-state-explorer {{ width: 100%; color: var(--foreground); }}
  #qwen-policy-state-explorer .qpse-layer-control {{ min-width: min(100%, 19rem); flex: 1 1 19rem; }}
  #qwen-policy-state-explorer .qpse-meta {{ margin: 0.7rem 0 1rem; }}
  #qwen-policy-state-explorer .qpse-panels {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; }}
  #qwen-policy-state-explorer h3 {{ margin: 0 0 0.55rem; font-weight: 500; }}
  #qwen-policy-state-explorer .qpse-bars {{ display: grid; gap: 0.32rem; }}
  #qwen-policy-state-explorer .qpse-row {{ display: grid; grid-template-columns: minmax(7.5rem, 1.1fr) minmax(5rem, 1.7fr) 4.6rem; gap: 0.55rem; align-items: center; min-width: 0; }}
  #qwen-policy-state-explorer .qpse-token {{ overflow: hidden; text-overflow: ellipsis; white-space: pre; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  #qwen-policy-state-explorer .qpse-track {{ height: 0.72rem; background: color-mix(in srgb, var(--muted) 72%, transparent); overflow: hidden; }}
  #qwen-policy-state-explorer .qpse-fill {{ display: block; height: 100%; min-width: 1px; background: var(--viz-series-1); transition: width 160ms ease; }}
  #qwen-policy-state-explorer section:last-child .qpse-fill {{ background: var(--viz-series-2); }}
  #qwen-policy-state-explorer .qpse-score {{ text-align: right; font-variant-numeric: tabular-nums; }}
  @media (max-width: 600px) {{
    #qwen-policy-state-explorer .qpse-panels {{ grid-template-columns: 1fr; }}
    #qwen-policy-state-explorer .qpse-row {{ grid-template-columns: minmax(7rem, 1.15fr) minmax(4rem, 1.35fr) 4.2rem; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    #qwen-policy-state-explorer .qpse-fill {{ transition: none; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('qwen-policy-state-explorer');
  const DATA = {encoded};
  const lens = root.querySelector('#qpse-lens');
  const position = root.querySelector('#qpse-position');
  const layer = root.querySelector('#qpse-layer');
  const layerValue = root.querySelector('#qpse-layer-value');
  const meta = root.querySelector('#qpse-meta');
  const game = root.querySelector('#qpse-game');
  const neutral = root.querySelector('#qpse-neutral');
  const summary = root.querySelector('#qpse-summary');

  const sourceGroup = document.createElement('optgroup');
  sourceGroup.label = 'Evaluation source states';
  const destinationGroup = document.createElement('optgroup');
  destinationGroup.label = 'Second-presentation and decision states';
  DATA.positions.forEach((key, index) => {{
    const option = document.createElement('option');
    option.value = key;
    option.textContent = DATA.labels[key];
    (index < DATA.sourceCount ? sourceGroup : destinationGroup).appendChild(option);
  }});
  position.append(sourceGroup, destinationGroup);
  position.value = 'source_3';

  function renderBars(target, rows, taskLabel) {{
    const maximum = Math.max(...rows.map(row => Math.abs(row[1])), 1e-9);
    target.replaceChildren(...rows.map(([token, score], index) => {{
      const row = document.createElement('div');
      row.className = 'qpse-row text-small';
      const tokenNode = document.createElement('code');
      tokenNode.className = 'qpse-token';
      tokenNode.textContent = JSON.stringify(token);
      const track = document.createElement('span');
      track.className = 'qpse-track';
      track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span');
      fill.className = 'qpse-fill';
      fill.style.width = `${{Math.max(1.5, 100 * Math.abs(score) / maximum)}}%`;
      track.appendChild(fill);
      const scoreNode = document.createElement('span');
      scoreNode.className = 'qpse-score';
      scoreNode.textContent = Number(score).toFixed(3);
      row.setAttribute('aria-label', `${{taskLabel}} rank ${{index + 1}}: ${{JSON.stringify(token)}}, score ${{Number(score).toFixed(3)}}`);
      row.append(tokenNode, track, scoreNode);
      return row;
    }}));
  }}

  function render() {{
    const selectedLens = lens.value;
    const selectedLayer = layer.value;
    const selectedPosition = position.value;
    const row = DATA.data[selectedLens][selectedLayer][selectedPosition];
    layerValue.value = selectedLayer;
    layerValue.textContent = selectedLayer;
    const group = DATA.positions.indexOf(selectedPosition) < DATA.sourceCount ? 'source state' : 'destination state';
    meta.textContent = `${{DATA.labels[selectedPosition]}} · ${{group}} · complete post-layer residual · held-out ${{DATA.heldoutQuestions}} questions · raw within-task readable-token scores`;
    renderBars(game, row.game, 'Game');
    renderBars(neutral, row.neutral, 'Neutral');
    summary.textContent = `At layer ${{selectedLayer}}, ${{DATA.labels[selectedPosition]}} under ${{selectedLens}}-lens. Game top tokens: ${{row.game.slice(0, 5).map(item => item[0].trim()).join(', ')}}. Neutral top tokens: ${{row.neutral.slice(0, 5).map(item => item[0].trim()).join(', ')}}.`;
  }}

  [lens, position, layer].forEach(control => control.addEventListener('input', render));
  render();
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    fragment = build_fragment(compact(payload, args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)


if __name__ == "__main__":
    main()
