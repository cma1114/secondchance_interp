from __future__ import annotations

import argparse
import json
from pathlib import Path


FRAGMENT = r'''<div id="qwen-gla-boundary-lenses">
<style>
#qwen-gla-boundary-lenses{width:100%;color:var(--foreground);font-size:var(--font-size-base)}
#qwen-gla-boundary-lenses .qgb-controls{display:grid;grid-template-columns:minmax(120px,1fr) minmax(260px,2fr);gap:12px;margin-bottom:14px}
#qwen-gla-boundary-lenses .qgb-layer-label{font-weight:500;margin-bottom:4px}
#qwen-gla-boundary-lenses .qgb-lists{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin-top:14px}
#qwen-gla-boundary-lenses .qgb-row{display:grid;grid-template-columns:minmax(90px,180px) 1fr 66px;gap:8px;align-items:center;margin:7px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
#qwen-gla-boundary-lenses .qgb-track{height:12px;background:var(--muted)}
#qwen-gla-boundary-lenses .qgb-bar{height:100%;background:var(--viz-series-1)}
#qwen-gla-boundary-lenses .qgb-negative .qgb-bar{background:var(--viz-series-2)}
#qwen-gla-boundary-lenses .qgb-score{text-align:right}
#qwen-gla-boundary-lenses .qgb-token{overflow-wrap:anywhere}
@media(max-width:600px){#qwen-gla-boundary-lenses .qgb-controls,#qwen-gla-boundary-lenses .qgb-lists{grid-template-columns:1fr}}
</style>
<div class="qgb-controls">
<label class="form-label">Lens<select class="form-select" id="qgb-lens"></select></label>
<label class="form-label">Readout<select class="form-select" id="qgb-view"></select></label>
</div>
<div class="qgb-layer-label" id="qgb-layer-label"></div>
<input class="form-range" id="qgb-layer" type="range" min="0" value="0" aria-label="GLA block">
<div class="qgb-lists">
<section><h3>Positive-scoring tokens</h3><div id="qgb-positive"></div></section>
<section><h3>Negative-scoring tokens</h3><div id="qgb-negative"></div></section>
</div>
</div>
<script>
(()=>{const DATA=__DATA__,root=document.getElementById('qwen-gla-boundary-lenses'),get=id=>root.querySelector('#'+id),layers=Object.keys(DATA.layers).map(Number).sort((a,b)=>a-b);
function fill(select,values){select.replaceChildren(...values.map(value=>{const option=document.createElement('option');option.value=value;option.textContent=value;return option}))}
fill(get('qgb-lens'),DATA.lenses);fill(get('qgb-view'),DATA.views);
function drawRows(target,items,negative){const max=Math.max(...items.map(item=>Math.abs(item.score)),1e-9);target.replaceChildren(...items.map(item=>{const row=document.createElement('div');row.className='qgb-row'+(negative?' qgb-negative':'');const token=document.createElement('div');token.className='qgb-token';token.textContent=item.token;token.setAttribute('aria-label','Token '+item.token+', score '+item.score.toFixed(2));const track=document.createElement('div');track.className='qgb-track';const bar=document.createElement('div');bar.className='qgb-bar';bar.style.width=(100*Math.abs(item.score)/max)+'%';track.append(bar);const score=document.createElement('div');score.className='qgb-score';score.textContent=item.score.toFixed(2);row.append(token,track,score);return row}))}
function render(){const block=layers[Number(get('qgb-layer').value)],cell=DATA.layers[String(block)][get('qgb-lens').value][get('qgb-view').value];get('qgb-layer-label').textContent='GLA block '+block;drawRows(get('qgb-positive'),cell.positive,false);drawRows(get('qgb-negative'),cell.negative,true)}
get('qgb-layer').max=String(layers.length-1);get('qgb-layer').value=String(Math.max(0,layers.indexOf(33)));get('qgb-lens').value='R-lens';get('qgb-view').value='Before GLA: Evaluation minus Neutral';get('qgb-lens').addEventListener('input',render);get('qgb-view').addEventListener('input',render);get('qgb-layer').addEventListener('input',render);render()})();
</script>'''


def compact(data: dict, top_k: int) -> dict:
    lens_names = list(data["lenses"])
    first_layer = next(iter(data["layers"].values()))
    views = list(first_layer[lens_names[0]])
    layers = {}
    for block, lens_cells in data["layers"].items():
        layers[block] = {}
        for lens in lens_names:
            layers[block][lens] = {}
            for view in views:
                cell = lens_cells[lens][view]
                layers[block][lens][view] = {
                    "positive": cell["positive"][:top_k],
                    "negative": cell["negative"][:top_k],
                }
    return {"lenses": lens_names, "views": views, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    data = compact(json.loads(args.input.read_text()), args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(FRAGMENT.replace("__DATA__", json.dumps(data, ensure_ascii=False)))


if __name__ == "__main__":
    main()

