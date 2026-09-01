from __future__ import annotations

import argparse
import json
from pathlib import Path


FRAGMENT = r'''<div id="qwen-workspace-lens">
<style>
#qwen-workspace-lens{width:100%;color:var(--foreground);font-size:var(--font-size-base)}
#qwen-workspace-lens .controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:14px}
#qwen-workspace-lens label{color:var(--muted-foreground)}
#qwen-workspace-lens select,#qwen-workspace-lens input{display:block;width:100%;margin-top:4px}
#qwen-workspace-lens .layer-label{font-weight:500;margin-bottom:4px}
#qwen-workspace-lens .lists{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin-top:14px}
#qwen-workspace-lens .token-row{display:grid;grid-template-columns:minmax(90px,180px) 1fr 64px;gap:8px;align-items:center;margin:7px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
#qwen-workspace-lens .track{height:12px;background:var(--muted)}
#qwen-workspace-lens .bar{height:100%;background:var(--viz-series-1)}
#qwen-workspace-lens .negative .bar{background:var(--viz-series-2)}
#qwen-workspace-lens .score{text-align:right}
#qwen-workspace-lens .token{overflow-wrap:anywhere}
@media(max-width:600px){#qwen-workspace-lens .controls,#qwen-workspace-lens .lists{grid-template-columns:1fr}}
</style>
<div class="controls">
<label>Lens<select id="qwl-lens"></select></label>
<label>Position<select id="qwl-position"></select></label>
<label>Vector<select id="qwl-vector"></select></label>
</div>
<div class="layer-label" id="qwl-layer-label"></div>
<input id="qwl-layer" type="range" min="0" value="0" aria-label="GLA block">
<div class="lists">
<section><h3>Positive-scoring tokens</h3><div id="qwl-positive"></div></section>
<section><h3>Negative-scoring tokens</h3><div id="qwl-negative"></div></section>
</div>
</div>
<script>
(()=>{const DATA=__DATA__,root=document.getElementById('qwen-workspace-lens'),get=id=>root.querySelector('#'+id),layers=Object.keys(DATA.layers).map(Number).sort((a,b)=>a-b),lenses=DATA.lenses,positions=DATA.positions;
function fill(select,values){select.replaceChildren(...values.map(value=>{const option=document.createElement('option');option.value=value;option.textContent=value;return option}))}
fill(get('qwl-lens'),lenses);fill(get('qwl-position'),positions);
function vectorNames(){const position=get('qwl-position').value;return [position+' / Evaluation',position+' / Matched Neutral',position+' / Evaluation minus Matched Neutral']}
function drawRows(target,items,negative){const max=Math.max(...items.map(item=>Math.abs(item.score)),1e-9);target.replaceChildren(...items.map(item=>{const row=document.createElement('div');row.className='token-row'+(negative?' negative':'');const token=document.createElement('div');token.className='token';token.textContent=item.token;token.title='token '+item.token_id;const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*Math.abs(item.score)/max)+'%';track.append(bar);const score=document.createElement('div');score.className='score';score.textContent=item.score.toFixed(2);row.append(token,track,score);return row}))}
function render(){const block=layers[Number(get('qwl-layer').value)],cell=DATA.layers[String(block)][get('qwl-lens').value][get('qwl-vector').value];get('qwl-layer-label').textContent='GLA block '+block;drawRows(get('qwl-positive'),cell.positive,false);drawRows(get('qwl-negative'),cell.negative,true)}
function resetVectors(){const prior=get('qwl-vector').value,names=vectorNames();fill(get('qwl-vector'),names);if(names.includes(prior))get('qwl-vector').value=prior;render()}
get('qwl-layer').max=String(layers.length-1);get('qwl-layer').value=String(Math.max(0,layers.indexOf(33)));get('qwl-lens').addEventListener('input',render);get('qwl-position').addEventListener('input',resetVectors);get('qwl-vector').addEventListener('input',render);get('qwl-layer').addEventListener('input',render);resetVectors()})();
</script>'''


def compact(data: dict) -> dict:
    positions = data["positions"]
    lens_names = list(data["lenses"])
    layers = {}
    for block, lens_cells in data["layers"].items():
        layers[block] = {}
        for lens in lens_names:
            layers[block][lens] = {}
            for position in positions:
                for suffix in ("Evaluation", "Matched Neutral", "Evaluation minus Matched Neutral"):
                    label = f"{position} / {suffix}"
                    cell = lens_cells[lens][label]
                    layers[block][lens][label] = {
                        "positive": cell["positive"][:12],
                        "negative": cell["negative"][:12],
                    }
    return {"positions": positions, "lenses": lens_names, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = compact(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(FRAGMENT.replace("__DATA__", json.dumps(data, ensure_ascii=False)))


if __name__ == "__main__":
    main()
