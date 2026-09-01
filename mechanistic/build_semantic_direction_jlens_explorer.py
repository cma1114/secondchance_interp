from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>W1 semantic direction JLens</title><style>
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;color:#202124;background:#fff}#app{max-width:1200px;margin:auto;padding:24px}.controls{display:grid;grid-template-columns:2fr 1fr;gap:14px}label{display:grid;gap:5px;color:#555}select,input{font:inherit;padding:8px;border:1px solid #aaa;border-radius:7px;background:white}.question{font-size:20px;line-height:1.45;margin:24px 0 10px}.options{display:grid;grid-template-columns:1fr 1fr;gap:8px 30px}.winner{font-weight:700;color:#1769aa}.slider{display:grid;grid-template-columns:170px 1fr;gap:15px;align-items:center;margin:26px 0}.layer{font-size:22px}.token-row{display:grid;grid-template-columns:minmax(150px,1.4fr) 3fr 80px;gap:10px;align-items:center;margin:6px 0}.token{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden;text-overflow:ellipsis}.track{height:11px;background:#eee}.bar{height:100%;background:#3190ef}.negative .bar{background:#f07f2f}.score{text-align:right;font-variant-numeric:tabular-nums}.note{color:#666;line-height:1.45;margin:16px 0}
@media(max-width:800px){.controls,.options{grid-template-columns:1fr}}
</style></head><body><div id="app"><div class="controls"><label>Held-out sample<select id="sample"></select></label><label>Direction<select id="sign"><option value="positive">Positive v(W1)</option><option value="negative">Negative −v(W1)</option></select></label></div><div class="question" id="question"></div><div class="options" id="options"></div><div class="slider"><div class="layer" id="layerLabel"></div><input id="layer" type="range" min="1" max="64" value="48"></div><h2 id="heading"></h2><div id="tokens"></div><div class="note">This decodes the exact normalized residual-space direction used in the causal intervention. Negative means the vector multiplied by −1 before JLens transport, normalization, and unembedding; it is not merely the bottom of the positive token list.</div></div><script>
const DATA=__DATA__,LETTERS=['A','B','C','D'];const $=x=>document.getElementById(x);function setup(){DATA.question_ids.forEach((q,i)=>{const a=DATA.audit[q],o=document.createElement('option');o.value=q;o.textContent=`Question ${i+1} — W1 ${a.w1}: ${a.options[a.w1]}`;$('sample').appendChild(o)});['sample','sign','layer'].forEach(x=>$(x).addEventListener('input',render));render()}function render(){const q=$('sample').value||DATA.question_ids[0],a=DATA.audit[q],L=+$('layer').value,s=$('sign').value;$('question').textContent=a.question;$('options').innerHTML=LETTERS.map(x=>`<div class="${x===a.w1?'winner':''}">${x}: ${a.options[x]}${x===a.w1?' ← W1':''}</div>`).join('');$('layerLabel').textContent=`Residual readout ${L}`;$('heading').textContent=s==='positive'?'Highest-scoring tokens for +v(W1)':'Highest-scoring tokens for −v(W1)';const rows=DATA.top_tokens[q][String(L)][s],mx=Math.max(...rows.map(x=>Math.abs(x.score)),1e-9);$('tokens').innerHTML=rows.map(x=>`<div class="token-row ${s}"><div class="token" title="token ${x.token_id}">${x.token}</div><div class="track"><div class="bar" style="width:${100*Math.abs(x.score)/mx}%"></div></div><div class="score">${x.score.toFixed(2)}</div></div>`).join('')}setup();
</script></body></html>'''.replace("__DATA__", embedded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
