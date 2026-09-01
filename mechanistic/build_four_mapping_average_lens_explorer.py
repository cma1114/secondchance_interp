from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Four-mapping averaged option lens audit</title>
<style>
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;color:#202124;background:#fff}
#app{max-width:1420px;margin:0 auto;padding:24px}.controls{display:grid;grid-template-columns:2fr 1.2fr 1.2fr 1.2fr 1.2fr;gap:14px}
label{display:grid;gap:5px;font-size:14px;color:#555}select,input{font:inherit;padding:8px;border:1px solid #aaa;border-radius:7px;background:#fff}
.question{font-size:20px;line-height:1.45;margin:24px 0 10px}.meta{color:#666;margin-bottom:14px}.options{display:grid;grid-template-columns:1fr 1fr;gap:9px 35px;font-size:17px}.option.selected{font-weight:700;color:#1769aa}
.slider{display:grid;grid-template-columns:auto 1fr;gap:15px;align-items:center;margin:26px 0 10px}.layer{font-size:23px;min-width:165px}
svg{width:100%;height:330px;display:block}.grid{stroke:#e6e6e6;stroke-width:1}.axis{stroke:#777;stroke-width:1}.line{fill:none;stroke-width:2.4}.point{stroke:#fff;stroke-width:1.5}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 20px}.legend span:before{content:'';display:inline-block;width:18px;height:3px;margin:0 6px 4px 0;background:var(--c)}
.lists{display:grid;grid-template-columns:1fr 1fr;gap:35px}.lists.single{grid-template-columns:1fr}.token-row{display:grid;grid-template-columns:minmax(150px,1.4fr) 3fr 80px;gap:10px;align-items:center;margin:5px 0}
.token{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow:hidden;text-overflow:ellipsis}.track{height:10px;background:#eee}.bar{height:100%;background:#3190ef}.bottom .bar{background:#f07f2f}.score{text-align:right;font-variant-numeric:tabular-nums}.status{color:#666;margin:10px 0 20px}.note{color:#666;font-size:14px;line-height:1.4;margin-top:20px}
@media(max-width:900px){.controls{grid-template-columns:1fr 1fr}.options,.lists{grid-template-columns:1fr}}
</style></head><body><div id="app">
<div class="controls">
<label>Held-out sample<select id="sample"></select></label>
<label>Displayed option<select id="option"></select></label>
<label>Readout position<select id="anchor"><option value="line_end">Option-closing newline</option><option value="content_end">Final option-content token</option></select></label>
<label>Lens<select id="method"><option value="jlens">JLens</option><option value="logit_lens">Native logit lens</option></select></label>
<label>Alignment<select id="alignment"><option value="aligned">Correct content alignment</option><option value="shuffled">Shuffled-content control</option></select></label>
</div>
<div class="question" id="question"></div><div class="meta" id="meta"></div><div class="options" id="options"></div>
<div class="slider"><div class="layer" id="layerLabel"></div><input id="layer" type="range" min="1" max="64" value="40"></div>
<div id="status" class="status"></div><svg id="plot" viewBox="0 0 1200 330"></svg><div class="legend" id="legend"></div>
<div class="controls" style="grid-template-columns:1fr 4fr;margin-bottom:18px"><label>Vocabulary view<select id="view"><option value="contrast">Option-specific contrast</option><option value="raw">Raw averaged state</option></select></label><div class="note">The contrast subtracts the four-option mean vocabulary score within the question, exposing what is distinctive about the selected option.</div></div>
<div class="lists" id="lists"><div><h2>Highest-scoring tokens</h2><div id="top"></div></div><div id="bottomWrap"><h2>Lowest-scoring tokens</h2><div id="bottom" class="bottom"></div></div></div>
<div class="note">Each candidate residual is the raw average of four prompts in which the same option content appears once at A, B, C, and D. The plotted lines are the maximum option-contrast vocabulary score among substantive tokenizer tokens from each of the question's four answer texts. They are a lexical sanity check, not a general semantic-similarity metric.</div>
</div><script>
const DATA=__DATA__;
const LETTERS=['A','B','C','D']; const COLORS=['#3190ef','#f07f2f','#55b96b','#df64a6'];
const translations={
  '正确答案':'correct answer','答案':'answer','正确':'correct','错误':'incorrect',
  '选择':'choice','替代':'alternative','舞蹈':'dance','牙科':'dental',
  '牙':'tooth / dental','齿':'tooth','口腔':'oral cavity',
  '实':'real / actual','實':'real / actual','芝加哥':'Chicago'
};
const $=id=>document.getElementById(id); const controls=['sample','option','anchor','method','alignment','layer','view'];
function translated(token){const plain=token.trim();return translations[plain]?`${token} [English: ${translations[plain]}]`:token}
function setup(){DATA.question_ids.forEach((qid,i)=>{const a=DATA.audit[qid];const o=document.createElement('option');o.value=qid;o.textContent=`Question ${i+1} — winner ${a.baseline_answer}: ${a.options[a.baseline_answer]}`;$('sample').appendChild(o)});LETTERS.forEach(l=>{const o=document.createElement('option');o.value=l;$('option').appendChild(o)});controls.forEach(id=>$(id).addEventListener('input',render));render()}
function vectorLetter(qid,displayed){if($('alignment').value==='aligned')return displayed;const shift=DATA.shuffled_shift_by_question[qid];return LETTERS[(LETTERS.indexOf(displayed)+shift)%4]}
function render(){const qid=$('sample').value||DATA.question_ids[0],a=DATA.audit[qid],displayed=$('option').value||'A';$('question').textContent=a.question;$('meta').textContent=`Baseline answer: ${a.baseline_answer} (${a.options[a.baseline_answer]}) · Ground truth: ${a.correct_answer}`;$('options').innerHTML=LETTERS.map(l=>`<div class="option ${l===displayed?'selected':''}">${l}: ${a.options[l]}${l===displayed?' ← displayed content':''}</div>`).join('');[...$('option').options].forEach(o=>o.textContent=`${o.value}: ${a.options[o.value]}`);const layer=+$('layer').value;$('layerLabel').textContent=`Residual readout ${layer}`;const source=vectorLetter(qid,displayed);$('status').textContent=$('alignment').value==='aligned'?`Decoding the A–D-balanced average for “${a.options[displayed]}”.`:`Shuffled control: displayed content “${a.options[displayed]}” is deliberately assigned the averaged vector for “${a.options[source]}”.`;draw(qid,source,displayed,layer);tokens(qid,source,layer)}
function draw(qid,source,displayed,layer){const svg=$('plot'),ai=DATA.anchors.indexOf($('anchor').value),mi=DATA.methods.indexOf($('method').value),qi=DATA.question_ids.indexOf(qid),vi=LETTERS.indexOf(source);const series=DATA.lexical_scores[qi][ai][mi].map(row=>row[vi]);const flat=series.flat(),min=Math.min(...flat),max=Math.max(...flat),pad=(max-min||1)*.12,lo=min-pad,hi=max+pad,W=1200,H=330,m={l:68,r:22,t:20,b:42};const x=i=>m.l+i*(W-m.l-m.r)/63,y=v=>m.t+(hi-v)*(H-m.t-m.b)/(hi-lo);let s='';for(let g=0;g<5;g++){const v=lo+g*(hi-lo)/4,yy=y(v);s+=`<line class="grid" x1="${m.l}" x2="${W-m.r}" y1="${yy}" y2="${yy}"/><text x="${m.l-9}" y="${yy+4}" text-anchor="end" fill="#777" font-size="12">${v.toFixed(1)}</text>`}s+=`<line class="axis" x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/>`;[1,8,16,24,32,40,48,56,64].forEach(L=>{const xx=x(L-1);s+=`<text x="${xx}" y="${H-13}" text-anchor="middle" fill="#777" font-size="12">${L}</text>`});series[0].forEach((_,content)=>{const pts=series.map((row,i)=>`${x(i)},${y(row[content])}`).join(' ');s+=`<polyline class="line" stroke="${COLORS[content]}" points="${pts}"/>`;const v=series[layer-1][content];s+=`<circle class="point" cx="${x(layer-1)}" cy="${y(v)}" r="5" fill="${COLORS[content]}"/>`});svg.innerHTML=s;$('legend').innerHTML=LETTERS.map((l,i)=>`<span style="--c:${COLORS[i]}">${l}: ${DATA.audit[qid].options[l]}${l===displayed?' (target)':''}</span>`).join('')}
function tokens(qid,source,layer){const row=DATA.top_tokens[qid][String(layer)][$('method').value][$('anchor').value][source],contrast=$('view').value==='contrast';renderList($('top'),contrast?row.contrast_top:row.raw_top,false);$('bottomWrap').style.display=contrast?'block':'none';$('lists').classList.toggle('single',!contrast);if(contrast)renderList($('bottom'),row.contrast_bottom,true)}
function renderList(root,rows,bottom){const extent=Math.max(...rows.map(r=>Math.abs(r.score)),1e-9);root.innerHTML=rows.map(r=>`<div class="token-row"><div class="token" title="token ${r.token_id}">${translated(r.token)}</div><div class="track"><div class="bar" style="width:${100*Math.abs(r.score)/extent}%"></div></div><div class="score">${r.score.toFixed(2)}</div></div>`).join('')}
setup();
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
