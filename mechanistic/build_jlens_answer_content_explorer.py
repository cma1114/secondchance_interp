from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHOR_LABELS = {
    "first_question_end": "First question end",
    "first_answer_decision": "First-answer decision",
    "historical_answer_end": "Historical assistant content end",
    "feedback_subject_end": 'Game "answer" (feedback 1) / Neutral "response"',
    "condition_keyword_end": 'Game "incorrect" / Neutral "lost"',
    "user_different": 'Game "different" (Game only)',
    "action_keyword_end": 'Game "answer" (feedback 2) / Neutral "again"',
    "feedback_end": 'Feedback end "."',
    "instruction_letter": 'Answer-only instruction "letter"',
    "instruction_choice": 'Answer-only instruction "choice"',
    "instruction_end": 'Answer-only instruction end "."',
    "repeated_choice": 'After repeated question: "choice"',
    "second_user_end": "Second user prompt final token",
    "decision": "Final decision position",
}


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    labels = json.dumps(ANCHOR_LABELS, separators=(",", ":"), ensure_ascii=True)
    fragment = f'''<div id="jlens-answer-representations">
  <div class="viz-controls">
    <label class="form-label">Prompt position
      <select id="jar-anchor" class="form-select"></select>
    </label>
    <label class="form-label">Condition
      <select id="jar-condition" class="form-select"></select>
    </label>
    <label class="form-label">Selected readout <span id="jar-layer-value">48</span>
      <input id="jar-layer" class="form-range" type="range" min="1" max="64" value="48" step="1">
    </label>
  </div>
  <div class="legend" aria-label="Original Baseline ranks">
    <span><i class="rank-one"></i>Original winner</span>
    <span><i class="rank-two"></i>Original runner-up</span>
    <span><i class="rank-three"></i>Original rank 3</span>
    <span><i class="rank-four"></i>Original rank 4</span>
  </div>
  <div class="charts">
    <section><h3>Baseline-ranked answer letters</h3><svg id="jar-letter" role="img"></svg></section>
    <section><h3>Option-content readout</h3><svg id="jar-content" role="img"></svg></section>
  </div>
  <div class="selected-values" id="jar-values" aria-live="polite"></div>
  <div class="text-small status" id="jar-status"></div>
</div>
<style>
  #jlens-answer-representations {{ width:100%; color:var(--foreground); }}
  #jlens-answer-representations .viz-controls {{ align-items:end; margin-bottom:.65rem; }}
  #jlens-answer-representations .form-label {{ min-width:min(100%, 13rem); }}
  #jlens-answer-representations .legend {{ display:flex; flex-wrap:wrap; gap:.45rem 1rem; margin:.1rem 0 .5rem; color:var(--muted-foreground); }}
  #jlens-answer-representations .legend span {{ display:inline-flex; align-items:center; gap:.35rem; }}
  #jlens-answer-representations .legend i {{ width:1.2rem; height:2px; display:inline-block; background:var(--viz-series-1); }}
  #jlens-answer-representations .legend .rank-two {{ background:var(--viz-series-2); }}
  #jlens-answer-representations .legend .rank-three {{ background:var(--viz-series-3); }}
  #jlens-answer-representations .legend .rank-four {{ background:var(--viz-series-4); }}
  #jlens-answer-representations .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }}
  #jlens-answer-representations h3 {{ margin:0; font-weight:500; }}
  #jlens-answer-representations svg {{ width:100%; height:300px; display:block; overflow:visible; }}
  #jlens-answer-representations .selected-values {{ display:grid; grid-template-columns:minmax(7rem,1.2fr) repeat(4,minmax(5rem,1fr)); gap:.25rem .65rem; align-items:baseline; margin-top:.35rem; font-variant-numeric:tabular-nums; }}
  #jlens-answer-representations .selected-values .head {{ color:var(--muted-foreground); }}
  #jlens-answer-representations .status {{ color:var(--muted-foreground); margin-top:.55rem; }}
  @media (max-width:620px) {{
    #jlens-answer-representations .charts {{ grid-template-columns:1fr; }}
    #jlens-answer-representations svg {{ height:270px; }}
    #jlens-answer-representations .selected-values {{ grid-template-columns:minmax(6rem,1fr) repeat(2,minmax(5rem,1fr)); }}
    #jlens-answer-representations .selected-values .hide-narrow {{ display:none; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById('jlens-answer-representations');
  const data = {data};
  const labels = {labels};
  const colors = ['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)'];
  const anchor = root.querySelector('#jar-anchor');
  const condition = root.querySelector('#jar-condition');
  const layer = root.querySelector('#jar-layer');
  const layerValue = root.querySelector('#jar-layer-value');
  const values = root.querySelector('#jar-values');
  const status = root.querySelector('#jar-status');
  const usableAnchors = data.anchors.filter(value => Object.keys(data.readouts.letter[value] || {{}}).length);
  usableAnchors.forEach(value => {{
    const option=document.createElement('option'); option.value=value; option.textContent=labels[value] || value;
    if(value==='decision') option.selected=true; anchor.appendChild(option);
  }});
  const availableConditions = () => Object.keys(data.readouts.letter[anchor.value] || {{}});
  const refreshConditions = () => {{
    const previous=condition.value; condition.replaceChildren();
    const ordered=['baseline','incorrect','neutral','game_minus_neutral'];
    ordered.filter(value => availableConditions().includes(value)).forEach(value => {{
      const option=document.createElement('option'); option.value=value; option.textContent=data.conditions[value]; condition.appendChild(option);
    }});
    condition.value=availableConditions().includes(previous) ? previous : (availableConditions().includes('game_minus_neutral') ? 'game_minus_neutral' : availableConditions()[0]);
  }};
  const NS='http://www.w3.org/2000/svg';
  const make=(name,attrs={{}})=>{{const el=document.createElementNS(NS,name); Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,String(value))); return el;}};
  function chart(svg, readout, title) {{
    const row=data.readouts[readout][anchor.value][condition.value];
    const width=720,height=300,left=58,right=18,top=15,bottom=42;
    svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
    svg.setAttribute('aria-label',`${{title}} across 64 residual readouts for ${{data.conditions[condition.value]}} at ${{labels[anchor.value]}}`);
    const extrema=row.series.flatMap(series=>series.ci_low.concat(series.ci_high).filter(Number.isFinite));
    let limit=Math.max(...extrema.map(Math.abs),.25)*1.08;
    const x=i=>left+i*(width-left-right)/63;
    const y=v=>top+(limit-v)*(height-top-bottom)/(2*limit);
    [-limit,-limit/2,0,limit/2,limit].forEach(tick=>{{
      const gy=y(tick); svg.appendChild(make('line',{{x1:left,x2:width-right,y1:gy,y2:gy,stroke:'var(--border)','stroke-width':tick===0?1.1:.7}}));
      const text=make('text',{{x:left-7,y:gy+4,'text-anchor':'end',fill:'var(--muted-foreground)','font-size':11}}); text.textContent=tick.toFixed(limit<2?2:1); svg.appendChild(text);
    }});
    [1,8,16,24,32,40,48,56,64].forEach(readoutIndex=>{{
      const text=make('text',{{x:x(readoutIndex-1),y:height-17,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); text.textContent=readoutIndex; svg.appendChild(text);
    }});
    row.series.forEach((series,index)=>{{
      const upper=series.ci_high.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' ');
      const lower=series.ci_low.map((v,i)=>`L${{x(63-i).toFixed(1)}},${{y(series.ci_low[63-i]).toFixed(1)}}`).join(' ');
      svg.appendChild(make('path',{{d:upper+' '+lower+' Z',fill:colors[index],opacity:.12,stroke:'none'}}));
      const path=series.mean.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' ');
      svg.appendChild(make('path',{{d:path,fill:'none',stroke:colors[index],'stroke-width':1.8}}));
    }});
    const selected=Number(layer.value)-1;
    svg.appendChild(make('line',{{x1:x(selected),x2:x(selected),y1:top,y2:height-bottom,stroke:'var(--foreground)','stroke-width':1,opacity:.55}}));
    row.series.forEach((series,index)=>svg.appendChild(make('circle',{{cx:x(selected),cy:y(series.mean[selected]),r:3.4,fill:colors[index]}})));
    const xlabel=make('text',{{x:(left+width-right)/2,y:height-2,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); xlabel.textContent='Residual readout (64 = natural final residual)'; svg.appendChild(xlabel);
    const ylabel=make('text',{{x:13,y:height/2,transform:`rotate(-90 13 ${{height/2}})`,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); ylabel.textContent='Centered JLens score (logit units)'; svg.appendChild(ylabel);
  }}
  function renderValues() {{
    values.replaceChildren();
    const selected=Number(layer.value)-1;
    const headers=['','Winner','Runner-up','Rank 3','Rank 4'];
    headers.forEach((text,index)=>{{const cell=document.createElement('span'); cell.className='head'+(index>2?' hide-narrow':''); cell.textContent=text; values.appendChild(cell);}});
    [['Letter code','letter'],['Option content','content']].forEach(([label,key])=>{{
      const title=document.createElement('span'); title.textContent=label; values.appendChild(title);
      data.readouts[key][anchor.value][condition.value].series.forEach((series,index)=>{{const cell=document.createElement('span'); cell.className=index>1?'hide-narrow':''; cell.textContent=series.mean[selected].toFixed(2); values.appendChild(cell);}});
    }});
  }}
  function update() {{
    layerValue.textContent=layer.value;
    chart(root.querySelector('#jar-letter'),'letter','Letter-code JLens trajectories');
    chart(root.querySelector('#jar-content'),'content','Option-content JLens trajectories');
    renderValues();
    const row=data.readouts.letter[anchor.value][condition.value];
    status.textContent=`${{data.conditions[condition.value]}} at ${{labels[anchor.value]}}; n=${{row.n}} questions. Ranks are fixed from the generated Baseline winner and its Baseline-logit runner-up. Shading is a 95% confidence interval.`;
    if(condition.value==='game_minus_neutral' && ['first_question_end','first_answer_decision','historical_answer_end','system_end'].includes(anchor.value)) {{
      status.textContent += ' This position precedes the Game/Neutral divergence, so the paired contrast is exactly zero.';
    }}
    if(row.balanced_accuracy_vs_condition_output) {{
      const index=Number(layer.value)-1;
      const content=data.readouts.content[anchor.value][condition.value];
      status.textContent += ` Balanced accuracy versus the eventual ${{data.conditions[condition.value]}} output: letter ${{(100*row.balanced_accuracy_vs_condition_output[index]).toFixed(1)}}%, content ${{(100*content.balanced_accuracy_vs_condition_output[index]).toFixed(1)}}%.`;
    }}
  }}
  anchor.addEventListener('change',()=>{{refreshConditions(); update();}}); condition.addEventListener('change',update); layer.addEventListener('input',update);
  refreshConditions(); condition.value=availableConditions().includes('game_minus_neutral') ? 'game_minus_neutral' : condition.value; update();
}})();
</script>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the interactive JLens answer-representation explorer")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
