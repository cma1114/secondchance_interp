from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE = (
    "spread_baseline", "spread_game", "spread_neutral",
    "entropy_baseline", "entropy_game", "entropy_neutral",
)
DERIVED = (
    "compression_game_vs_baseline", "compression_neutral_vs_baseline",
    "entropy_increase_game_vs_baseline", "entropy_increase_neutral_vs_baseline",
)


def rounded(values: np.ndarray):
    return np.round(values.astype(float), 5).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as source:
        payload = {
            "q": source["question_ids"].tolist(),
            "values": {name: rounded(source[name]) for name in BASE},
            "mean": {name: rounded(source[f"mean_{name}"]) for name in BASE + DERIVED},
            "lo": {name: rounded(source[f"ci_low_{name}"]) for name in BASE + DERIVED},
            "hi": {name: rounded(source[f"ci_high_{name}"]) for name in BASE + DERIVED},
        }
    data = json.dumps(payload, separators=(",", ":"))
    fragment = f'''<div id="jlens-compression-explorer" style="width:100%;">
  <div class="viz-row" style="justify-content:space-between;align-items:end;margin-bottom:12px;">
    <label for="jce-question" style="min-width:min(100%,420px);">Question
      <select id="jce-question" class="select" style="display:block;width:100%;margin-top:4px;"></select>
    </label>
    <div class="viz-row text-small" aria-label="Legend" style="gap:16px;">
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--muted-foreground);vertical-align:middle;margin-right:6px;"></span>Baseline</span>
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--viz-series-1);vertical-align:middle;margin-right:6px;"></span>Game</span>
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--viz-series-2);vertical-align:middle;margin-right:6px;"></span>Neutral</span>
    </div>
  </div>
  <div id="jce-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:16px;"></div>
  <div id="jce-readout" class="text-small" style="margin-top:8px;color:var(--muted-foreground);min-height:1.4em;"></div>
</div>
<script>
(() => {{
  const root=document.getElementById('jlens-compression-explorer'); const data={data};
  const select=root.querySelector('#jce-question'), grid=root.querySelector('#jce-grid'), readout=root.querySelector('#jce-readout');
  const meanOption=document.createElement('option'); meanOption.value='-1'; meanOption.textContent='Mean across 500 matched questions (95% CI)'; select.appendChild(meanOption);
  data.q.forEach((qid,index)=>{{const option=document.createElement('option'); option.value=String(index); option.textContent=`${{index+1}} — ${{qid}}`; select.appendChild(option);}});
  const panels=[
    {{title:'A  A–D evidence spread',type:'spread',names:['spread_baseline','spread_game','spread_neutral'],ylabel:'JLens score SD'}},
    {{title:'B  Compression relative to Baseline',type:'compression',names:['compression_game_vs_baseline','compression_neutral_vs_baseline'],ylabel:'Baseline minus condition spread'}},
    {{title:'C  A–D entropy',type:'entropy',names:['entropy_baseline','entropy_game','entropy_neutral'],ylabel:'Entropy (bits)'}},
    {{title:'D  Entropy increase relative to Baseline',type:'entropyDelta',names:['entropy_increase_game_vs_baseline','entropy_increase_neutral_vs_baseline'],ylabel:'Condition minus Baseline'}},
  ];
  const svgNS='http://www.w3.org/2000/svg';
  const make=(tag,attrs={{}})=>{{const node=document.createElementNS(svgNS,tag); Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,String(v))); return node;}};
  const pathFor=(values,x,y)=>values.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(2)}},${{y(v).toFixed(2)}}`).join(' ');
  const bandFor=(low,high,x,y)=>{{const a=high.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(2)}},${{y(v).toFixed(2)}}`).join(' '); const b=low.map((_,i)=>{{const j=low.length-1-i;return `L${{x(j).toFixed(2)}},${{y(low[j]).toFixed(2)}}`;}}).join(' '); return `${{a}} ${{b}} Z`;}};
  function individual(name,index) {{
    if(data.values[name]) return data.values[name][index];
    if(name==='compression_game_vs_baseline') return data.values.spread_baseline[index].map((v,i)=>v-data.values.spread_game[index][i]);
    if(name==='compression_neutral_vs_baseline') return data.values.spread_baseline[index].map((v,i)=>v-data.values.spread_neutral[index][i]);
    if(name==='entropy_increase_game_vs_baseline') return data.values.entropy_game[index].map((v,i)=>v-data.values.entropy_baseline[index][i]);
    return data.values.entropy_neutral[index].map((v,i)=>v-data.values.entropy_baseline[index][i]);
  }}
  function draw(panel,index) {{
    const width=600,height=270,left=64,right=16,top=34,bottom=42;
    const series=panel.names.map(name=>({{name,values:index<0?data.mean[name]:individual(name,index),lo:index<0?data.lo[name]:null,hi:index<0?data.hi[name]:null}}));
    const all=series.flatMap(s=>[...s.values,...(s.lo||[]),...(s.hi||[])]); let ymin=Math.min(...all),ymax=Math.max(...all);
    if(panel.type==='compression'||panel.type==='entropyDelta'){{const span=Math.max(Math.abs(ymin),Math.abs(ymax),.02);ymin=-span*1.08;ymax=span*1.08;}} else {{const pad=Math.max((ymax-ymin)*.08,.02);ymin=Math.max(0,ymin-pad);ymax+=pad;}}
    const x=i=>left+i*(width-left-right)/63, y=v=>top+(ymax-v)*(height-top-bottom)/(ymax-ymin);
    const svg=make('svg',{{viewBox:`0 0 ${{width}} ${{height}}`,role:'img','aria-label':panel.title,style:'width:100%;display:block;color:var(--foreground);'}});
    const title=make('text',{{x:left,y:20,fill:'var(--foreground)','font-weight':'500'}});title.textContent=panel.title;svg.appendChild(title);
    for(let tick=0;tick<=4;tick++){{const value=ymin+(ymax-ymin)*tick/4,yy=y(value);svg.appendChild(make('line',{{x1:left,x2:width-right,y1:yy,y2:yy,stroke:'var(--border)','stroke-width':1}}));const label=make('text',{{x:left-8,y:yy+4,'text-anchor':'end',fill:'var(--muted-foreground)','font-size':12}});label.textContent=value.toFixed(2);svg.appendChild(label);}}
    [1,16,32,48,64].forEach(layer=>{{const label=make('text',{{x:x(layer-1),y:height-17,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':12}});label.textContent=layer;svg.appendChild(label);}});
    if(ymin<0&&ymax>0)svg.appendChild(make('line',{{x1:left,x2:width-right,y1:y(0),y2:y(0),stroke:'var(--muted-foreground)','stroke-width':1}}));
    const colors=panel.names.length===3?['var(--muted-foreground)','var(--viz-series-1)','var(--viz-series-2)']:['var(--viz-series-1)','var(--viz-series-2)'];
    series.forEach((s,i)=>{{if(s.lo)svg.appendChild(make('path',{{d:bandFor(s.lo,s.hi,x,y),fill:colors[i],opacity:.12}}));svg.appendChild(make('path',{{d:pathFor(s.values,x,y),fill:'none',stroke:colors[i],'stroke-width':2.5}}));}});
    const cursor=make('line',{{x1:left,x2:left,y1:top,y2:height-bottom,stroke:'var(--foreground)','stroke-width':1,opacity:0}});svg.appendChild(cursor);
    const hit=make('rect',{{x:left,y:top,width:width-left-right,height:height-top-bottom,fill:'transparent'}});
    hit.addEventListener('pointermove',event=>{{const box=svg.getBoundingClientRect(),px=(event.clientX-box.left)*width/box.width,layer=Math.max(1,Math.min(64,Math.round((px-left)*63/(width-left-right))+1));cursor.setAttribute('x1',x(layer-1));cursor.setAttribute('x2',x(layer-1));cursor.setAttribute('opacity','.55');const selected=index<0?'Mean':`Question ${{index+1}}`;readout.textContent=`${{selected}}, readout ${{layer}}: `+series.map(s=>`${{s.name.replaceAll('_',' ')}} ${{s.values[layer-1].toFixed(4)}}`).join('; ');}});
    hit.addEventListener('pointerleave',()=>cursor.setAttribute('opacity','0'));svg.appendChild(hit);return svg;
  }}
  function render(){{const index=Number(select.value);grid.replaceChildren();panels.forEach(panel=>grid.appendChild(draw(panel,index)));readout.textContent=index<0?'Hover over a panel for exact layer values.':data.q[index];}}
  select.addEventListener('change',render);render();
}})();
</script>
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)


if __name__ == "__main__":
    main()
