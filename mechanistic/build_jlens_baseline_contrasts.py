from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> None:
    data = json.dumps(json.loads(source.read_text()), separators=(",", ":"), ensure_ascii=True)
    fragment = f'''<div id="jlens-baseline-contrasts">
  <div class="legend" aria-label="Fixed Baseline ranks">
    <span><i class="rank-one"></i>Original winner</span><span><i class="rank-two"></i>Original runner-up</span><span><i class="rank-three"></i>Original rank 3</span><span><i class="rank-four"></i>Original rank 4</span>
  </div>
  <div class="charts">
    <section><h3>A&nbsp;&nbsp;Game minus Baseline</h3><svg id="jbc-game" role="img"></svg></section>
    <section><h3>B&nbsp;&nbsp;Neutral minus Baseline</h3><svg id="jbc-neutral" role="img"></svg></section>
    <section class="wide"><h3>C&nbsp;&nbsp;Game minus Neutral</h3><svg id="jbc-game-neutral" role="img"></svg></section>
  </div>
  <div class="text-small note">JLens; paired within question; fixed Baseline ranks; centered across A-D; 95% confidence intervals. Shading marks low answer-decoding reliability.</div>
</div>
<style>
  #jlens-baseline-contrasts {{ width:100%; color:var(--foreground); }}
  #jlens-baseline-contrasts .legend {{ display:flex; flex-wrap:wrap; justify-content:center; gap:.45rem 1rem; margin-bottom:.65rem; color:var(--muted-foreground); }}
  #jlens-baseline-contrasts .legend span {{ display:inline-flex; align-items:center; gap:.35rem; }}
  #jlens-baseline-contrasts .legend i {{ width:1.25rem; height:2px; display:inline-block; background:var(--viz-series-1); }}
  #jlens-baseline-contrasts .legend .rank-two {{ background:var(--viz-series-2); }}
  #jlens-baseline-contrasts .legend .rank-three {{ background:var(--viz-series-3); }}
  #jlens-baseline-contrasts .legend .rank-four {{ background:var(--viz-series-4); }}
  #jlens-baseline-contrasts .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }}
  #jlens-baseline-contrasts .charts .wide {{ grid-column:1 / -1; }}
  #jlens-baseline-contrasts h3 {{ margin:0; font-weight:500; }}
  #jlens-baseline-contrasts svg {{ display:block; width:100%; height:310px; overflow:visible; }}
  #jlens-baseline-contrasts .note {{ color:var(--muted-foreground); text-align:center; margin-top:.35rem; }}
  @media (max-width:620px) {{ #jlens-baseline-contrasts .charts {{ grid-template-columns:1fr; }} #jlens-baseline-contrasts svg {{ height:275px; }} }}
</style>
<script>
(() => {{
  const root=document.getElementById('jlens-baseline-contrasts'); const data={data};
  const colors=['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)']; const NS='http://www.w3.org/2000/svg';
  const make=(name,attrs={{}})=>{{const el=document.createElementNS(NS,name); Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,String(value))); return el;}};
  const all=Object.values(data.contrasts).flatMap(row=>row.series.flatMap(series=>series.ci_low.concat(series.ci_high).map(Math.abs))); const limit=1.08*Math.max(...all);
  function draw(svg,key) {{
    const row=data.contrasts[key],width=650,height=310,left=58,right=14,top=12,bottom=44; svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`); svg.setAttribute('aria-label',`${{row.label}} fixed-rank JLens trajectories`);
    const x=i=>left+i*(width-left-right)/63, y=v=>top+(limit-v)*(height-top-bottom)/(2*limit);
    svg.appendChild(make('rect',{{x:left,y:top,width:x(46.5)-left,height:height-top-bottom,fill:'var(--muted)',opacity:.32}}));
    [-limit,-limit/2,0,limit/2,limit].forEach(tick=>{{const gy=y(tick); svg.appendChild(make('line',{{x1:left,x2:width-right,y1:gy,y2:gy,stroke:'var(--border)','stroke-width':tick===0?1.1:.7}})); const text=make('text',{{x:left-7,y:gy+4,'text-anchor':'end',fill:'var(--muted-foreground)','font-size':11}}); text.textContent=tick.toFixed(1); svg.appendChild(text);}});
    [1,8,16,24,32,40,48,56,64].forEach(readout=>{{const text=make('text',{{x:x(readout-1),y:height-18,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); text.textContent=readout; svg.appendChild(text);}});
    row.series.forEach((series,index)=>{{const upper=series.ci_high.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' '); const lower=series.ci_low.map((_v,i)=>`L${{x(63-i).toFixed(1)}},${{y(series.ci_low[63-i]).toFixed(1)}}`).join(' '); svg.appendChild(make('path',{{d:upper+' '+lower+' Z',fill:colors[index],opacity:.13,stroke:'none'}})); const line=series.mean.map((v,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(' '); svg.appendChild(make('path',{{d:line,fill:'none',stroke:colors[index],'stroke-width':1.8}}));}});
    svg.appendChild(make('line',{{x1:x(47),x2:x(47),y1:top,y2:height-bottom,stroke:'var(--muted-foreground)','stroke-width':1,'stroke-dasharray':'4 3'}}));
    const xlabel=make('text',{{x:(left+width-right)/2,y:height-2,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); xlabel.textContent='Residual readout'; svg.appendChild(xlabel); const ylabel=make('text',{{x:13,y:height/2,transform:`rotate(-90 13 ${{height/2}})`,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); ylabel.textContent='Centered contrast (logit units)'; svg.appendChild(ylabel);
  }}
  draw(root.querySelector('#jbc-game'),'game_minus_baseline'); draw(root.querySelector('#jbc-neutral'),'neutral_minus_baseline'); draw(root.querySelector('#jbc-game-neutral'),'game_minus_neutral');
}})();
</script>
'''
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(fragment)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); build(args.source,args.output)


if __name__ == "__main__": main()
