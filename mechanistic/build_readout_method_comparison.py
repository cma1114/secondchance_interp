from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text())
    compact = {
        "layers": payload["layers"],
        "ranks": payload["ranks"],
        "methods": payload["methods"],
    }
    data = json.dumps(compact, separators=(",", ":"), ensure_ascii=True)
    fragment = f'''<div id="readout-method-comparison">
  <div class="legend" aria-label="Fixed Baseline ranks">
    <span><i class="rank-one"></i>Original winner</span>
    <span><i class="rank-two"></i>Original runner-up</span>
    <span><i class="rank-three"></i>Original rank 3</span>
    <span><i class="rank-four"></i>Original rank 4</span>
  </div>
  <div class="charts">
    <section><h3>A&nbsp;&nbsp;Native logit lens</h3><svg id="rmc-native" role="img"></svg></section>
    <section><h3>B&nbsp;&nbsp;Jacobian lens</h3><svg id="rmc-jlens" role="img"></svg></section>
    <section><h3>C&nbsp;&nbsp;Cross-fitted pooled probe</h3><svg id="rmc-probe" role="img"></svg></section>
  </div>
  <div class="text-small note">Paired Game minus Neutral; fixed Baseline ranks; 95% confidence intervals. Shading marks the region where answer decoding is unreliable; dashed line marks readout 48.</div>
</div>
<style>
  #readout-method-comparison {{ width:100%; color:var(--foreground); }}
  #readout-method-comparison .legend {{ display:flex; flex-wrap:wrap; justify-content:center; gap:.45rem 1rem; margin-bottom:.65rem; color:var(--muted-foreground); }}
  #readout-method-comparison .legend span {{ display:inline-flex; align-items:center; gap:.35rem; }}
  #readout-method-comparison .legend i {{ width:1.25rem; height:2px; display:inline-block; background:var(--viz-series-1); }}
  #readout-method-comparison .legend .rank-two {{ background:var(--viz-series-2); }}
  #readout-method-comparison .legend .rank-three {{ background:var(--viz-series-3); }}
  #readout-method-comparison .legend .rank-four {{ background:var(--viz-series-4); }}
  #readout-method-comparison .charts {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.9rem; }}
  #readout-method-comparison h3 {{ margin:0; font-weight:500; }}
  #readout-method-comparison svg {{ display:block; width:100%; height:295px; overflow:visible; }}
  #readout-method-comparison .note {{ color:var(--muted-foreground); text-align:center; margin-top:.35rem; }}
  @media (max-width:700px) {{
    #readout-method-comparison .charts {{ grid-template-columns:1fr; gap:.75rem; }}
    #readout-method-comparison svg {{ height:265px; }}
  }}
</style>
<script>
(() => {{
  const root=document.getElementById('readout-method-comparison');
  const data={data};
  const colors=['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)'];
  const NS='http://www.w3.org/2000/svg';
  const make=(name,attrs={{}})=>{{const el=document.createElementNS(NS,name); Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,String(value))); return el;}};
  const logitMethods=['native_logit_lens','jlens'];
  const logitLimit=1.2*Math.max(...logitMethods.flatMap(method=>data.methods[method].series.flatMap(series=>series.ci_low.concat(series.ci_high).map(Math.abs))));
  function draw(svg,method) {{
    const row=data.methods[method];
    const width=520,height=295,left=55,right=12,top=12,bottom=43;
    svg.replaceChildren(); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
    svg.setAttribute('aria-label',`${{row.label}} fixed-rank Game minus Neutral trajectories across residual readouts 1 to 64`);
    const ownLimit=1.12*Math.max(...row.series.flatMap(series=>series.ci_low.concat(series.ci_high).map(Math.abs)),.1);
    const limit=logitMethods.includes(method)?logitLimit:ownLimit;
    const x=index=>left+index*(width-left-right)/63;
    const y=value=>top+(limit-value)*(height-top-bottom)/(2*limit);
    svg.appendChild(make('rect',{{x:left,y:top,width:x(46.5)-left,height:height-top-bottom,fill:'var(--muted)',opacity:.32}}));
    [-limit,-limit/2,0,limit/2,limit].forEach(tick=>{{
      const gy=y(tick); svg.appendChild(make('line',{{x1:left,x2:width-right,y1:gy,y2:gy,stroke:'var(--border)','stroke-width':tick===0?1.1:.7}}));
      const label=make('text',{{x:left-7,y:gy+4,'text-anchor':'end',fill:'var(--muted-foreground)','font-size':11}}); label.textContent=tick.toFixed(limit<.6?2:1); svg.appendChild(label);
    }});
    [1,8,16,24,32,40,48,56,64].forEach(readout=>{{const label=make('text',{{x:x(readout-1),y:height-18,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); label.textContent=readout; svg.appendChild(label);}});
    row.series.forEach((series,index)=>{{
      const upper=series.ci_high.map((value,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(value).toFixed(1)}}`).join(' ');
      const lower=series.ci_low.map((_value,i)=>`L${{x(63-i).toFixed(1)}},${{y(series.ci_low[63-i]).toFixed(1)}}`).join(' ');
      svg.appendChild(make('path',{{d:upper+' '+lower+' Z',fill:colors[index],opacity:.13,stroke:'none'}}));
      const line=series.mean.map((value,i)=>`${{i?'L':'M'}}${{x(i).toFixed(1)}},${{y(value).toFixed(1)}}`).join(' ');
      svg.appendChild(make('path',{{d:line,fill:'none',stroke:colors[index],'stroke-width':1.7}}));
    }});
    svg.appendChild(make('line',{{x1:x(47),x2:x(47),y1:top,y2:height-bottom,stroke:'var(--muted-foreground)','stroke-width':1,'stroke-dasharray':'4 3'}}));
    const xlabel=make('text',{{x:(left+width-right)/2,y:height-2,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); xlabel.textContent='Residual readout'; svg.appendChild(xlabel);
    const ylabel=make('text',{{x:12,y:height/2,transform:`rotate(-90 12 ${{height/2}})`,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':11}}); ylabel.textContent=method==='pooled_probe'?'Game - Neutral (Baseline SD)':'Game - Neutral (logit units)'; svg.appendChild(ylabel);
  }}
  draw(root.querySelector('#rmc-native'),'native_logit_lens');
  draw(root.querySelector('#rmc-jlens'),'jlens');
  draw(root.querySelector('#rmc-probe'),'pooled_probe');
}})();
</script>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the readout-method comparison visualization")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
