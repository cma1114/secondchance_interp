from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rounded(values: np.ndarray, digits: int = 5):
    return np.round(values.astype(float), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.data, allow_pickle=False) as source:
        payload = {
            "q": source["question_ids"].tolist(),
            "rawBG": rounded(source["raw_baseline_game"]),
            "rawBN": rounded(source["raw_baseline_neutral"]),
            "centerBG": rounded(source["centered_baseline_game"]),
            "centerBN": rounded(source["centered_baseline_neutral"]),
            "mean": {
                name: rounded(source[f"mean_{name}"])
                for name in (
                    "raw_baseline_game",
                    "raw_baseline_neutral",
                    "raw_neutral_minus_game",
                    "centered_baseline_game",
                    "centered_baseline_neutral",
                    "centered_neutral_minus_game",
                )
            },
            "lo": {
                name: rounded(source[f"ci_low_{name}"])
                for name in (
                    "raw_baseline_game",
                    "raw_baseline_neutral",
                    "raw_neutral_minus_game",
                    "centered_baseline_game",
                    "centered_baseline_neutral",
                    "centered_neutral_minus_game",
                )
            },
            "hi": {
                name: rounded(source[f"ci_high_{name}"])
                for name in (
                    "raw_baseline_game",
                    "raw_baseline_neutral",
                    "raw_neutral_minus_game",
                    "centered_baseline_game",
                    "centered_baseline_neutral",
                    "centered_neutral_minus_game",
                )
            },
        }
    data = json.dumps(payload, separators=(",", ":"))
    fragment = f'''<div id="residual-cosine-explorer" style="width:100%;">
  <div class="viz-row" style="justify-content:space-between;align-items:end;margin-bottom:12px;">
    <label for="rce-question" style="min-width:min(100%,420px);">Question
      <select id="rce-question" class="select" style="display:block;width:100%;margin-top:4px;"></select>
    </label>
    <div class="viz-row text-small" aria-label="Legend" style="gap:16px;">
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--viz-series-1);vertical-align:middle;margin-right:6px;"></span>Baseline–Game</span>
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--viz-series-2);vertical-align:middle;margin-right:6px;"></span>Baseline–Neutral</span>
      <span><span style="display:inline-block;width:18px;border-top:3px solid var(--viz-series-3);vertical-align:middle;margin-right:6px;"></span>Neutral minus Game</span>
    </div>
  </div>
  <div id="rce-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:16px;"></div>
  <div id="rce-readout" class="text-small" style="margin-top:8px;color:var(--muted-foreground);min-height:1.4em;"></div>
</div>
<script>
(() => {{
  const root = document.getElementById('residual-cosine-explorer');
  const data = {data};
  const select = root.querySelector('#rce-question');
  const grid = root.querySelector('#rce-grid');
  const readout = root.querySelector('#rce-readout');
  const meanOption = document.createElement('option');
  meanOption.value = '-1';
  meanOption.textContent = 'Mean across 500 matched questions (95% CI)';
  select.appendChild(meanOption);
  data.q.forEach((qid, index) => {{
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${{index + 1}} — ${{qid}}`;
    select.appendChild(option);
  }});

  const panels = [
    {{title:'A  Raw residual cosine', kind:'pair', first:'raw_baseline_game', second:'raw_baseline_neutral'}},
    {{title:'B  Raw paired difference', kind:'diff', first:'raw_neutral_minus_game'}},
    {{title:'C  Question-centered residual cosine', kind:'pair', first:'centered_baseline_game', second:'centered_baseline_neutral'}},
    {{title:'D  Centered paired difference', kind:'diff', first:'centered_neutral_minus_game'}}
  ];
  const svgNS = 'http://www.w3.org/2000/svg';
  const make = (tag, attrs={{}}) => {{
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key,String(value)));
    return node;
  }};
  const pathFor = (values, x, y) => values.map((value,index) => `${{index ? 'L':'M'}}${{x(index).toFixed(2)}},${{y(value).toFixed(2)}}`).join(' ');
  const bandFor = (low, high, x, y) => {{
    const top = high.map((value,index) => `${{index ? 'L':'M'}}${{x(index).toFixed(2)}},${{y(value).toFixed(2)}}`).join(' ');
    const bottom = low.map((value,index) => `L${{x(low.length-1-index).toFixed(2)}},${{y(low[low.length-1-index]).toFixed(2)}}`).join(' ');
    return `${{top}} ${{bottom}} Z`;
  }};

  function seriesFor(panel, index) {{
    if (index < 0) return {{
      first:data.mean[panel.first], second:panel.second ? data.mean[panel.second] : null,
      firstLo:data.lo[panel.first], firstHi:data.hi[panel.first],
      secondLo:panel.second ? data.lo[panel.second] : null,
      secondHi:panel.second ? data.hi[panel.second] : null
    }};
    if (panel.first === 'raw_baseline_game') return {{first:data.rawBG[index],second:data.rawBN[index]}};
    if (panel.first === 'centered_baseline_game') return {{first:data.centerBG[index],second:data.centerBN[index]}};
    if (panel.first === 'raw_neutral_minus_game') return {{first:data.rawBN[index].map((v,i)=>v-data.rawBG[index][i]),second:null}};
    return {{first:data.centerBN[index].map((v,i)=>v-data.centerBG[index][i]),second:null}};
  }}

  function drawPanel(panel, index) {{
    const values = seriesFor(panel,index);
    const width=600, height=270, left=58, right=16, top=34, bottom=42;
    const all=[...values.first,...(values.second||[]),...(values.firstLo||[]),...(values.firstHi||[]),...(values.secondLo||[]),...(values.secondHi||[])];
    let ymin=0, ymax=1.02;
    if (panel.kind==='diff') {{
      const raw=seriesFor(panels[1],index), centered=seriesFor(panels[3],index);
      const shared=[...raw.first,...centered.first,...(raw.firstLo||[]),...(raw.firstHi||[]),...(centered.firstLo||[]),...(centered.firstHi||[])];
      ymin=Math.min(...shared); ymax=Math.max(...shared);
      const pad=Math.max((ymax-ymin)*0.05,0.002); ymin-=pad; ymax+=pad;
    }}
    const x=i=>left+i*(width-left-right)/64;
    const y=v=>top+(ymax-v)*(height-top-bottom)/(ymax-ymin);
    const svg=make('svg',{{viewBox:`0 0 ${{width}} ${{height}}`,role:'img','aria-label':panel.title,style:'width:100%;display:block;color:var(--foreground);'}});
    const title=make('text',{{x:left,y:20,fill:'var(--foreground)','font-weight':'500'}}); title.textContent=panel.title; svg.appendChild(title);
    for(let tick=0;tick<=4;tick++) {{
      const value=ymin+(ymax-ymin)*tick/4, yy=y(value);
      svg.appendChild(make('line',{{x1:left,x2:width-right,y1:yy,y2:yy,stroke:'var(--border)','stroke-width':1}}));
      const label=make('text',{{x:left-8,y:yy+4,'text-anchor':'end',fill:'var(--muted-foreground)','font-size':12}}); label.textContent=value.toFixed(panel.kind==='diff'?3:2); svg.appendChild(label);
    }}
    [0,16,32,48,64].forEach(layer=>{{
      const label=make('text',{{x:x(layer),y:height-17,'text-anchor':'middle',fill:'var(--muted-foreground)','font-size':12}}); label.textContent=layer; svg.appendChild(label);
    }});
    if(panel.kind==='diff' && ymin<0 && ymax>0) svg.appendChild(make('line',{{x1:left,x2:width-right,y1:y(0),y2:y(0),stroke:'var(--muted-foreground)','stroke-width':1}}));
    const colors=panel.kind==='pair' ? ['var(--viz-series-1)','var(--viz-series-2)'] : ['var(--viz-series-3)'];
    if(values.firstLo) svg.appendChild(make('path',{{d:bandFor(values.firstLo,values.firstHi,x,y),fill:colors[0],opacity:.13}}));
    if(values.secondLo) svg.appendChild(make('path',{{d:bandFor(values.secondLo,values.secondHi,x,y),fill:colors[1],opacity:.13}}));
    svg.appendChild(make('path',{{d:pathFor(values.first,x,y),fill:'none',stroke:colors[0],'stroke-width':2.5}}));
    if(values.second) svg.appendChild(make('path',{{d:pathFor(values.second,x,y),fill:'none',stroke:colors[1],'stroke-width':2.5}}));
    const cursor=make('line',{{x1:left,x2:left,y1:top,y2:height-bottom,stroke:'var(--foreground)','stroke-width':1,opacity:0}}); svg.appendChild(cursor);
    const hit=make('rect',{{x:left,y:top,width:width-left-right,height:height-top-bottom,fill:'transparent'}});
    hit.addEventListener('pointermove',event=>{{
      const box=svg.getBoundingClientRect(); const px=(event.clientX-box.left)*width/box.width; const layer=Math.max(0,Math.min(64,Math.round((px-left)*64/(width-left-right))));
      cursor.setAttribute('x1',x(layer)); cursor.setAttribute('x2',x(layer)); cursor.setAttribute('opacity','.55');
      const selected=index<0?'Mean':`Question ${{index+1}}`;
      if(panel.kind==='pair') readout.textContent=`${{selected}}, readout ${{layer}}: Baseline–Game ${{values.first[layer].toFixed(5)}}; Baseline–Neutral ${{values.second[layer].toFixed(5)}}; Neutral−Game ${{(values.second[layer]-values.first[layer]).toFixed(5)}}`;
      else readout.textContent=`${{selected}}, readout ${{layer}}: Neutral−Game ${{values.first[layer].toFixed(5)}}`;
    }});
    hit.addEventListener('pointerleave',()=>cursor.setAttribute('opacity','0'));
    svg.appendChild(hit);
    return svg;
  }}

  function render() {{
    const index=Number(select.value); grid.replaceChildren();
    panels.forEach(panel=>grid.appendChild(drawPanel(panel,index)));
    readout.textContent=index<0?'Hover over a panel for exact layer values.':data.q[index];
  }}
  select.addEventListener('change',render);
  render();
}})();
</script>
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)


if __name__ == "__main__":
    main()
