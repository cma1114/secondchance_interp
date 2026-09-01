from __future__ import annotations

import argparse
import json
from pathlib import Path


def _scenario(split: dict, scenario_id: str) -> dict | None:
    return next((row for row in split["scenarios"] if row["scenario"] == scenario_id), None)


def _average_rank_writes(rows: list[dict]) -> list[float]:
    return [
        sum(row["rank_writes"][rank]["mean"] for row in rows) / len(rows)
        for rank in range(4)
    ]


def build(payload_path: Path, output: Path, leader_path: Path | None = None) -> None:
    payload = json.loads(payload_path.read_text())
    jlens = payload["jlens"]
    trajectory = {}
    for name in ("incorrect", "neutral", "game_minus_neutral"):
        entry = jlens["conditions"][name]["rank_opposed_slope"]
        trajectory[name] = {
            "mean": entry["mean"],
            "low": entry["ci_low"],
            "high": entry["ci_high"],
            "opposition": jlens["conditions"][name]["baseline_opposition_coefficient"],
            "opposition_r2": jlens["conditions"][name]["baseline_opposition_r2"],
        }

    discovery = []
    for row in payload["causal"]["discovery"]["scenarios"]:
        if row["direction"] != "neutral_into_game" or row["n_targets"] != 1:
            continue
        metric = row["letter_macro"]
        discovery.append({
            "component": row["component"],
            "kind": row["kind"],
            "block": row["layer"] + 1,
            "mean": metric["mediation_mean"],
            "low": metric["mediation_ci_low"],
            "high": metric["mediation_ci_high"],
        })

    confirmed = payload["causal"]["confirmation"]
    followup = payload["causal"].get("rank_confirmation")
    signatures = []
    for component, source in (("mixer_l47", followup), ("mixer_l62", confirmed), ("mlp_l63", confirmed)):
        if source is None:
            continue
        rows = [
            row for row in source["scenarios"]
            if row["component"] == component and row["n_targets"] == 1
        ]
        if rows:
            signatures.append({
                "component": component.replace("mixer_l", "Mixer ").replace("mlp_l", "MLP "),
                "writes": _average_rank_writes(rows),
            })

    groups = []
    for label, source, stem in (
        ("Previously selected 8", confirmed, "all_candidates"),
        ("Rank-selected 7", followup, "all_rank_candidates"),
    ):
        if source is None:
            continue
        for direction, short in (("neutral_into_game", "Remove from Game"), ("game_into_neutral", "Add to Neutral")):
            row = _scenario(source, f"{direction}__{stem}")
            if row is not None:
                groups.append({
                    "label": label,
                    "direction": short,
                    "fraction": row["letter_macro"]["fraction_gap_mediated"],
                })

    leader = []
    if leader_path is not None:
        leader_payload = json.loads(leader_path.read_text())
        for row in leader_payload["rows"]:
            if row["component"] == "mlp_l63":
                leader.append({
                    "direction": "Remove from Game" if row["direction"] == "neutral_into_game" else "Add to Neutral",
                    "quartiles": [entry["suppression_mean"] for entry in row["margin_quartiles"]],
                    "fraction": row["logit_lens_fraction_leader_suppressed"],
                    "correlation": row["logit_lens_margin_pearson_r"],
                })

    data = json.dumps({
        "layers": jlens["layers"],
        "trajectory": trajectory,
        "discovery": discovery,
        "signatures": signatures,
        "groups": groups,
        "leader": leader,
    }, separators=(",", ":"))

    fragment = f'''<div id="rank-opposition-mechanism">
  <div class="panel-grid">
    <section class="panel" aria-labelledby="trajectory-heading">
      <h3 id="trajectory-heading">A. Ordered change by Baseline rank</h3>
      <div class="legend text-small" aria-label="Series legend">
        <span><i class="s1"></i>Game − Baseline</span>
        <span><i class="s2"></i>Neutral − Baseline</span>
        <span><i class="s3"></i>Game − Neutral</span>
      </div>
      <svg id="rank-trajectory" role="img" aria-label="Layerwise rank-opposed JLens slope with 95 percent confidence intervals"></svg>
    </section>
    <section class="panel" aria-labelledby="opposition-heading">
      <h3 id="opposition-heading">B. Question-specific Baseline opposition</h3>
      <div class="legend text-small"><span><i class="s1"></i>Game</span><span><i class="s2"></i>Neutral</span></div>
      <svg id="opposition-trajectory" role="img" aria-label="Coefficient of each condition's change opposite its same-question Baseline evidence vector"></svg>
    </section>
  </div>
  <div class="panel-grid">
    <section class="panel" aria-labelledby="components-heading">
      <h3 id="components-heading">C. Causal mediation by individual component</h3>
      <div class="legend text-small"><span><i class="dot s1"></i>Mixer</span><span><i class="square s2"></i>MLP</span></div>
      <svg id="component-sweep" role="img" aria-label="Held-out-target discovery estimates of rank-opposition mediation for every mixer and MLP"></svg>
    </section>
    <section class="panel" aria-labelledby="writes-heading">
      <h3 id="writes-heading">D. Rank-resolved causal signatures</h3>
      <div id="write-legend" class="legend text-small"></div>
      <svg id="rank-writes" role="img" aria-label="Causal final-logit writes by original Baseline rank for selected confirmed components"></svg>
    </section>
  </div>
  <section class="panel panel-wide" id="leader-panel" aria-labelledby="leader-heading">
    <h3 id="leader-heading">E. MLP 63 suppresses the current leader more when its margin is larger</h3>
    <div class="legend text-small"><span><i class="s1"></i>Remove from Game</span><span><i class="s2"></i>Add to Neutral</span></div>
    <svg id="leader-margin" role="img" aria-label="MLP 63 causal suppression of the current Game leader by incoming JLens margin quartile"></svg>
  </section>
  <section class="panel panel-wide" aria-labelledby="groups-heading">
    <h3 id="groups-heading">F. Jointly mediated Game–Neutral rank-opposition gap</h3>
    <div id="group-bars" class="group-bars" role="img" aria-label="Fraction of the natural Game minus Neutral rank-opposition gap mediated by grouped component patches"></div>
  </section>
</div>
<style>
  #rank-opposition-mechanism {{ color:var(--foreground); width:100%; }}
  #rank-opposition-mechanism .panel {{ margin:0 0 18px; min-width:0; }}
  #rank-opposition-mechanism .panel-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }}
  #rank-opposition-mechanism h3 {{ margin:0 0 6px; font-weight:500; }}
  #rank-opposition-mechanism .legend {{ display:flex; flex-wrap:wrap; gap:12px; color:var(--muted-foreground); margin-bottom:4px; }}
  #rank-opposition-mechanism .legend span {{ display:inline-flex; align-items:center; gap:5px; }}
  #rank-opposition-mechanism .legend i {{ width:18px; height:3px; display:inline-block; background:var(--viz-series-1); }}
  #rank-opposition-mechanism .legend i.s2 {{ background:var(--viz-series-2); }}
  #rank-opposition-mechanism .legend i.s3 {{ background:var(--viz-series-3); }}
  #rank-opposition-mechanism .legend i.dot {{ width:8px; height:8px; border-radius:50%; }}
  #rank-opposition-mechanism .legend i.square {{ width:8px; height:8px; }}
  #rank-opposition-mechanism svg {{ width:100%; height:auto; display:block; overflow:visible; }}
  #rank-opposition-mechanism .axis {{ stroke:var(--border); stroke-width:1; }}
  #rank-opposition-mechanism .grid {{ stroke:var(--border); stroke-width:1; opacity:.55; }}
  #rank-opposition-mechanism .tick {{ fill:var(--muted-foreground); font-size:11px; }}
  #rank-opposition-mechanism .axis-label {{ fill:var(--muted-foreground); font-size:11px; }}
  #rank-opposition-mechanism .annotation {{ fill:var(--foreground); font-size:11px; font-weight:500; }}
  #rank-opposition-mechanism .group-bars {{ display:grid; gap:8px; }}
  #rank-opposition-mechanism .bar-row {{ display:grid; grid-template-columns:minmax(120px,1.2fr) minmax(110px,1fr) minmax(180px,4fr) 52px; gap:8px; align-items:center; }}
  #rank-opposition-mechanism .bar-track {{ height:12px; background:var(--muted); position:relative; }}
  #rank-opposition-mechanism .bar-fill {{ height:100%; background:var(--viz-series-1); }}
  #rank-opposition-mechanism .bar-row:nth-child(even) .bar-fill {{ background:var(--viz-series-2); }}
  @media (max-width:620px) {{
    #rank-opposition-mechanism .panel-grid {{ grid-template-columns:1fr; gap:0; }}
    #rank-opposition-mechanism .bar-row {{ grid-template-columns:1fr 1fr; }}
    #rank-opposition-mechanism .bar-track {{ grid-column:1 / span 1; }}
  }}
</style>
<script>
(() => {{
  const root=document.getElementById('rank-opposition-mechanism');
  const data={data};
  const NS='http://www.w3.org/2000/svg';
  const make=(tag,attrs={{}})=>{{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el;}};
  const color=i=>`var(--viz-series-${{i}})`;
  function chart(svgId,width,height,margin,xDomain,yDomain,xLabel,yLabel){{
    const svg=root.querySelector('#'+svgId); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`);
    const x=v=>margin.l+(v-xDomain[0])/(xDomain[1]-xDomain[0])*(width-margin.l-margin.r);
    const y=v=>height-margin.b-(v-yDomain[0])/(yDomain[1]-yDomain[0])*(height-margin.t-margin.b);
    const g=make('g');svg.appendChild(g);
    const yt=[];for(let i=0;i<=4;i++)yt.push(yDomain[0]+i*(yDomain[1]-yDomain[0])/4);
    yt.forEach(v=>{{g.appendChild(make('line',{{x1:margin.l,x2:width-margin.r,y1:y(v),y2:y(v),class:'grid'}}));const t=make('text',{{x:margin.l-7,y:y(v)+4,'text-anchor':'end',class:'tick'}});t.textContent=v.toFixed(2);g.appendChild(t);}});
    const ticks=xDomain[0]>=40?[40,48,56,64]:[1,16,32,48,64];ticks.forEach(v=>{{const t=make('text',{{x:x(v),y:height-margin.b+17,'text-anchor':'middle',class:'tick'}});t.textContent=v;g.appendChild(t);}});
    g.appendChild(make('line',{{x1:margin.l,x2:width-margin.r,y1:y(0),y2:y(0),class:'axis'}}));
    const xl=make('text',{{x:(margin.l+width-margin.r)/2,y:height-3,'text-anchor':'middle',class:'axis-label'}});xl.textContent=xLabel;g.appendChild(xl);
    const yl=make('text',{{x:11,y:(margin.t+height-margin.b)/2,transform:`rotate(-90 11 ${{(margin.t+height-margin.b)/2}})`,'text-anchor':'middle',class:'axis-label'}});yl.textContent=yLabel;g.appendChild(yl);
    return {{svg,g,x,y,width,height,margin}};
  }}
  function path(points,x,y){{return points.map((p,i)=>`${{i?'L':'M'}}${{x(p[0]).toFixed(2)}},${{y(p[1]).toFixed(2)}}`).join(' ');}}

  const tr=chart('rank-trajectory',360,300,{{l:52,r:12,t:12,b:38}},[40,64],[-.12,.72],'JLens readout layer','Rank-opposed slope');
  const trSeries=[['incorrect',1],['neutral',2],['game_minus_neutral',3]];
  trSeries.forEach(([name,ci])=>{{
    const s=data.trajectory[name];
    const upper=data.layers.map((l,i)=>[l,s.high[i]]).filter(p=>p[0]>=40), lower=data.layers.map((l,i)=>[l,s.low[i]]).filter(p=>p[0]>=40).reverse();
    const band=path(upper,tr.x,tr.y)+' '+lower.map((p,i)=>`${{i?'L':'L'}}${{tr.x(p[0]).toFixed(2)}},${{tr.y(p[1]).toFixed(2)}}`).join(' ')+' Z';
    tr.g.appendChild(make('path',{{d:band,fill:color(ci),opacity:'0.08'}}));
    tr.g.appendChild(make('path',{{d:path(data.layers.map((l,i)=>[l,s.mean[i]]).filter(p=>p[0]>=40),tr.x,tr.y),fill:'none',stroke:color(ci),'stroke-width':'2'}}));
  }});
  tr.g.appendChild(make('line',{{x1:tr.x(48),x2:tr.x(48),y1:tr.margin.t,y2:tr.height-tr.margin.b,stroke:'var(--foreground)','stroke-width':'1','stroke-dasharray':'4 4',opacity:'.55'}}));
  const a48=make('text',{{x:tr.x(48)+5,y:tr.margin.t+12,class:'annotation'}});a48.textContent='L48';tr.g.appendChild(a48);

  const op=chart('opposition-trajectory',360,300,{{l:52,r:12,t:12,b:38}},[40,64],[-.45,.60],'JLens readout layer','Opposition coefficient');
  [['incorrect',1],['neutral',2]].forEach(([name,ci])=>{{const s=data.trajectory[name];const pts=data.layers.map((l,i)=>[l,s.opposition[i]]).filter(p=>p[0]>=40);op.g.appendChild(make('path',{{d:path(pts,op.x,op.y),fill:'none',stroke:color(ci),'stroke-width':'2'}}));}});
  op.g.appendChild(make('line',{{x1:op.x(48),x2:op.x(48),y1:op.margin.t,y2:op.height-op.margin.b,stroke:'var(--foreground)','stroke-width':'1','stroke-dasharray':'4 4',opacity:'.55'}}));
  const op64=make('text',{{x:op.x(64)-4,y:op.y(data.trajectory.incorrect.opposition[63])-7,'text-anchor':'end',class:'annotation'}});op64.textContent=`Game ${{data.trajectory.incorrect.opposition[63].toFixed(2)}}`;op.g.appendChild(op64);

  const cs=chart('component-sweep',360,300,{{l:48,r:12,t:14,b:38}},[1,64],[-.14,.14],'Transformer block','Mediated slope');
  data.discovery.forEach(d=>{{
    const shape=d.kind==='mixer'?'circle':'rect', attrs=d.kind==='mixer'?{{cx:cs.x(d.block),cy:cs.y(d.mean),r:3.1}}:{{x:cs.x(d.block)-3,y:cs.y(d.mean)-3,width:6,height:6}};
    Object.assign(attrs,{{fill:d.kind==='mixer'?color(1):color(2)}});cs.g.appendChild(make(shape,attrs));
    if(d.mean>.03 || d.mean<-.045){{const t=make('text',{{x:cs.x(d.block)+(d.kind==='mixer'?4:-4),y:cs.y(d.mean)-5,'text-anchor':d.kind==='mixer'?'start':'end',class:'tick'}});t.textContent=d.component.replace('_l',' ');cs.g.appendChild(t);}}
  }});
  cs.g.appendChild(make('line',{{x1:cs.x(48),x2:cs.x(48),y1:cs.margin.t,y2:cs.height-cs.margin.b,stroke:'var(--foreground)','stroke-width':'1','stroke-dasharray':'4 4',opacity:'.4'}}));

  const rw=chart('rank-writes',360,300,{{l:48,r:12,t:14,b:38}},[1,4],[-.28,.20],'Original Baseline rank','Game-like causal write');
  data.signatures.forEach((s,si)=>{{
    const c=color(si+1);rw.g.appendChild(make('path',{{d:path(s.writes.map((v,i)=>[i+1,v]),rw.x,rw.y),fill:'none',stroke:c,'stroke-width':'2'}}));
    s.writes.forEach((v,i)=>rw.g.appendChild(make('circle',{{cx:rw.x(i+1),cy:rw.y(v),r:3.5,fill:c}})));
    const span=document.createElement('span'),mark=document.createElement('i');mark.className=`s${{si+1}}`;span.appendChild(mark);span.appendChild(document.createTextNode(s.component));root.querySelector('#write-legend').appendChild(span);
  }});

  if(data.leader.length){{
    const lm=chart('leader-margin',720,250,{{l:52,r:18,t:12,b:42}},[1,4],[0,.9],'Incoming leader-margin quartile','Causal leader suppression');
    data.leader.forEach((s,si)=>{{const c=color(si+1),pts=s.quartiles.map((v,i)=>[i+1,v]);lm.g.appendChild(make('path',{{d:path(pts,lm.x,lm.y),fill:'none',stroke:c,'stroke-width':'2'}}));pts.forEach(p=>lm.g.appendChild(make('circle',{{cx:lm.x(p[0]),cy:lm.y(p[1]),r:4,fill:c}})));}});
  }} else {{ root.querySelector('#leader-panel').remove(); }}

  const bars=root.querySelector('#group-bars');
  data.groups.forEach(g=>{{const row=document.createElement('div');row.className='bar-row text-small';const label=document.createElement('span');label.textContent=g.label;const direction=document.createElement('span');direction.className='text-muted';direction.textContent=g.direction;const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${{Math.max(0,Math.min(100,100*g.fraction))}}%`;track.appendChild(fill);const value=document.createElement('span');value.textContent=`${{(100*g.fraction).toFixed(0)}}%`;row.append(label,direction,track,value);bars.appendChild(row);}});
}})();
</script>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the rank-opposition mechanism visualization")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--leader-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.output, args.leader_input)


if __name__ == "__main__":
    main()
