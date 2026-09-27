/* Illustrative fixtures only: this is not a live model or benchmark replay. */
(() => {
  'use strict';
  const root = document.querySelector('#world-explorer');
  if (!root) return;
  const $ = id => root.querySelector(`#ex-${id}`);
  const scenes = {
  "living": {
    "name": "Living & kitchen",
    "sceneId": "104348511_171513654",
    "image": "assets/explorer/hssd_104348511_topdown.png",
    "imageSize": [
      1300,
      1500
    ],
    "crop": [
      120,
      100,
      1080,
      1080
    ],
    "object": "mug",
    "id": "023",
    "appearance": "Illustrative ceramic mug",
    "locations": [
      "Dining table",
      "Coffee table",
      "Kitchen island",
      "Unknown"
    ],
    "prior": [
      0.15,
      0.65,
      0.1,
      0.1
    ],
    "relation": "Supported by dining table",
    "pattern": "In this illustrative routine, a used mug is likely to move toward the living area. The coffee table is the first inspection hypothesis.",
    "activity": "Daily use may move the target between surfaces.",
    "recovered": "Kitchen island",
    "shape": "mug",
    "anchors": [
      [
        1030,
        870
      ],
      [
        686,
        408
      ],
      [
        487,
        985
      ]
    ],
    "start": [
      361,
      661
    ],
    "routes": [
      [
        [
          361,
          661
        ],
        [
          468,
          457
        ]
      ],
      [
        [
          468,
          457
        ],
        [
          528.5,
          695.3
        ],
        [
          719.1,
          870.6
        ],
        [
          788,
          1007
        ]
      ]
    ]
  },
  "dining": {
    "name": "Dining & kitchen",
    "sceneId": "102344280",
    "image": "assets/explorer/hssd_102344280_topdown.png",
    "imageSize": [
      1400,
      900
    ],
    "crop": [
      240,
      170,
      1150,
      585
    ],
    "object": "bowl",
    "id": "041",
    "appearance": "Illustrative ceramic bowl",
    "locations": [
      "Center table",
      "Dining table",
      "Kitchen sink",
      "Unknown"
    ],
    "prior": [
      0.15,
      0.65,
      0.1,
      0.1
    ],
    "relation": "Supported by center table",
    "pattern": "In this illustrative routine, a bowl is likely to move to the dining table. A fresh inspection tests that hypothesis.",
    "activity": "Daily use may move the target between surfaces.",
    "recovered": "Kitchen sink",
    "shape": "bowl",
    "anchors": [
      [
        1040,
        470
      ],
      [
        493,
        423
      ],
      [
        945.4,
        644.1
      ]
    ],
    "start": [
      720,
      310
    ],
    "routes": [
      [
        [
          720,
          310
        ],
        [
          650,
          429
        ]
      ],
      [
        [
          650,
          429
        ],
        [
          926.9,
          558.3
        ]
      ]
    ]
  }
};
  let shown = [0.15,0.65,0.10,0.10], beliefFrame = 0, story = false, storyRun = 0;
  const initial = [.96,.02,.01,.01];
  let key = 'living', hours = 48, stage = 'predict', busy = false, generation = 0;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const delay = ms => new Promise(resolve => setTimeout(resolve, reduced.matches ? 0 : ms));
  function beliefs() {
    const s = scenes[key], t = hours / 48;
    let p = initial.map((v,i) => v * (1-t) + s.prior[i] * t);
    if (stage === 'updated' || stage === 'found') {
      // P(not detected | candidate): 95% visibility at the inspected location.
      p = p.map((v,i) => v * (i === 1 ? .05 : 1));
      const z = p.reduce((a,b) => a+b,0); p = p.map(v => v/z);
    }
    if (stage === 'found') {
      // Positive detection: high likelihood at the recovered location, small false-positive likelihood elsewhere.
      p = p.map((v,i) => v * (i === 2 ? .98 : .002));
      const z = p.reduce((a,b) => a+b,0); p = p.map(v => v/z);
    }
    return p;
  }
  function percentages(p) {
    const values=p.map(v=>Math.floor(v*100));
    const order=p.map((v,i)=>({i,r:v*100-values[i]})).sort((a,b)=>b.r-a.r);
    const remaining=100-values.reduce((a,b)=>a+b,0);
    for(let i=0;i<remaining;i++) values[order[i].i]++;
    return values;
  }
  function mapLayout() {
    const [x,y,w,h]=scenes[key].crop, scale=Math.min(660/w,500/h);
    return {x,y,w,h,scale,left:(720-w*scale)/2,top:(600-h*scale)/2};
  }
  function mapPoint([x,y]) {
    const m=mapLayout();return [m.left+(x-m.x)*m.scale,m.top+(y-m.y)*m.scale];
  }
  // A small articulated 3D model projected into the same coordinates as the room.
  // Diagonal leg pairs share a trot phase; heading follows the path tangent.
  let robotHeading = -Math.PI / 2;
  function quadruped(heading, phase=0, walking=false) {
    const c=Math.cos(heading),s=Math.sin(heading);
    const bob=walking?Math.cos(phase*2)*.65:0;
    const point=([x,y,z])=>[(x*c-y*s-(x*s+y*c))*.82,(x*c-y*s+x*s+y*c)*.45-z];
    const coord=p=>point(p).map(n=>n.toFixed(2)).join(',');
    const poly=(v,fill,stroke='#8d8d8d')=>`<polygon points="${v.map(coord).join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width=".65" stroke-linejoin="round"/>`;
    function chassis(x,y,z,w,d,h,top,side) {
      const faces=[
        {v:[[x,y,z],[x+w,y,z],[x+w,y,z+h],[x,y,z+h]],fill:side},
        {v:[[x+w,y,z],[x+w,y+d,z],[x+w,y+d,z+h],[x+w,y,z+h]],fill:side},
        {v:[[x+w,y+d,z],[x,y+d,z],[x,y+d,z+h],[x+w,y+d,z+h]],fill:side},
        {v:[[x,y+d,z],[x,y,z],[x,y,z+h],[x,y+d,z+h]],fill:side}
      ].sort((a,b)=>a.v.reduce((n,p)=>n+point([p[0],p[1],0])[1],0)-b.v.reduce((n,p)=>n+point([p[0],p[1],0])[1],0));
      return faces.map(f=>poly(f.v,f.fill)).join('')+poly([[x,y,z+h],[x+w,y,z+h],[x+w,y+d,z+h],[x,y+d,z+h]],top,'#d4d4d4');
    }
    const legs=[[-18,-12,0],[18,-12,Math.PI],[-18,12,Math.PI],[18,12,0]].map(([x,y,offset])=>{
      const cycle=phase+offset,stride=walking?Math.sin(cycle)*6:0,lift=walking?Math.max(0,Math.cos(cycle))*5:0;
      const hip=[x,y,24+bob],knee=[x-5+stride*.4,y*1.1,13+lift*.4],foot=[x+stride+2,y*1.2,2+lift];
      const h=point(hip),k=point(knee),f=point(foot),near=point([x,y,0])[1];
      return {near,html:`<g class="ex-robot-leg"><path d="M${coord(hip)}L${coord(knee)}L${coord(foot)}" fill="none" stroke="#1c1c1c" stroke-width="5.5" stroke-linejoin="round" stroke-linecap="round"/><path d="M${coord(hip)}L${coord(knee)}" stroke="#c5c5c5" stroke-width="3.4"/><path d="M${coord(knee)}L${coord(foot)}" stroke="#909090" stroke-width="2.5"/><circle cx="${h[0]}" cy="${h[1]}" r="3.5" fill="#4d4d4d" stroke="#dedede" stroke-width=".8"/><circle cx="${k[0]}" cy="${k[1]}" r="2.4" fill="#555" stroke="#bdbdbd" stroke-width=".7"/><ellipse cx="${f[0]}" cy="${f[1]}" rx="3.6" ry="1.9" fill="#171717" stroke="#8b8b8b" stroke-width=".5"/></g>`};
    }).sort((a,b)=>a.near-b.near);
    let body=chassis(-25,-11,22+bob,50,22,11,'url(#ex-robot-metal)','#707070');
    body+=chassis(-14,-7,33+bob,26,14,2,'#3c3c3c','#555');
    body+=chassis(24,-8,24+bob,6,16,7,'#8c8c8c','#292929');
    // Forward stereo cameras, top lidar, side panel and restrained status light.
    for(const y of [-4,4]) {const p=point([30,y,28+bob]);body+=`<circle cx="${p[0]}" cy="${p[1]}" r="1.8" fill="#0c0c0c" stroke="#dfdfdf" stroke-width=".8"/>`;}
    const lidar=point([9,0,38+bob]);body+=`<ellipse cx="${lidar[0]}" cy="${lidar[1]}" rx="5" ry="3.5" fill="#2a2a2a" stroke="#d3d3d3" stroke-width="1"/>`;
    for(let i=0;i<4;i++)body+=`<path d="M${coord([-10+i*4,-4,35+bob])}L${coord([-10+i*4,4,35+bob])}" stroke="#858585" stroke-width=".8"/>`;
    return `<ellipse cy="2" rx="32" ry="12" fill="#090909" opacity=".35"/>${legs.slice(0,2).map(l=>l.html).join('')}${body}${legs.slice(2).map(l=>l.html).join('')}`;
  }
  function poseRobot(phase=0,walking=false) {$('agent').innerHTML=quadruped(robotHeading,phase,walking);}
  function robotState(text,mode='idle') {
    $('robot-status').textContent=text;root.dataset.robotState=mode;
  }
  function buildScene() {
    const s=scenes[key],m=mapLayout();
    $('task').textContent=`Find the ${s.object}`;
    const overlays=s.anchors.map((point,i)=>{
      const [a,b]=mapPoint(point),labelX=Math.min(648,Math.max(72,a)),labelY=b-44;
      return `<g><ellipse id="ex-halo-${i}" class="ex-halo" cx="${a}" cy="${b}" rx="55" ry="36" fill="url(#ex-glow)"/><circle cx="${a}" cy="${b}" r="17" fill="none" stroke="#fff" stroke-opacity=".65" stroke-dasharray="2 4"/><circle cx="${a}" cy="${b}" r="3" fill="#fff"/><path d="M${a} ${b-18}L${labelX} ${labelY+13}" stroke="#ddd" stroke-width="1"/><rect class="ex-location-chip" x="${labelX-66}" y="${labelY-24}" width="132" height="43" rx="6"/><text class="ex-location-label" x="${labelX}" y="${labelY-7}">${s.locations[i]}</text><text class="ex-probability" id="ex-prob-${i}" x="${labelX}" y="${labelY+10}"></text></g>`;
    }).join('');
    $('map').setAttribute('viewBox','0 0 720 600');
    $('map').innerHTML=`<title id="ex-map-title">HSSD ${s.sceneId}: ${s.name}</title><desc id="ex-map-desc"></desc><defs><linearGradient id="ex-robot-metal" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f5f5f5"/><stop offset=".45" stop-color="#bcbcbc"/><stop offset="1" stop-color="#ececec"/></linearGradient><radialGradient id="ex-glow"><stop stop-color="#fff" stop-opacity=".65"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient></defs><g class="ex-room-enter"><rect x="${m.left-1}" y="${m.top-1}" width="${m.w*m.scale+2}" height="${m.h*m.scale+2}" rx="2" fill="#181818" stroke="#666"/><svg class="ex-captured-scene" style="width:${m.w*m.scale}px;height:${m.h*m.scale}px;overflow:hidden;aspect-ratio:auto" x="${m.left}" y="${m.top}" width="${m.w*m.scale}" height="${m.h*m.scale}" viewBox="${s.crop.join(' ')}"><image href="${s.image}" width="${s.imageSize[0]}" height="${s.imageSize[1]}"/></svg><path id="ex-route-shadow" fill="none" stroke="#171717" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0"/><path id="ex-route" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0"/>${overlays}<g id="ex-agent" aria-label="Illustrative quadruped inspection robot"></g><g id="ex-scan"><ellipse class="ex-scan" rx="45" ry="30"/></g><g id="ex-object"><circle r="12" fill="#eee" stroke="#292929" stroke-width="1.5"/><path d="M-4-5H3V4H-4Z M3-3H6V1H3" fill="none" stroke="#525252" stroke-width="1.5"/></g></g>`;
    $('distribution').innerHTML=s.locations.map((name,i)=>`<div class="ex-bar" id="ex-bar-${i}"><div class="ex-bar-label"><span>${name}</span><strong id="ex-value-${i}"></strong></div><div class="ex-track"><i id="ex-fill-${i}"></i></div></div>`).join('');
    if(s.shape==='bowl')$('object').innerHTML='<circle r="12" fill="#eee" stroke="#292929" stroke-width="1.5"/><path d="M-7-2Q-6 7 0 7Q6 7 7-2Z" fill="#999" stroke="#333" stroke-width="1.2"/>';
    $('robot-preview').innerHTML=quadruped(-.35);$('scene-source').textContent=`HSSD ${s.sceneId}`;$('scene-source').href=s.image;
    shown=beliefs();
    root.querySelectorAll('[data-scene]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scene===key)));
  }
  function render() {
    if(!busy)robotState(stage==='found'?'Target confirmed':stage==='updated'?'Ready to replan':'Standing by',stage==='found'?'success':'idle');
    const s=scenes[key], p=beliefs(), pc=percentages(p), leader=p.indexOf(Math.max(...p));
    animateBelief(p);
    $('map-clock').textContent=`DAY ${String(1+Math.floor((9+hours)/24)).padStart(2,'0')} / ${String((9+hours)%24).padStart(2,'0')}:00`;
    $('map-desc').textContent=`HSSD ${s.sceneId}; captured scene with illustrative beliefs and robot. ${s.locations.map((n,i)=>`${n}: ${pc[i]}%`).join(', ')}. ${stage==='found'?'Object recovered.':hours===0?'Object directly observed.':'Object location unobserved.'}`;
    $('time').value=hours; $('time').setAttribute('aria-valuetext',`${hours} hours since observation`); $('elapsed').textContent=hours===0?'Just observed':`${hours} hours`;
    root.querySelectorAll('[data-time]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.time)===hours)));
    $('phase').textContent=busy?'Inspecting':stage==='found'?'Target recovered':stage==='updated'?'Belief updated':hours===0?'Direct observation':hours<48?'Time elapsing':'Current query';
    $('evidence-age').textContent=stage==='found'?'Just observed':`${hours}h since observation`;
    $('evidence-source').textContent=stage==='found'?'Positive detection':stage==='updated'?'Negative inspection':'Historical observation';
    $('evidence-location').textContent=stage==='found'?s.recovered:stage==='updated'?s.locations[1]:s.locations[0];
    $('context').textContent=stage==='found'?`New observation · ${s.recovered} · Day 3, 09:15`: `Last seen: ${s.locations[0]} · Day 1, 09:00 · ${hours}h ago`;
    $('reason-title').textContent=stage==='updated'?'VISIBILITY-AWARE UPDATE':stage==='found'?'NEW EVIDENCE, NEW MEMORY':'WHY THIS PREDICTION?';
    $('explanation').textContent=stage==='updated'?`No ${s.object} detected at the ${s.locations[1].toLowerCase()}. With 95% inspection visibility, its probability falls to ${pc[1]}%. Other locations remain plausible.`:stage==='found'?`A positive observation at the ${s.recovered.toLowerCase()} raises its belief to ${pc[2]}%. The observation is recorded as a new memory version.`:hours===0?`The ${s.object} is directly observed on the ${s.locations[0].toLowerCase()}. A fresh observation anchors the belief at 96%.`:hours<48?`${s.activity} No new observation is available; uncertainty grows as the original memory ages.`:s.pattern;
    $('action').disabled=busy;
    $('action').innerHTML=busy?'Navigating & inspecting… <span>···</span>':stage==='found'?'Replay this scenario <span>↺</span>':stage==='updated'?`Inspect ${s.recovered.toLowerCase()} <span>→</span>`:hours<48?'Advance to current query <span>→</span>':`Inspect ${s.locations[1].toLowerCase()} <span>→</span>`;
    $('action-hint').textContent=stage==='updated'?'Inspect another plausible location to resolve uncertainty.':stage==='found'?'Evidence closes the loop. Memory stays current.':'Follow the belief. Gather new evidence.';
    $('object').setAttribute('transform',`translate(${(mapPoint(s.anchors[stage==='found'?2:0])).join(' ')})`);
    $('object').style.opacity=hours===0||stage==='found'?'1':'0';
    const active=busy?1:stage==='found'?3:stage==='updated'?2:0;
    root.querySelectorAll('[data-step]').forEach(el=>{el.dataset.active=String(Number(el.dataset.step)<=active);if(Number(el.dataset.step)===active)el.setAttribute('aria-current','step');else el.removeAttribute('aria-current');});
    $('status').textContent=busy?`Moving to the ${stage==='updated'?s.recovered.toLowerCase():s.locations[1].toLowerCase()} to gather evidence…`:stage==='found'?`Target found at the ${s.recovered.toLowerCase()}. The agent’s memory now contains a fresh observation.`:stage==='updated'?`${s.locations[1]} checked · target not detected · posterior renormalized across all candidates.`:hours===0?`Day 1 · ${s.object} observed on the ${s.locations[0].toLowerCase()}.`:`${hours} hours without direct observation. ${hours===48?'Inspect the leading hypothesis to see how evidence changes the belief.':'Move through time to see the belief evolve.'}`;
    $('memory-name').textContent=`${s.object} · object_${s.id}`;
    $('memory-content').innerHTML=`<div><h4>IDENTITY</h4><p>object_${s.id}<br>${s.appearance}<br>First observed: Day 1, 09:00</p></div><div><h4>OBSERVATION HISTORY</h4><p>Day 1 · ${s.locations[0]} · confidence 0.96<br>${hours>0?`Following ${hours}h · no direct observation`:'Fresh observation'}${stage==='updated'||stage==='found'?`<br>Day 3 · ${s.locations[1]} · negative evidence`:''}${stage==='found'?`<br>Day 3, 09:15 · ${s.recovered} · detected`:''}</p></div><div><h4>RELATIONS & PROVENANCE</h4><p>${stage==='found'?`Located at ${s.recovered}`:s.relation}<br>Scene: HSSD ${s.sceneId}<br>Beliefs and target: illustrative scenario<br>${stage==='updated'?'P(s | no detection) ∝ P(no detection | s) P(s)':'Persistent identity · versioned observations'}</p></div>`;
  }
  function paintBelief(values) {
    shown=values;const rounded=percentages(values),leader=values.indexOf(Math.max(...values));
    for(let i=0;i<4;i++) {
      $(`value-${i}`).textContent=`${rounded[i]}%`;$(`fill-${i}`).style.width=`${values[i]*100}%`;$(`bar-${i}`).dataset.leader=String(i===leader);
      if(i<3){$(`prob-${i}`).textContent=`${rounded[i]}%`;$(`halo-${i}`).style.opacity=.12+values[i]*.88;$(`halo-${i}`).setAttribute('rx',40+values[i]*65);$(`halo-${i}`).setAttribute('ry',22+values[i]*34);}
    }
  }
  function animateBelief(target) {
    cancelAnimationFrame(beliefFrame);
    if(reduced.matches){paintBelief(target);return;}
    const from=[...shown],start=performance.now();
    const tick=now=>{const t=Math.min(1,(now-start)/700),ease=1-Math.pow(1-t,4);paintBelief(from.map((v,i)=>v+(target[i]-v)*ease));if(t<1)beliefFrame=requestAnimationFrame(tick);};
    beliefFrame=requestAnimationFrame(tick);
  }
  function stopStory(){storyRun++;story=false;$('play').textContent='▶ Start demonstration';$('play').setAttribute('aria-pressed','false');}
  function reset(time=48, keepStory=false) {
    if(!keepStory)stopStory();
    generation++;busy=false;stage='predict';hours=time;
    robotHeading=-Math.PI/2;poseRobot();
    $('route').style.opacity=0;$('route-shadow').style.opacity=0;$('agent').setAttribute('transform',`translate(${mapPoint(scenes[key].start).join(' ')})`);
    $('scan').querySelector('ellipse').classList.remove('scanning');render();
  }
  function travel(path,token) {
    return new Promise(resolve=>{
      const route=$('route');route.setAttribute('d',path);route.style.opacity=1;
      const shadow=$('route-shadow');shadow.setAttribute('d',path);shadow.style.opacity='.75';
      const length=route.getTotalLength();shadow.style.strokeDasharray=length;shadow.style.strokeDashoffset=length;route.style.strokeDasharray=length;route.style.strokeDashoffset=length;
      const start=performance.now(),duration=reduced.matches?0:2600;
      robotState('Walking · route tracking','walking');
      let last= start;
      function frame(now){
        if(token!==generation){resolve(false);return;}
        const t=duration?Math.min(1,(now-start)/duration):1;
        const ease=t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
        const point=route.getPointAtLength(length*ease);
        const before=route.getPointAtLength(Math.max(0,length*ease-.6));
        const after=route.getPointAtLength(Math.min(length,length*ease+.6));
        const dx=after.x-before.x,dy=after.y-before.y;
        const heading=Math.atan2((dy/.45-dx/.82)/2,(dy/.45+dx/.82)/2);
        const delta=Math.atan2(Math.sin(heading-robotHeading),Math.cos(heading-robotHeading));
        robotHeading+=delta*(reduced.matches?1:1-Math.exp(-Math.min(60,now-last)/85));last=now;
        poseRobot(length*ease*.12,t>0&&t<1&&!reduced.matches);
        $('agent').setAttribute('transform',`translate(${point.x} ${point.y})`);route.style.strokeDashoffset=length*(1-ease);shadow.style.strokeDashoffset=length*(1-ease);
        if(t<1)requestAnimationFrame(frame);else resolve(true);
      }
      requestAnimationFrame(frame);
    });
  }
  async function inspect() {
    if(busy)return;
    if(stage==='found'){reset();return;}
    if(hours<48){reset();return;}
    const token=++generation,recovering=stage==='updated';busy=true;render();
    // Keep the measured NavMesh waypoints; rounded line joins don't cut corners.
    const path=scenes[key].routes[recovering?1:0].map((v,i)=>`${i?'L':'M'}${mapPoint(v).join(' ')}`).join(' ');
    if(!await travel(path,token))return;
    robotState('Stationary · visual scan','scanning');
    $('phase').textContent='Scanning';$('action').innerHTML='Acquiring visual evidence… <span>◌</span>';
    $('status').textContent=`Inspecting the ${recovering?scenes[key].recovered.toLowerCase():scenes[key].locations[1].toLowerCase()} · checking target visibility.`;
    $('scan').setAttribute('transform',`translate(${mapPoint(scenes[key].anchors[recovering?2:1]).join(' ')})`);
    $('scan').querySelector('ellipse').classList.add('scanning');
    await delay(1100);if(token!==generation)return;
    $('scan').querySelector('ellipse').classList.remove('scanning');stage=recovering?'found':'updated';busy=false;render();
  }
  root.querySelectorAll('[data-scene]').forEach(b=>b.addEventListener('click',()=>{key=b.dataset.scene;stage='predict';hours=48;buildScene();reset();}));
  root.querySelectorAll('[data-time]').forEach(b=>b.addEventListener('click',()=>reset(Number(b.dataset.time))));
  $('time').addEventListener('input',e=>reset(Number(e.target.value)));
  $('reset').addEventListener('click',()=>reset());
  $('inspect-memory').addEventListener('click',()=>{$('memory').open=true;$('memory').querySelector('summary').focus();});
  $('action').addEventListener('click',()=>{stopStory();inspect();});
  $('play').addEventListener('click',async()=>{
    if(story){reset(hours);return;}
    reset(0);const run=++storyRun;story=true;$('play').textContent='Ⅱ Stop demonstration';$('play').setAttribute('aria-pressed','true');
    let token=generation;await delay(1000);if(!story||run!==storyRun||token!==generation)return;
    // Advance historical time at a steady cadence; manual controls always cancel the story.
    for(let h=1;h<=48;h++){if(!story||run!==storyRun||token!==generation)return;hours=h;render();await delay(45);}
    await delay(600);if(!story||run!==storyRun||token!==generation)return;
    await inspect();if(!story||run!==storyRun)return;token=generation;
    await delay(1500);if(!story||run!==storyRun||token!==generation)return;
    await inspect();if(story&&run===storyRun)stopStory();
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&(story||busy))reset(hours);});
  buildScene();reset();root.hidden=false;
})();
