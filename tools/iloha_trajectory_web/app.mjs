import {clamp, forward, sample, defaultViewSettings, defaultReplaySettings, createTrajectorySettings, validateSpacing, projectView, projectClipName, validateSource, makeDemo} from './core.mjs';

const $=id=>document.getElementById(id);
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,d=2)=>Number(x).toFixed(d);
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(3).padStart(6,'0')}`;
let clips=[],viewSettings=defaultViewSettings(),result=null,trajectory=null;
let worker=null,revision=0,timer=null,playing=false,currentTime=0,lastFrame=0,dirty=true,needsFit=true;
let catalogEntries=[];
const canvas=$('scene'),ctx=canvas.getContext('2d'),timeline=$('timeline-canvas'),tc=timeline.getContext('2d');
let width=800,height=500,tw=700,th=89;
let camera={yaw:0.85,elevation:0.42,distance:2.6,target:[0,0,0.25],pan:[0,0]};
let projectedIdeal=null;

function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
async function api(path){const response=await fetch(path);const data=await response.json();if(!response.ok)throw Error(data.error||'読み込みに失敗しました');return data;}
function editSettings(){return {mode:$('transition').value,blend:Number($('blend').value),fps:60};}
function setPlaying(value){playing=value;$('play').textContent=playing?'Ⅱ 一時停止':'▶ 再生';lastFrame=performance.now();}
function enableTransport(enabled){for(const id of ['play','restart','boundary','scrub'])$(id).disabled=!enabled;$('export').disabled=!clips.length;}
function changed(){setPlaying(false);enableTransport(false);worker?.terminate();revision++;clearTimeout(timer);status('変更を反映しています…');timer=setTimeout(compute,250);}
function compute(){
  clearTimeout(timer);worker?.terminate();setPlaying(false);enableTransport(false);
  if(!clips.length){result=null;trajectory=null;$('empty').hidden=false;$('empty').querySelector('h2').textContent='軌道を追加してください';status('データセットを選択して追加してください');dirty=true;return;}
  try{validateSpacing(viewSettings.spacing);}catch(error){status(error.message,true);return;}
  const id=++revision,started=performance.now();
  worker=new Worker('/worker.mjs',{type:'module'});
  status('理想軌道を作成しています…');
  worker.onmessage=({data})=>{
    if(data.id!==revision)return;
    if(data.error){status(data.error,true);worker?.terminate();return;}
    result=data.result;trajectory=data.trajectory;
    window.ilohaStudio={get result(){return result;},get trajectory(){return trajectory;},get clips(){return clips;},get view(){return viewSettings;},get currentTime(){return currentTime;}};
    currentTime=clamp(currentTime,0,result.duration);$('scrub').max=result.duration;
    $('empty').hidden=true;enableTransport(true);$('time-total').textContent=clock(result.duration);
    $('data-label').textContent=clips.some(c=>c.source.synthetic)?'DEMO · 人工データ':'DATASET · '+clips.map(c=>c.source.name).join(' → ');
    document.querySelectorAll('[data-duration]').forEach(el=>{
      const i=Number(el.dataset.duration),s=result.segments[i];if(s)el.textContent=`再生 ${num(s.end-s.start,2)} s`;
    });
    projectedIdeal=null;if(needsFit){fit();needsFit=false;}dirty=true;
    status(`理想軌道 · ${num(result.duration,2)} s · ${num((performance.now()-started)/1000,2)} sで作成`);
    worker?.terminate();
  };
  worker.onerror=event=>{status(`軌道の作成に失敗しました: ${event.message}`,true);worker?.terminate();};
  worker.postMessage({id,clips,spacing:viewSettings.spacing,edit:editSettings()});
}

const replayFields=[['base_speed','ベース速度','×',0.05,0.1],['max_speedup','追加倍率上限','×',1,0.1],['gripper_margin','グリッパー前後の余白','s',0,0.1],['speedup_distance','追加倍率までの範囲','s',0.01,0.1],['gripper_threshold','グリッパー動作の判定閾値','',0,0.0001]];
function renderClips(){
  $('clips').innerHTML=clips.map((c,i)=>{
    const d=(c.source.actions.length-1)/c.source.fps;
    return `<article class="clip ${c.source.synthetic?'synthetic':''}" data-clip="${i}"><div class="clip-title"><span class="clip-number">${String(i+1).padStart(2,'0')}</span><span class="clip-name">${escapeHTML(c.source.name)}</span><button class="clip-remove" data-move="${i}" title="前へ移動" ${i===0?'disabled':''}>↑</button><button class="clip-remove" data-remove="${i}" title="クリップを削除">×</button></div><div class="clip-body"><p class="clip-meta">EP ${c.source.episode??0} · ${c.source.actions.length} frames · ${c.source.fps} FPS · ${num(d,2)} s${c.source.synthetic?' · 人工データ':''}</p><div class="two-col"><label>開始 <span class="unit">元データ s</span><input type="number" min="0" max="${d}" step="${1/c.source.fps}" value="${c.start}" data-trim="start" aria-label="${escapeHTML(c.source.name)} 開始時刻"></label><label>終了 <span class="unit">元データ s</span><input type="number" min="0" max="${d}" step="${1/c.source.fps}" value="${c.end}" data-trim="end" aria-label="${escapeHTML(c.source.name)} 終了時刻"></label></div><div class="trim-sliders"><input type="range" min="0" max="${d}" step="${1/c.source.fps}" value="${c.start}" data-trim="start" aria-label="${escapeHTML(c.source.name)} 開始トリム"><input type="range" min="0" max="${d}" step="${1/c.source.fps}" value="${c.end}" data-trim="end" aria-label="${escapeHTML(c.source.name)} 終了トリム"></div><details open><summary>再生速度 <span class="unit" data-duration="${i}"></span></summary><div class="replay-fields">${replayFields.map(([key,label,unit,min,step])=>`<label>${label}<span class="unit">${unit}</span><input type="number" min="${min}" step="${step}" value="${c.replay[key]}" data-replay="${key}" aria-label="${escapeHTML(c.source.name)} ${label}"></label>`).join('')}</div><p class="unit-speed">余白・範囲は速度変更前の記録時間です。</p></details></div></article>`;
  }).join('');
}
$('clips').addEventListener('input',event=>{
  const target=event.target,article=target.closest('[data-clip]');if(!article)return;
  const c=clips[Number(article.dataset.clip)];
  if(target.dataset.trim){
    const key=target.dataset.trim;
    let value=Number(target.value);
    if(target.type==='range') value=key==='start'?Math.min(value,c.end-1/c.source.fps):Math.max(value,c.start+1/c.source.fps);
    c[key]=value;
    article.querySelectorAll(`[data-trim="${key}"]`).forEach(el=>{if(el!==target||target.type==='range')el.value=value;});
    changed();
  }else if(target.dataset.replay){c.replay[target.dataset.replay]=Number(target.value);changed();}
});
$('clips').addEventListener('click',event=>{
  const b=event.target.closest('button');if(!b)return;
  if(b.dataset.remove!==undefined)clips.splice(Number(b.dataset.remove),1);
  else if(b.dataset.move!==undefined){const i=Number(b.dataset.move);[clips[i-1],clips[i]]=[clips[i],clips[i-1]];}
  else return;
  renderClips();changed();
});

async function fetchSource(name,episode=0){const source=await api(`/api/episode?dataset=${encodeURIComponent(name)}&episode=${episode}`);validateSource(source);return source;}
function clipFor(source){return {source,start:0,end:(source.actions.length-1)/source.fps,replay:defaultReplaySettings()};}
async function loadCatalog(initial=false){
  try{
    const data=await api('/api/catalog');catalogEntries=data.datasets;
    $('dataset').innerHTML=catalogEntries.map(x=>`<option value="${escapeHTML(x.name)}" ${x.available?'':'disabled'}>${escapeHTML(x.name)}${x.available?'':'（データなし）'}</option>`).join('');
    const messages=catalogEntries.filter(x=>!x.available).map(x=>`${x.name}: ${x.error}`);
    $('catalog-notice').hidden=!messages.length;$('catalog-notice').textContent=messages.join('\n');
    if(initial){
      const defaults=['iloha-best','iloha-common'].filter(n=>catalogEntries.some(x=>x.name===n&&x.available));
      const selected=defaults.length?defaults:catalogEntries.filter(x=>x.available).slice(0,1).map(x=>x.name);
      clips=(await Promise.all(selected.map(n=>fetchSource(n)))).map(clipFor);
      renderClips();compute();
    }else status('データセット一覧を更新しました');
  }catch(error){status(error.message,true);$('empty').querySelector('h2').textContent='データを読み込めませんでした';}
}
$('reload').onclick=()=>loadCatalog(clips.length===0);
$('add').onclick=async()=>{try{$('add').disabled=true;const source=await fetchSource($('dataset').value,Number($('episode').value));clips.push(clipFor(source));renderClips();needsFit=true;changed();}catch(error){status(error.message,true);}finally{$('add').disabled=false;}};
$('demo').onclick=()=>{clips=[clipFor(makeDemo()),clipFor(makeDemo('動作確認用デモ B',1.2))];renderClips();needsFit=true;changed();};

function renderViewSettings(){$('spacing').value=viewSettings.spacing;}
$('spacing').oninput=()=>{viewSettings.spacing=Number($('spacing').value);needsFit=true;changed();};
$('normal-speed').onclick=()=>{for(const c of clips){c.replay.base_speed=1;c.replay.max_speedup=1;}renderClips();changed();};
$('compute').onclick=compute;
function transitionHelp(){
  const descriptions={crossfade:'末尾と先頭を重ね、関節角度を滑らかな重みで混合します。重ねた時間だけ短くなります。',linear:'使用区間の末尾と次の先頭を、関節角度の直線補間で接続します。',smooth:'両端の関節速度に合わせた補間で接続します。位置は端点間からはみ出す場合があります。',cut:'選択区間を続けて再生します。境界の角度差はそのまま残ります。'};
  $('transition-help').textContent=descriptions[$('transition').value];$('blend').disabled=$('transition').value==='cut';
}
$('transition').onchange=()=>{transitionHelp();changed();};$('blend').oninput=changed;

function download(name,data){const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('save-project').onclick=()=>download('iloha-trajectory-project.json',{version:2,view:viewSettings,edit:editSettings(),clips:clips.map(c=>({name:c.source.name,episode:c.source.episode,start:c.start,end:c.end,replay:c.replay,...(c.source.synthetic?{source:c.source}:{})}))});
$('load-project').onclick=()=>$('project-file').click();
$('project-file').onchange=async()=>{
  try{
    const file=$('project-file').files[0];if(!file)return;if(file.size>20000000)throw Error('設定ファイルが大きすぎます');
    const data=JSON.parse(await file.text());if(!Array.isArray(data.clips)||!data.clips.length||data.clips.length>20)throw Error('設定ファイルの形式が違います');
    const nextView=projectView(data);
    const next=await Promise.all(data.clips.map(async c=>{const source=c.source||await fetchSource(projectClipName(c),c.episode);validateSource(source);return {source,start:c.start,end:c.end,replay:{...defaultReplaySettings(),...c.replay}};}));
    clips=next;viewSettings=nextView;$('transition').value=data.edit?.mode??'crossfade';$('blend').value=data.edit?.blend??1;renderClips();renderViewSettings();transitionHelp();needsFit=true;changed();
  }catch(error){status(error.message,true);}finally{$('project-file').value='';}
};
function trajectorySettings(){return createTrajectorySettings(clips,editSettings(),trajectory);}
function exportSummary(){
  try{
    const t=trajectorySettings().trajectory;
    $('export-summary').textContent=`${t.frames.toLocaleString()} フレーム · ${t.fps} FPS · ${num(t.duration,2)} 秒（速度変更を含む、画面と同じ軌道）`;
    $('export-settings').disabled=false;
  }catch(error){$('export-summary').textContent=error.message;$('export-settings').disabled=true;}
}
$('export').onclick=()=>{$('export-result').textContent='';exportSummary();$('export-dialog').showModal();};
$('close-export').onclick=()=>$('export-dialog').close();
$('export-settings').onclick=()=>{
  try{download('trajectory_settings.json',trajectorySettings());$('export-result').textContent='trajectory_settings.json を保存しました。リポジトリ直下に置いて上のコマンドを実行してください。';}
  catch(error){$('export-result').textContent=error.message;}
};

function seek(t){if(!result)return;currentTime=clamp(t,0,result.duration);dirty=true;}
$('scrub').oninput=()=>{setPlaying(false);seek(Number($('scrub').value));};
$('play').onclick=()=>{if(!result)return;if(currentTime>=result.duration)seek(0);setPlaying(!playing);};
$('restart').onclick=()=>{setPlaying(false);seek(0);};
$('boundary').onclick=()=>{if(!result)return;setPlaying(false);const b=result.boundaries.find(b=>b.start>currentTime+0.05)||result.boundaries[0];if(b)seek(Math.max(0,b.start-0.5));};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName)&&!$('play').disabled){e.preventDefault();$('play').click();}});
for(const id of ['show-left','show-right','show-ideal-arm'])$(id).onchange=()=>{dirty=true;};

function resize(){
  const dpr=Math.min(devicePixelRatio||1,2),r=canvas.getBoundingClientRect(),tr=timeline.getBoundingClientRect();
  width=r.width;height=r.height;tw=tr.width;th=tr.height;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
  timeline.width=Math.round(tw*dpr);timeline.height=Math.round(th*dpr);tc.setTransform(dpr,0,0,dpr,0,0);
  projectedIdeal=null;dirty=true;
}
new ResizeObserver(resize).observe($('viewport'));new ResizeObserver(resize).observe($('timeline-track'));
function fit(){
  if(!result)return;
  const min=[-0.05,-result.spacing/2,0],max=[0.05,result.spacing/2,0.1];
  for(const path of result.idealPaths)for(const p of path)for(let j=0;j<3;j++){min[j]=Math.min(min[j],p[j]);max[j]=Math.max(max[j],p[j]);}
  camera.target=min.map((v,j)=>(v+max[j])/2);camera.distance=Math.max(1.5,Math.hypot(...min.map((v,j)=>max[j]-v))*1.35);camera.pan=[0,0];projectedIdeal=null;dirty=true;
}
$('fit').onclick=fit;
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
  const views={perspective:[0.85,0.42],front:[0,0],top:[0,Math.PI/2-0.001],side:[Math.PI/2,0]};
  [camera.yaw,camera.elevation]=views[button.dataset.view];projectedIdeal=null;dirty=true;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b===button));
});
let drag=null;
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===1};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.pan){camera.pan[0]+=dx;camera.pan[1]+=dy;}else{camera.yaw-=dx*0.007;camera.elevation=clamp(camera.elevation+dy*0.007,-1.5,1.56);}
  projectedIdeal=null;dirty=true;
};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();camera.distance=clamp(camera.distance*Math.exp(e.deltaY*0.001),0.35,15);projectedIdeal=null;dirty=true;},{passive:false});

function projection(){
  const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),ce=Math.cos(camera.elevation),se=Math.sin(camera.elevation);
  const right=[-sy,cy,0],up=[-se*cy,-se*sy,ce],normal=[ce*cy,ce*sy,se];
  return p=>{const d=p.map((v,i)=>v-camera.target[i]);const depth=camera.distance-d.reduce((s,v,i)=>s+v*normal[i],0),scale=Math.min(width,height)*1.1/Math.max(0.05,depth);
    return [width/2+camera.pan[0]+scale*d.reduce((s,v,i)=>s+v*right[i],0),height/2+camera.pan[1]-scale*d.reduce((s,v,i)=>s+v*up[i],0),depth,scale];};
}
function line(points,color,lineWidth=1,dashed=false){if(points.length<2)return;ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dashed?[4,4]:[]);ctx.stroke();ctx.setLineDash([]);}
function currentState(){return {ideal:sample(result.ideal,result.fps,currentTime)};}
function drawScene(){
  ctx.clearRect(0,0,width,height);const project=projection();
  for(let k=-10;k<=10;k++) {const p=k/10;line([project([p,-1,0]),project([p,1,0])],k===0?'#d0dbe8':'#e7edf4',k===0?1.1:0.6);line([project([-1,p,0]),project([1,p,0])],k===0?'#d0dbe8':'#e7edf4',k===0?1.1:0.6);}
  if(!result)return;
  const state=currentState();
  if(!projectedIdeal)projectedIdeal=result.idealPaths.map(path=>{const p=new Path2D();path.forEach((v,i)=>{const x=project(v);if(i===0)p.moveTo(x[0],x[1]);else p.lineTo(x[0],x[1]);});return p;});
  const segments=[],dots=[],color='#63748b';
  for(let side=0;side<2;side++){
    if(!$(side?'show-right':'show-left').checked)continue;
    ctx.strokeStyle='#a1a9b6';ctx.globalAlpha=.65;ctx.lineWidth=1.3;ctx.stroke(projectedIdeal[side]);ctx.globalAlpha=1;
    const arm=forward(state.ideal,side,result.spacing);
    if($('show-ideal-arm').checked){
      const p=arm.points.map(project);
      for(let j=0;j<p.length-1;j++)segments.push({points:[p[j],p[j+1]],depth:(p[j][2]+p[j+1][2])/2});
      segments.push({points:arm.fingers.map(project),depth:p.at(-1)[2]});
      p.forEach(point=>dots.push(point));
    }
    const b=project(arm.points[0]);ctx.fillStyle='#8195b1';ctx.font='10px Segoe UI';ctx.fillText(side?'R':'L',b[0]+9,b[1]+15);
  }
  segments.sort((a,b)=>b.depth-a.depth).forEach(s=>line(s.points,color,3));
  dots.sort((a,b)=>b[2]-a[2]).forEach(point=>{ctx.beginPath();ctx.arc(point[0],point[1],4.5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();});
  const origin=project([-.7,-.7,0]);for(const [axis,label,axisColor] of [[[.15,0,0],'X','#bb8080'],[[0,.15,0],'Y','#83a896'],[[0,0,.15],'Z','#7c99c2']]){const end=project([-.7+axis[0],-.7+axis[1],axis[2]]);line([origin,end],axisColor,1.5);ctx.fillStyle=axisColor;ctx.font='9px Segoe UI';ctx.fillText(label,end[0]+4,end[1]-4);}
}
function drawTimeline(){
  tc.clearRect(0,0,tw,th);if(!result)return;
  const x=t=>8+(tw-16)*t/result.duration;
  tc.fillStyle='#f4f6f9';tc.fillRect(8,23,tw-16,42);
  tc.font='9px Segoe UI';tc.textBaseline='middle';
  const tick=result.duration>120?30:result.duration>30?10:5;
  for(let t=0;t<=result.duration;t+=tick){tc.fillStyle='#9eabbc';tc.fillText(`${Math.floor(t/60)}:${String(t%60).padStart(2,'0')}`,x(t),9);}
  result.segments.forEach((s,i)=>{
    tc.fillStyle=i%2?'#e5ece9':'#e4ebf6';tc.fillRect(x(s.start),24,Math.max(1,x(s.end)-x(s.start)),40);
    tc.save();tc.beginPath();tc.rect(x(s.start),24,x(s.end)-x(s.start),40);tc.clip();tc.fillStyle=i%2?'#638678':'#5e7ca6';tc.fillText(`${i+1}  ${s.name}`,x(s.start)+8,36);tc.restore();
  });
  result.boundaries.forEach(b=>{tc.fillStyle='#a5b4d966';tc.fillRect(x(b.start),23,Math.max(2,x(b.end)-x(b.start)),42);});
  // Shape overview: ideal shoulder angle, sampled in playback time.
  tc.beginPath();for(let px=8;px<tw-8;px++){const i=Math.min(result.ideal.length-1,Math.floor((px-8)/(tw-16)*(result.ideal.length-1))),y=54-8*Math.sin(result.ideal[i][1]);if(px===8)tc.moveTo(px,y);else tc.lineTo(px,y);}tc.strokeStyle='#8ca4c1';tc.lineWidth=1;tc.stroke();
  const sx=x(currentTime);tc.beginPath();tc.moveTo(sx,18);tc.lineTo(sx,73);tc.strokeStyle='#2563eb';tc.lineWidth=1.7;tc.stroke();tc.fillStyle='#2563eb';tc.beginPath();tc.moveTo(sx-4,16);tc.lineTo(sx+4,16);tc.lineTo(sx,22);tc.closePath();tc.fill();
  $('time-now').textContent=clock(currentTime);$('scrub').value=currentTime;
  const active=result.segments.filter(s=>s.start<=currentTime&&s.end>=currentTime).map(s=>s.name);
  const boundary=result.boundaries.find(b=>b.start<=currentTime&&b.end>=currentTime);
  $('active-clip').textContent=boundary?'接続区間 · '+active.join(' + '):active.join(' → ')||'再生終了';
}
function frame(now){
  if(playing&&result){const elapsed=Math.min((now-lastFrame)/1000,.2);currentTime+=elapsed*Number($('preview-speed').value);if(currentTime>=result.duration){if($('loop').checked)currentTime%=result.duration;else{currentTime=result.duration;setPlaying(false);}}dirty=true;}
  lastFrame=now;if(dirty){drawScene();drawTimeline();dirty=false;}requestAnimationFrame(frame);
}
renderViewSettings();transitionHelp();resize();requestAnimationFrame(frame);loadCatalog(true);
