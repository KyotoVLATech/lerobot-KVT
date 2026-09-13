import {forward,sample} from './core.mjs';
const $=id=>document.getElementById(id), rad=Math.PI/180;
let source=null,edits=[],edited=[],time=0,playing=false,last=0,yaw=.85,elevation=.42,zoom=400;
const status=(s)=>$('status').textContent=s;
async function api(url,options){const r=await fetch(url,options),d=await r.json();if(!r.ok)throw Error(d.error);return d;}
function pending(){
  const e={joint:Number($('joint').value),center:Number($('center').value),sigma:Number($('sigma').value),amplitude:Number($('amplitude').value)*rad};
  if(['center','sigma','amplitude'].some(k=>$(k).value.trim()==='')||!Object.values(e).every(Number.isFinite)||e.center<0||e.center>source.duration||e.sigma<=0||Math.abs(e.amplitude)>Math.PI)throw Error('時刻・幅・角度を範囲内で入力してください');
  return e;
}
function operations(){const p=pending();return [...edits,...(p.amplitude?[p]:[])];}
function refresh(){
  if(!source)return;
  try{const ops=operations();edited=source.actions.map((q,i)=>{const a=q.slice();for(const e of ops)a[e.joint]+=e.amplitude*Math.exp(-.5*((i/source.fps-e.center)/e.sigma)**2);return a;});$('save').disabled=!ops.length;$('apply').disabled=false;status('プレビュー更新済み');draw();}
  catch(e){playing=false;$('save').disabled=true;$('apply').disabled=true;status(e.message);}
}
async function load(){
  playing=false;$('load').disabled=true;$('save').disabled=true;
  try{const next=await api(`/api/episode?dataset=${encodeURIComponent($('dataset').value)}&episode=${Number($('episode').value)}`);source=next;edits=[];time=0;$('center').value=0;$('center').max=source.duration;$('amplitude').value=0;$('scrub').max=source.duration;$('name').value=source.name+'-edited';list();refresh();}
  catch(e){status(e.message);}finally{$('load').disabled=false;}
}
function list(){$('edits').textContent=edits.map((e,i)=>`${i+1}. joint_${e.joint} / ${e.center.toFixed(2)}s / σ${e.sigma}s / ${(e.amplitude/rad).toFixed(1)}°`).join('\n');$('edits').style.whiteSpace='pre-line';}
$('load').onclick=load;
for(const id of ['joint','center','sigma','amplitude'])$(id).oninput=refresh;
$('use-time').onclick=()=>{$('center').value=time.toFixed(3);refresh();};
$('apply').onclick=()=>{try{const p=pending();if(p.amplitude)edits.push(p);$('amplitude').value=0;list();refresh();}catch(e){status(e.message);}};
$('undo').onclick=()=>{edits.pop();$('amplitude').value=0;list();refresh();};
$('save').onclick=async()=>{
  if(!source)return;
  try{const request={dataset:source.name,episode:source.episode,name:$('name').value,edits:operations(),update_state:$('update-state').checked};$('save').disabled=true;status('データセットを複製・保存しています…');const r=await api('/api/save-dataset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});status(`保存しました: ${r.path} (${r.frames} frames)`);await catalog(r.name);}
  catch(e){status(e.message);}finally{$('save').disabled=false;}
};
$('scrub').oninput=()=>{playing=false;time=Number($('scrub').value);draw();};
$('play').onclick=()=>{if(source){playing=!playing;if(time>=source.duration)time=0;}};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>{[yaw,elevation]=({'3d':[.85,.42],front:[0,0],top:[0,Math.PI/2],side:[Math.PI/2,0]})[b.dataset.view];draw();};
let drag=null;const canvas=$('scene');canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw-=(e.clientX-drag[0])*.007;elevation=Math.max(-1.5,Math.min(1.57,elevation+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];draw();}};canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(100,Math.min(1200,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
function context(id){const c=$(id),r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const ctx=c.getContext('2d');ctx.scale(d,d);return [ctx,r.width,r.height];}
function line(ctx,points,color,width=2){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function draw(){
  const [ctx,w,h]=context('scene');if(!source||!edited.length)return;
  const project=([x,y,z])=>[w/2+zoom*(-Math.sin(yaw)*x+Math.cos(yaw)*y),h/2-zoom*(-Math.sin(elevation)*(Math.cos(yaw)*x+Math.sin(yaw)*y)+Math.cos(elevation)*(z-.3))];
  for(let i=-10;i<=10;i++){line(ctx,[project([i/10,-1,0]),project([i/10,1,0])],'#e7edf4',1);line(ctx,[project([-1,i/10,0]),project([1,i/10,0])],'#e7edf4',1);}
  const poses=[sample(source.actions,source.fps,time),sample(edited,source.fps,time)];
  for(let v=0;v<2;v++)for(let side=0;side<2;side++){
    const color=v?'#2563eb':'#a1a9b6',arm=forward(poses[v],side);
    const frames=v?edited:source.actions,step=Math.max(1,Math.floor(frames.length/500));
    line(ctx,frames.filter((_,i)=>i%step===0).map(q=>project(forward(q,side).tip)),v?'#93b5ed':'#d1d5db',1);
    line(ctx,arm.points.map(project),color,v?4:2);line(ctx,arm.fingers.map(project),color,3);
    for(const point of arm.points){const p=project(point);ctx.beginPath();ctx.arc(...p,4,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
    const b=project(arm.points[0]);ctx.fillText(side?'R':'L',b[0]+10,b[1]);
  }
  const distances=poses.map(q=>{const a=forward(q);return Math.hypot(...a.tip.map((v,i)=>v-a.points[0][i]));});
  $('distance').textContent=`左腕 ベース→手先：元 ${distances[0].toFixed(3)} m → 編集後 ${distances[1].toFixed(3)} m（差 ${((distances[1]-distances[0])*1000).toFixed(1)} mm）`;
  $('time').textContent=`${time.toFixed(3)} / ${source.duration.toFixed(3)} s`;$('scrub').value=time;$('play').textContent=playing?'Ⅱ 一時停止':'▶ 再生';
  const j=Number($('joint').value);$('angles').textContent=`joint_${j}：${(poses[0][j]/rad).toFixed(2)}° → ${(poses[1][j]/rad).toFixed(2)}°`;
  const [c,cw,ch]=context('chart');let low=Infinity,high=-Infinity;for(const rows of [source.actions,edited])for(const q of rows){low=Math.min(low,q[j]/rad);high=Math.max(high,q[j]/rad);}low-=2;high+=2;
  const x=i=>40+i/(edited.length-1)*(cw-50),y=a=>ch-20-(a-low)/(high-low)*(ch-40);
  c.fillStyle='#63748b';c.fillText(`${high.toFixed(1)}°`,0,14);c.fillText(`${low.toFixed(1)}°`,0,ch-15);c.fillText('0 s',40,ch-2);c.fillText(`${source.duration.toFixed(1)} s`,cw-55,ch-2);
  [source.actions,edited].forEach((rows,i)=>line(c,rows.map((q,n)=>[x(n),y(q[j]/rad)]),i?'#2563eb':'#a1a9b6'));
  line(c,[[x(time*source.fps),0],[x(time*source.fps),ch-20]],'#d97706',1);
}
new ResizeObserver(draw).observe(canvas);
function tick(now){if(playing&&source){time=Math.min(source.duration,time+Math.min((now-last)/1000,.1));if(time>=source.duration)playing=false;draw();}last=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
async function catalog(selected='iloha-common'){const d=await api('/api/catalog');$('dataset').replaceChildren(...d.datasets.filter(x=>x.available).map(x=>new Option(x.name,x.name)));if([...$('dataset').options].some(o=>o.value===selected))$('dataset').value=selected;}
$('save').disabled=true;catalog().then(load).catch(e=>status(e.message));
