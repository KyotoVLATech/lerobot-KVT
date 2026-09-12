// Pure numerical code shared by the browser worker and Node regression tests.
export const LENGTHS = [0.1, 0.305834, 0.2033, 0.0967, 0.07015, 0.03];
export const ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12];
const TAU = 2 * Math.PI;
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const turn = (v, axis, angle) => add(add(mul(v, Math.cos(angle)), mul(cross(axis, v), Math.sin(angle))), mul(axis, dot(axis, v)*(1-Math.cos(angle))));

export function validateSource(source) {
  if (!Number.isFinite(source.fps) || source.fps <= 0 || !Array.isArray(source.actions) || source.actions.length < 2)
    throw Error("2フレーム以上の軌道と正のFPSが必要です");
  if (source.coordinates !== "iloha") throw Error("関節座標はiloha形式である必要があります");
  for (const row of source.actions) if (row.length !== 14 || !row.every(Number.isFinite)) throw Error("関節値が不正です");
}

export function unwrap(actions) {
  const result = actions.map(q => q.slice());
  for (let i = 1; i < result.length; i++) for (const j of ARM_JOINTS)
    result[i][j] += TAU * Math.round((result[i-1][j] - result[i][j]) / TAU);
  return result;
}

export function sample(actions, fps, time) {
  const x = clamp(time * fps, 0, actions.length - 1), i = Math.floor(x);
  return mix(actions[i], actions[Math.min(i + 1, actions.length - 1)], x - i);
}

// Invert iloha_catch_server.py's UDP conversion, then ref_ik.cs's theta offsets.
// Coordinates in this module: x/y horizontal, z up (the analytical IK frame).
export function forward(q, side = 0, spacing = 0.59) {
  const o = side * 7, [L1,L2,L3,L4,L5,L6] = LENGTHS;
  const t0 = -(q[o] + (side === 0 ? -Math.PI/2 : Math.PI/2));
  const t1 = -(q[o+1] + Math.PI/2) + 0.19599;
  const t2 = q[o+2] + Math.PI - 0.19599;
  const t3 = q[o+3], t4 = q[o+4], t5 = -q[o+5];
  // ref_ik.cs: analytic +Y corresponds to Unity +X (right).
  const base = [0, side === 0 ? -spacing/2 : spacing/2, 0];
  const radial = [Math.cos(t0), Math.sin(t0), 0];
  const tangent = [-Math.sin(t0), Math.cos(t0), 0];
  const direction = t => add(mul(radial, Math.sin(t)), [0,0,Math.cos(t)]);
  const z23 = direction(t1+t2);
  const x23 = add(mul(radial, Math.cos(t1+t2)), [0,0,-Math.sin(t1+t2)]);
  const radialWrist = add(mul(x23, Math.cos(t3)), mul(tangent, Math.sin(t3)));
  const wristAxis = add(mul(x23, -Math.sin(t3)), mul(tangent, Math.cos(t3)));
  const tool = add(mul(radialWrist, Math.sin(t4)), mul(z23, Math.cos(t4)));
  const shoulder = add(base, [0,0,L1]);
  const elbow = add(shoulder, mul(direction(t1), L2));
  const roll = add(elbow, mul(z23, L3));
  const wrist = add(roll, mul(z23, L4));
  const toolRoll = add(wrist, mul(tool, L5));
  const tip = add(toolRoll, mul(tool, L6));
  const points = [base, shoulder, elbow, roll, wrist, toolRoll, tip];
  // Joint axes include signs w.r.t. Iloha hardware coordinates.
  const axes = [[0,0,-1], mul(tangent,-1), tangent, z23, wristAxis, mul(tool,-1)];
  const opening = 0.008 + 0.045 * (1-clamp(q[o+6],0,1));
  const fingerAxis = turn(wristAxis, tool, t5);
  return {points, axes, tip, fingers: [add(tip,mul(fingerAxis,opening)), sub(tip,mul(fingerAxis,opening))]};
}

export function defaultReplaySettings() {
  return {base_speed:1,max_speedup:2,gripper_margin:0.5,speedup_distance:1,gripper_threshold:1e-4};
}

export function replayIntervals(actions, fps, options={}) {
  const s={...defaultReplaySettings(),...options};
  for(const key of ["base_speed","speedup_distance"]) if(!Number.isFinite(s[key])||s[key]<=0) throw Error(`${key} は正の値が必要です`);
  if(!Number.isFinite(s.max_speedup)||s.max_speedup<1) throw Error("max_speedup は1以上が必要です");
  for(const key of ["gripper_margin","gripper_threshold"]) if(!Number.isFinite(s[key])||s[key]<0) throw Error(`${key} は0以上が必要です`);
  const moving=actions.slice(1).map((q,i)=>Math.abs(q[6]-actions[i][6])>s.gripper_threshold||Math.abs(q[13]-actions[i][13])>s.gripper_threshold);
  const distances=Array(moving.length).fill(Infinity);
  let prev=-Infinity;
  for(let i=0;i<moving.length;i++) {if(moving[i]) prev=i;distances[i]=i-prev;}
  let next=Infinity;
  for(let i=moving.length-1;i>=0;i--) {if(moving[i]) next=i;distances[i]=Math.min(distances[i],next-i);}
  return distances.map(d=>{
    const distance=Math.max(0,d-1)/fps, ramp=clamp((distance-s.gripper_margin)/s.speedup_distance,0,1);
    const speedup=1+(s.max_speedup-1)*ramp*ramp*(3-2*ramp);
    const interval=1/fps/s.base_speed/speedup;
    if(!Number.isFinite(interval)||interval<=0) throw Error("再生速度の設定範囲を超えています");
    return interval;
  });
}

function lowerSegment(times,t) {
  let lo=0,hi=times.length-1;
  while(lo+1<hi){const mid=(lo+hi)>>1;if(times[mid]<=t)lo=mid;else hi=mid;}
  return lo;
}

function prepareClip(clip, fps) {
  validateSource(clip.source);
  const maxTime=(clip.source.actions.length-1)/clip.source.fps;
  if (![clip.start,clip.end].every(Number.isFinite) || clip.start<0 || clip.end>maxTime+1e-6 || clip.end-clip.start<1/fps)
    throw Error(`${clip.source.name}: 使用区間は1フレーム以上にしてください`);
  const src=unwrap(clip.source.actions), intervals=replayIntervals(src,clip.source.fps,clip.replay);
  const times=[0]; for(const dt of intervals)times.push(times.at(-1)+dt);
  const warped=t=>{const x=t*clip.source.fps,i=Math.min(Math.floor(x),times.length-2);return times[i]+(x-i)*intervals[i];};
  const start=warped(clip.start),end=warped(clip.end),count=Math.max(1,Math.round((end-start)*fps));
  if(count>150000)throw Error("クリップが長すぎます。区間または速度を調整してください");
  return Array.from({length:count+1},(_,i)=>{
    const t=start+(end-start)*i/count,j=lowerSegment(times,t);
    return mix(src[j],src[j+1],clamp((t-times[j])/intervals[j],0,1));
  });
}

export function stitch(clips, {mode="crossfade", blend=1, fps=60}={}) {
  if (!clips.length) throw Error("データセットを読み込んでください");
  if (!Number.isFinite(fps)||fps<1||fps>240||!Number.isFinite(blend)||blend<0||blend>60) throw Error("接続時間またはFPSが不正です");
  if (!["cut","crossfade","linear","smooth"].includes(mode)) throw Error("接続方式が不正です");
  let actions=[], segments=[], boundaries=[];
  for (let c=0;c<clips.length;c++) {
    let next=prepareClip(clips[c],fps), startFrame=actions.length;
    if (c===0) { actions=next; startFrame=0; }
    else {
      const last=actions.at(-1);
      for(const j of ARM_JOINTS) {
        const shift=TAU*Math.round((last[j]-next[0][j])/TAU);
        for(const row of next) row[j]+=shift;
      }
      if(mode==="crossfade" && blend>0) {
        const n=Math.round(blend*fps);
        if(n<1 || n>=actions.length || n>=next.length || n/fps>segments.at(-1).end-segments.at(-1).start+1e-6)
          throw Error("クロスディゾルブ時間を各クリップの使用区間より短くしてください");
        startFrame=actions.length-1-n;
        for(let k=0;k<=n;k++) {
          const u=k/n, w=u*u*(3-2*u);
          actions[startFrame+k]=mix(actions[startFrame+k],next[k],w);
        }
        for(let k=n+1;k<next.length;k++)actions.push(next[k]);
        boundaries.push({start:startFrame/fps,end:(startFrame+n)/fps,mode});
      } else if((mode==="linear"||mode==="smooth") && blend>0) {
        const n=Math.max(1,Math.round(blend*fps)), before=actions.at(-2), after=next[1];
        const begin=actions.length-1;
        for(let k=1;k<n;k++) {
          const u=k/n;
          if(mode==="linear") actions.push(mix(last,next[0],u));
          else { // Cubic Hermite: match the trimmed clips' endpoint velocities.
            const h00=2*u**3-3*u*u+1,h10=u**3-2*u*u+u,h01=-2*u**3+3*u*u,h11=u**3-u*u;
            actions.push(last.map((v,j)=>h00*v+h10*n*(v-before[j])+h01*next[0][j]+h11*n*(after[j]-next[0][j])));
          }
        }
        startFrame=actions.length;
        for(const row of next)actions.push(row);
        boundaries.push({start:begin/fps,end:startFrame/fps,mode});
      } else {
        for(const row of next)actions.push(row);
        boundaries.push({start:(startFrame-1)/fps,end:startFrame/fps,mode:"cut"});
      }
    }
    segments.push({name:clips[c].source.name,sourceStart:clips[c].start,sourceEnd:clips[c].end,
      start:startFrame/fps,end:(actions.length-1)/fps,synthetic:!!clips[c].source.synthetic});
  }
  // Trigger values are not angles; interpolation must remain in their valid range.
  for(const row of actions) for(const j of [6,13]) row[j]=clamp(row[j],0,1);
  return {actions,fps,duration:(actions.length-1)/fps,segments,boundaries};
}

export function defaultViewSettings() { return {spacing:0.59}; }

export function validateSpacing(spacing) {
  if(!Number.isFinite(spacing)||spacing<0||spacing>3)throw Error("ベース間隔は0〜3 mです");
}

// Accepts the editor's project JSON and the exported trajectory_settings.json alike.
export function projectView(project) {
  const editor=project?.version,settings=project?.schema_version;
  if(![1,2].includes(editor)&&![2,3].includes(settings))throw Error("設定ファイルの形式が違います");
  // Import legacy edits/speeds and geometry, never the removed actuator model.
  const spacing=(editor===1?project.settings?.spacing:project.view?.spacing)??0.59;
  validateSpacing(spacing);
  return {spacing};
}

// trajectory_settings.json names its clips "dataset"; project files use "name".
export function projectClipName(clip) {
  const name=clip?.name??clip?.dataset;
  if(typeof name!=="string"||!name)throw Error("設定ファイルにデータセット名がありません");
  return name;
}

export function createPreview(trajectory, spacing=0.59) {
  validateSpacing(spacing);
  return {ideal:trajectory.actions, times:trajectory.actions.map((_,i)=>i/trajectory.fps),
    idealPaths:[0,1].map(side=>trajectory.actions.map(q=>forward(q,side,spacing).tip)),
    duration:trajectory.duration,fps:trajectory.fps,spacing,
    segments:trajectory.segments,boundaries:trajectory.boundaries};
}

// The settings document is the only export: iloha_catch_game2.py rebuilds these
// exact frames from it, so the robot moves like the preview without a dataset copy.
export function createTrajectorySettings(clips, edit, trajectory) {
  if(!clips.length||!trajectory)throw Error("データセットを読み込んでください");
  const synthetic=clips.filter(c=>c.source.synthetic).map(c=>c.source.name);
  if(synthetic.length)throw Error(`人工データは実機で再生できません: ${synthetic.join(", ")}`);
  const tasks=[...new Set(clips.map(c=>c.source.task).filter(Boolean))];
  return {schema_version:3,kind:"iloha_trajectory",speed_applied:true,
    clips:clips.map(c=>({dataset:c.source.name,episode:c.source.episode??0,start:c.start,end:c.end,replay:{...c.replay},synthetic:false})),
    edit:{mode:edit.mode,blend:edit.blend,fps:trajectory.fps,time_basis:"speed_adjusted_seconds"},
    trajectory:{fps:trajectory.fps,frames:trajectory.actions.length,duration:trajectory.duration,
      segments:trajectory.segments,boundaries:trajectory.boundaries},
    task:tasks.join(" → "),coordinates:"iloha",
    note:"Replay this with iloha_catch_game2.py --settings. Trimming, per-clip speed and transitions are all included."};
}

export function makeDemo(name="動作確認用デモ A", phase=0) {
  const fps=30, actions=Array.from({length:361},(_,i)=>{
    const t=i/fps, q=Array(14).fill(0);
    for(let s=0;s<2;s++) { const o=s*7, u=t*0.55+phase+s*0.4;
      q[o]=0.35*Math.sin(u);q[o+1]=-0.3+0.24*Math.sin(u*0.8);
      q[o+2]=0.45+0.35*Math.cos(u);q[o+3]=0.3*Math.sin(u*1.3);
      q[o+4]=0.4*Math.cos(u*0.9);q[o+5]=0.5*Math.sin(u);q[o+6]=(Math.sin(u)+1)/2;
    } return q;
  });
  return {name,fps,actions,coordinates:"iloha",duration:12,episode:0,synthetic:true};
}
