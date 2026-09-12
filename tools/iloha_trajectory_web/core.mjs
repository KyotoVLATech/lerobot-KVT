// Pure numerical code shared by the browser worker and Node regression tests.
export const LENGTHS = [0.1, 0.305834, 0.2033, 0.0967, 0.07015, 0.03];
export const ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12];
// Output-shaft specs. RobStride Kt is published per RMS phase ampere;
// the default CAN-Iq conversion assumes peak phase amperes (not verified on hardware).
// DYNAMIXEL coefficients are stall torque/current estimates at 12 V, not continuous ratings.
export const MOTOR_SPECS = Object.freeze([
  {model:'RobStride 03',ratedTorque:20,peakTorque:60,ktRms:2.36,gearRatio:9,noLoadRpm:195,voltage:48,source:'https://www.robstride.com/products/robStride03'},
  {model:'RobStride 06',ratedTorque:11,peakTorque:36,ktRms:1.10,gearRatio:9,noLoadRpm:480,voltage:48,source:'https://www.robstride.com/products/robStride06'},
  {model:'RobStride 00',ratedTorque:5,peakTorque:14,ktRms:1.48,gearRatio:10,noLoadRpm:315,voltage:48,source:'https://robstride.com/products/robStride00'},
  {model:'XM540-W270',ratedTorque:null,peakTorque:10.6,stallCurrent:4.4,gearRatio:272.5,noLoadRpm:30,voltage:12,source:'https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/'},
  {model:'XM540-W270',ratedTorque:null,peakTorque:10.6,stallCurrent:4.4,gearRatio:272.5,noLoadRpm:30,voltage:12,source:'https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/'},
  {model:'XM430-W350',ratedTorque:null,peakTorque:4.1,stallCurrent:2.3,gearRatio:353.5,noLoadRpm:46,voltage:12,source:'https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/'}
].map(Object.freeze));
export function manufacturerKt(joint, basis='peak') {
  const spec=MOTOR_SPECS[joint%6];
  return spec.ktRms ? spec.ktRms/(basis==='rms'?1:Math.SQRT2) : spec.peakTorque/spec.stallCurrent;
}
export function torqueCapacity(joint, limit) {
  return Math.min(limit.kt*limit.current,MOTOR_SPECS[joint%6].peakTorque);
}
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

// Distributed uniform rods, total mass per arm. Integrate distance squared along
// every downstream segment; retain only the diagonal of the inertia matrix.
export function rodDynamics(q, side, mass = 6, spacing = 0.59) {
  const {points, axes} = forward(q, side, spacing);
  const total = LENGTHS.reduce((s,x)=>s+x,0), inertia = [], gravity = [];
  for (let j=0; j<6; j++) {
    let J=0, g=0;
    for (let k=j; k<6; k++) {
      const m=mass*LENGTHS[k]/total, r0=sub(points[k],points[j]), r1=sub(points[k+1],points[j]);
      const a=cross(axes[j],r0), b=cross(axes[j],r1);
      J += m*(dot(a,a)+dot(a,b)+dot(b,b))/3;
      // Finite radius regularizes roll-axis inertia of an otherwise thin rod.
      J += m*0.02**2/2;
      g += dot(axes[j],cross(mul(add(r0,r1),0.5),[0,0,-9.81*m]));
    }
    inertia.push(Math.max(1e-5,J)); gravity.push(g);
  }
  return {inertia, gravity};
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

export function defaultSettings() {
  return {modelVersion:3,currentBasis:'peak',speed:1,mass:6,spacing:0.59,hold:2,
    limits:Array.from({length:12},(_,i)=>({velocity:i%6<3?Math.PI:MOTOR_SPECS[i%6].noLoadRpm*TAU/60,
      acceleration:0,current:i%6===1?16:i%6===5?2.3:4,kt:manufacturerKt(i),holdingCurrent:0}))};
}

export function validateSettings(s) {
  if(s.currentBasis!==undefined&&!['peak','rms'].includes(s.currentBasis)) throw Error('RobStrideの電流換算が不正です');
  for(const [key,min,max] of [["speed",0.05,8],["mass",0.01,100],["spacing",0,3],["hold",0,20]])
    if(!Number.isFinite(s[key])||s[key]<min||s[key]>max) throw Error(`${key} の設定範囲は ${min}〜${max} です`);
  if(!Array.isArray(s.limits)||s.limits.length!==12) throw Error("12関節の制限が必要です");
  for(const limit of s.limits) {
    if(limit.holdingCurrent!==undefined&&(!Number.isFinite(limit.holdingCurrent)||limit.holdingCurrent<0||limit.holdingCurrent>100))throw Error('静止分の予約電流は0〜100 Aです');
    for(const [key,min,max] of [["velocity",0.001,100],["acceleration",0,1000],["current",0,100],["kt",0.001,100]])
    if(!Number.isFinite(limit[key])||limit[key]<min||limit[key]>max) throw Error(`${key} の設定範囲は ${min}〜${max} です`);
  }
}

// Incremental closed-loop approximation about an already supported pose.
// Static equilibrium is supplied by the functioning position controller.
// Rod inertia restricts acceleration; unverified rod gravity cannot invent sag.
// holdingCurrent optionally reserves a measured/assumed static-current budget.
export function motionCapacity(joint, limit, inertia) {
  const availableTorque=Math.max(0,torqueCapacity(joint,limit)-limit.kt*(limit.holdingCurrent??0));
  return {torque:availableTorque,
    velocity:Math.min(limit.velocity,MOTOR_SPECS[joint%6].noLoadRpm*TAU/60),
    profileAcceleration:limit.acceleration===0?Infinity:limit.acceleration,
    currentAcceleration:availableTorque/inertia};
}

// Follow a moving reference instead of braking to a stop at every waypoint.
// This is a closed-loop approximation, not a replica of vendor firmware.
export function trackingStep(position, velocity, targetStart, targetEnd, dt, capacity) {
  const targetVelocity=(targetEnd-targetStart)/dt, error=targetStart-position;
  const accelerationLimit=Math.min(capacity.profileAcceleration,capacity.currentAcceleration);
  const braking=Math.max(0,Math.sqrt((accelerationLimit*dt)**2+2*accelerationLimit*Math.abs(error))-accelerationLimit*dt);
  const correction=Math.sign(error)*Math.min(braking,Math.abs(error)/dt);
  const requestedVelocity=targetVelocity+correction;
  const wanted=clamp(requestedVelocity,-capacity.velocity,capacity.velocity);
  const requestedAcceleration=(wanted-velocity)/dt;
  const acceleration=clamp(requestedAcceleration,-accelerationLimit,accelerationLimit);
  const nextVelocity=velocity+acceleration*dt;
  return {position:position+0.5*(velocity+nextVelocity)*dt,velocity:nextVelocity,acceleration,
    velocityLimited:Math.abs(requestedVelocity)>capacity.velocity+1e-9,
    accelerationLimited:Math.abs(requestedAcceleration)>capacity.profileAcceleration+1e-9,
    currentLimited:Math.abs(requestedAcceleration)>capacity.currentAcceleration+1e-9
      &&capacity.currentAcceleration<=capacity.profileAcceleration};
}

export function simulate(trajectory, settings, progress=()=>{}, {substeps=4}={}) {
  validateSettings(settings);
  if(!Number.isInteger(substeps)||substeps<1||substeps>64)throw Error('Invalid integration substeps');
  const fps=60;
  const motionDuration=trajectory.duration/settings.speed, duration=motionDuration+settings.hold;
  if(duration*fps>150000) throw Error("再生時間が長すぎます。区間を短くするか速度を上げてください");
  const count=Math.ceil(duration*fps)+1, times=[],ideal=[],actual=[],currents=[],paths=[[],[]],idealPaths=[[],[]];
  let q=trajectory.actions[0].slice(), v=Array(14).fill(0);
  let saturation=0,total=0,maxError=0,sumSquared=0,maxVelocity=0,maxAcceleration=0;
  const jointStats=ARM_JOINTS.map((_,a)=>({model:MOTOR_SPECS[a%6].model,
    availableTorque:motionCapacity(a,settings.limits[a],1).torque,
    maxAngleError:0,saturated:0,velocityLimited:0,accelerationLimited:0,currentLimited:0,
    maxTorque:0,maxDynamicCurrent:0}));
  for(let i=0;i<count;i++) {
    const t=Math.min(i/fps,duration);
    const desired=sample(trajectory.actions,trajectory.fps,Math.min(t,motionDuration)*settings.speed);
    let amps=Array(12).fill(0);
    if(i>0) {
      const step=(t-times.at(-1))/substeps;
      for(let k=0;k<substeps;k++) {
        const dynamics=[rodDynamics(q,0,settings.mass,settings.spacing),rodDynamics(q,1,settings.mass,settings.spacing)];
        const commandStart=times.at(-1)+k*step, commandEnd=commandStart+step;
        const targetStart=sample(trajectory.actions,trajectory.fps,Math.min(commandStart,motionDuration)*settings.speed);
        const targetEnd=sample(trajectory.actions,trajectory.fps,Math.min(commandEnd,motionDuration)*settings.speed);
        for(let a=0;a<12;a++) {
          const j=ARM_JOINTS[a], lim=settings.limits[a], d=jointStats[a];
          const J=dynamics[Math.floor(a/6)].inertia[a%6], capacity=motionCapacity(a,lim,J);
          const next=trackingStep(q[j],v[j],targetStart[j],targetEnd[j],step,capacity);
          q[j]=next.position;v[j]=next.velocity;
          const torque=J*next.acceleration;
          amps[a]=torque/lim.kt; // incremental current, NOT total/holding current
          d.maxTorque=Math.max(d.maxTorque,Math.abs(torque));
          d.maxDynamicCurrent=Math.max(d.maxDynamicCurrent,Math.abs(amps[a]));
          d.velocityLimited+=Number(next.velocityLimited);
          d.accelerationLimited+=Number(next.accelerationLimited);
          d.currentLimited+=Number(next.currentLimited);
          d.saturated+=Number(next.currentLimited);
          saturation+=Number(next.currentLimited);total++;
          maxVelocity=Math.max(maxVelocity,Math.abs(v[j]));
          maxAcceleration=Math.max(maxAcceleration,Math.abs(next.acceleration));
        }
        for(const j of [6,13]) q[j]=targetEnd[j];
      }
    }
    if(!q.every(Number.isFinite)||q.some(x=>Math.abs(x)>1e6)) throw Error("モデルが発散しました。設定を見直してください");
    times.push(t);ideal.push(desired);actual.push(q.slice());currents.push(amps);
    for(let side=0;side<2;side++) {
      for(let j=0;j<6;j++) {
        const a=side*6+j,index=ARM_JOINTS[a];
        jointStats[a].maxAngleError=Math.max(jointStats[a].maxAngleError,Math.abs(q[index]-desired[index]));
      }
      const p=forward(q,side,settings.spacing).tip, ref=forward(desired,side,settings.spacing).tip;
      paths[side].push(p);idealPaths[side].push(ref);
      const e=dot(sub(p,ref),sub(p,ref)); maxError=Math.max(maxError,Math.sqrt(e));sumSquared+=e;
    }
    if(i%600===0) progress(i/count);
  }
  for(const d of jointStats)for(const key of ['saturated','velocityLimited','accelerationLimited','currentLimited'])d[key]/=Math.max(1,(count-1)*substeps);
  return {times,ideal,actual,currents,paths,idealPaths,duration,motionDuration,fps,
    jointStats,modelVersion:3,
    stats:{maxError,rmsError:Math.sqrt(sumSquared/(count*2)),saturation:total?saturation/total:0,maxVelocity,maxAcceleration},
    segments:trajectory.segments.map(s=>({...s,start:s.start/settings.speed,end:s.end/settings.speed})),
    boundaries:trajectory.boundaries.map(b=>({...b,start:b.start/settings.speed,end:b.end/settings.speed}))};
}

export function createIdealExport(clips, edit, settings, fps=30) {
  if(!Number.isInteger(fps)||fps<1||fps>240)throw Error("書き出しFPSは1〜240の整数にしてください");
  // Rebuild from ORIGINAL data. Do not reuse the retimed preview trajectory.
  const original=clips.map(c=>({...c,replay:{...defaultReplaySettings(),base_speed:1,max_speedup:1}}));
  const ideal=stitch(original,{...edit,fps});
  const tasks=[...new Set(clips.map(c=>c.source.task).filter(Boolean))];
  return {coordinates:"iloha",fps,actions:ideal.actions,task:tasks.join(" → ")||"Merged ideal Iloha trajectory",
    settings:{schema_version:1,speed_applied:false,actuator_limits_applied:false,
      clips:clips.map(c=>({dataset:c.source.name,episode:c.source.episode??0,start:c.start,end:c.end,replay:{...c.replay},synthetic:!!c.source.synthetic})),
      edit:{...edit,fps,time_basis:"original_recording_seconds"},actuator:structuredClone(settings),
      simulation_model:{version:3,motor_specs:MOTOR_SPECS,current_basis:settings.currentBasis??'custom',
        control:'moving-reference velocity feedforward with velocity/acceleration/incremental-current saturation',dynamics:'diagonal uniform rod inertia about a supported equilibrium',
        equilibrium_assumption:'Static holding is supported, based on user observation; holdingCurrent reserves an optional per-joint current budget; no invented gravity-driven collapse',
        calibration:'Not fitted to measured feedback. Recorded observation.state may duplicate commands.',
        electrical_assumptions:'RobStride peak/RMS conversion selectable; DYNAMIXEL 12 V stall ratio, not continuous rating; peak cap, no thermal or torque-speed curve simulation'},
      ideal_segments:ideal.segments,ideal_boundaries:ideal.boundaries,
      note:"Trimming and transitions only. No replay speed scaling, actuator simulation, or final hold is applied to the dataset."}};
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
