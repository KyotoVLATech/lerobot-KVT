import test from 'node:test';
import assert from 'node:assert/strict';
import {forward,LENGTHS,rodDynamics,stitch,simulate,defaultSettings,defaultReplaySettings,replayIntervals,unwrap,createIdealExport,MOTOR_SPECS,manufacturerKt,torqueCapacity,motionCapacity,trackingStep,validateSettings} from '../tools/iloha_trajectory_web/core.mjs';

const near=(a,b,e=1e-8)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
const constant=(value,frames=61)=>({name:'fixture',fps:30,coordinates:'iloha',actions:Array.from({length:frames},()=>Array(14).fill(value))});
const clip=source=>({source,start:0,end:(source.actions.length-1)/source.fps,replay:{...defaultReplaySettings(),max_speedup:1}});

test('dataset export ignores every speed and actuator setting but retains trimming and transition',()=>{
  const a=clip(constant(0,121)),b=clip(constant(1,121));a.start=1;a.end=3;b.start=.5;b.end=3.5;
  const edit={mode:'linear',blend:1},settings=defaultSettings();
  const first=createIdealExport([a,b],edit,settings,30);
  a.replay={base_speed:5,max_speedup:6,gripper_margin:10,speedup_distance:3,gripper_threshold:.2};
  b.replay={base_speed:.2,max_speedup:2,gripper_margin:0,speedup_distance:.1,gripper_threshold:0};
  settings.limits.forEach(l=>{l.current=0;l.velocity=.01;l.acceleration=.01;});settings.hold=20;settings.mass=50;
  const second=createIdealExport([a,b],edit,settings,30);
  assert.deepEqual(first.actions,second.actions);
  assert.equal(first.actions.length,181); // 2 + 1 + 3 seconds at 30 FPS, inclusive endpoints.
  near(first.actions[75][0],.5);
  assert.equal(second.settings.clips[0].replay.base_speed,5);
  assert.equal(second.settings.actuator.limits[0].current,0);
  assert.equal(second.settings.speed_applied,false);
  assert.equal(second.settings.actuator_limits_applied,false);
});

test('FK inverts the reference IK wrist equations and preserves link lengths',()=>{
  const q=[.2,-.3,.4,.5,.6,.7,0, -.4,.1,.2,-.3,.2,.4,0];
  for(let side=0;side<2;side++){
    const {points,tip}=forward(q,side),o=side*7;
    for(let j=0;j<6;j++)near(Math.hypot(...points[j+1].map((v,k)=>v-points[j][k])),LENGTHS[j]);
    const t0=-(q[o]+(side===0?-Math.PI/2:Math.PI/2));
    const t1=-q[o+1]-Math.PI/2+.19599,t2=q[o+2]+Math.PI-.19599;
    const C=Math.cos(t1+t2),S=Math.sin(t1+t2),c=Math.cos(t0),s=Math.sin(t0);
    const ax=C*c*Math.sin(q[o+4])*Math.cos(q[o+3])-s*Math.sin(q[o+4])*Math.sin(q[o+3])+S*c*Math.cos(q[o+4]);
    const ay=C*s*Math.sin(q[o+4])*Math.cos(q[o+3])+c*Math.sin(q[o+4])*Math.sin(q[o+3])+S*s*Math.cos(q[o+4]);
    const az=-S*Math.sin(q[o+4])*Math.cos(q[o+3])+C*Math.cos(q[o+4]);
    const radius=LENGTHS[1]*Math.sin(t1)+(LENGTHS[2]+LENGTHS[3])*S;
    near(tip[0],radius*c+(LENGTHS[4]+LENGTHS[5])*ax);
    near(tip[1],(side===0?-.295:.295)+radius*s+(LENGTHS[4]+LENGTHS[5])*ay);
    near(tip[2],LENGTHS[0]+LENGTHS[1]*Math.cos(t1)+(LENGTHS[2]+LENGTHS[3])*C+(LENGTHS[4]+LENGTHS[5])*az);
  }
});

test('joint axes agree with finite-difference FK and roll affects fingers',()=>{
  const q=Array(14).fill(.3),base=forward(q),epsilon=1e-6;
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  for(let j=0;j<6;j++){
    const next=q.slice();next[j]+=epsilon;
    const predicted=cross(base.axes[j],base.tip.map((v,k)=>v-base.points[j][k]));
    const actual=forward(next).tip.map((v,k)=>(v-base.tip[k])/epsilon);
    actual.forEach((v,k)=>near(v,predicted[k],2e-6));
  }
  const rolled=q.slice();rolled[5]+=1;
  assert.notDeepEqual(base.fingers,forward(rolled).fingers);
});

test('distributed rod inertia scales with mass and gravity is potential gradient',()=>{
  const q=Array(14).fill(.3),a=rodDynamics(q,0,6),b=rodDynamics(q,0,12);
  a.inertia.forEach((v,j)=>{assert.ok(v>0);near(b.inertia[j],v*2);});
  const total=LENGTHS.reduce((s,v)=>s+v,0);
  const potential=q=>{const p=forward(q).points;return LENGTHS.reduce((s,l,j)=>s+6*l/total*9.81*(p[j][2]+p[j+1][2])/2,0);};
  for(let j=0;j<6;j++){const next=q.slice();next[j]+=1e-6;near(a.gravity[j],-(potential(next)-potential(q))/1e-6,2e-5);}
});

test('replay speed settings preserve gripper activity and its adjacent intervals',()=>{
  const rows=constant(0,301).actions;
  for(let i=151;i<rows.length;i++)rows[i][6]=1;
  const intervals=replayIntervals(rows,30,{base_speed:2,max_speedup:3,gripper_margin:.5,speedup_distance:1});
  near(intervals[150],1/60);near(intervals[149],1/60);near(intervals[151],1/60);
  near(intervals[0],1/180);near(intervals.at(-1),1/180);
  assert.ok(intervals[120]>1/180&&intervals[120]<1/60);
  near(replayIntervals(constant(0).actions,30,{base_speed:2,max_speedup:3})[0],1/180);
});

test('each clip has independent speed settings',()=>{
  const a=clip(constant(0)),b=clip(constant(.2));a.replay.base_speed=2;
  const t=stitch([a,b],{mode:'cut',fps:60});
  near(t.segments[0].end-t.segments[0].start,1);
  near(t.segments[1].end-t.segments[1].start,2);
});

test('crossfade overlaps in time, preserves endpoints, and trims sources',()=>{
  const a=clip(constant(0,121)),b=clip(constant(1,121));a.start=1;a.end=3;b.start=.5;b.end=3.5;
  const t=stitch([a,b],{mode:'crossfade',blend:1,fps:60});
  near(t.duration,4);near(t.actions[0][0],0);near(t.actions.at(-1)[0],1);
  near(t.actions[90][0],.5);near(t.boundaries[0].start,1);near(t.boundaries[0].end,2);
  assert.throws(()=>stitch([a,b],{mode:'crossfade',blend:3}),/短く/);
});

test('linear transition has correct duration and uniformly interpolated angles',()=>{
  const a=clip(constant(0)),b=clip(constant(1));const t=stitch([a,b],{mode:'linear',blend:1,fps:60});
  near(t.duration,5);near(t.actions[150][0],.5);near(t.actions[120][0],0);near(t.actions[180][0],1);
  const smooth=stitch([a,b],{mode:'smooth',blend:1,fps:60});near(smooth.actions[150][0],.5);
  assert.ok(smooth.actions[121][0]<t.actions[121][0]);
});

test('angular wrapping is short-path and grippers are not wrapped',()=>{
  const rows=constant(0,2).actions;rows[0][0]=Math.PI-.1;rows[1][0]=-Math.PI+.1;rows[1][6]=5;
  const result=unwrap(rows);near(result[1][0]-result[0][0],.2);near(result[1][6],5);
});

test('supported stationary poses do not invent gravity sag',()=>{
  const s=defaultSettings();s.hold=3;
  const t=stitch([clip(constant(.3,31))]),out=simulate(t,s);
  out.actual.forEach(q=>q.forEach(v=>near(v,.3)));
  near(out.stats.maxError,0);out.currents.flat().forEach(i=>near(i,0));
});

test('left/right bases follow Unity right axis and stay 590 mm apart',()=>{
  const q=Array(14).fill(0),left=forward(q,0).points[0],right=forward(q,1).points[0];
  assert.deepEqual(left,[0,-.295,0]);assert.deepEqual(right,[0,.295,0]);near(right[1]-left[1],.59);
});

test('joint-specific manufacturer values convert RMS only once and cap peak torque',()=>{
  const s=defaultSettings();
  assert.deepEqual(MOTOR_SPECS.slice(0,3).map(m=>m.model),['RobStride 03','RobStride 06','RobStride 00']);
  for(let side=0;side<2;side++) {
    [2.36,1.10,1.48].forEach((kt,j)=>{near(s.limits[side*6+j].kt,kt/Math.SQRT2);near(manufacturerKt(j,'rms'),kt);});
    near(s.limits[side*6+3].kt,10.6/4.4);near(s.limits[side*6+5].kt,4.1/2.3);
  }
  for(let a=0;a<12;a++)near(torqueCapacity(a,{current:100,kt:100}),MOTOR_SPECS[a%6].peakTorque);
  near(torqueCapacity(2,s.limits[2]),4*1.48/Math.SQRT2);
});

test('automatic acceleration derives from torque and inertia; explicit caps remain hard caps',()=>{
  const l={current:4,kt:1,holdingCurrent:1,velocity:2,acceleration:0};
  const c=motionCapacity(2,l,.5);near(c.torque,3);near(c.currentAcceleration,6);
  assert.equal(c.profileAcceleration,Infinity);
  l.acceleration=.5;assert.equal(motionCapacity(2,l,.5).profileAcceleration,.5);
  l.holdingCurrent=5;assert.equal(motionCapacity(2,l,.5).torque,0);
  assert.equal(defaultSettings().limits[3].acceleration,0); // no fictitious DYNAMIXEL PP cap
  assert.throws(()=>validateSettings({...defaultSettings(),limits:Array(12).fill({...l,acceleration:-1})}));
});

test('moving-target feedforward has no waypoint stopping or artificial steady lag',()=>{
  const cap={profileAcceleration:2,currentAcceleration:10,velocity:3};
  let p=0,v=1;
  for(let k=0;k<240;k++){
    const n=trackingStep(p,v,k/240,(k+1)/240,1/240,cap);
    p=n.position;v=n.velocity;near(p,(k+1)/240);near(v,1);
  }
});

test('position is integrated continuously and explicit velocity and acceleration limits hold',()=>{
  const cap={profileAcceleration:.5,currentAcceleration:10,velocity:.2};
  let p=0,v=0;
  for(let k=0;k<480;k++){
    const n=trackingStep(p,v,k/240,(k+1)/240,1/240,cap);
    assert.ok(Math.abs(n.acceleration)<=.5+1e-9);assert.ok(Math.abs(n.velocity)<=.2+1e-9);
    near(n.position-p,(v+n.velocity)/480);p=n.position;v=n.velocity;
  }
  assert.ok(p<.5); // must not teleport to the requested position 2
});

test('current budget and a static reserve change tracking without changing the ideal data',()=>{
  const src=constant(0,61);src.actions.forEach((q,i)=>q[2]=i/60);
  const t=stitch([clip(src)]),low=defaultSettings();low.hold=0;low.limits[2].current=.05;
  const high=defaultSettings();high.hold=0;
  const a=simulate(t,low),b=simulate(t,high);
  assert.ok(a.stats.rmsError>b.stats.rmsError*2);
  assert.ok(a.jointStats[2].currentLimited>0);
  a.currents.forEach(row=>assert.ok(Math.abs(row[2])<=.05+1e-9));
  const reserved=structuredClone(high);reserved.limits[2].holdingCurrent=3.95;
  const c=simulate(t,reserved);
  near(c.stats.rmsError,a.stats.rmsError,1e-8);
  assert.deepEqual(a.ideal,b.ideal);assert.deepEqual(c.ideal,b.ideal);
});

test('relaxing an explicit acceleration limit improves ramp tracking',()=>{
  const src=constant(0,91);src.actions.forEach((q,i)=>q[0]=i/90);
  const t=stitch([clip(src)]),low=defaultSettings();low.hold=0;low.limits[0].acceleration=.1;
  const high=structuredClone(low);high.limits[0].acceleration=10;
  const a=simulate(t,low),b=simulate(t,high);
  assert.ok(a.jointStats[0].accelerationLimited>0);
  assert.ok(Math.abs(b.actual.at(-1)[0]-1)<Math.abs(a.actual.at(-1)[0]-1));
});

test('zero incremental current allows no motion from rest',()=>{
  const src=constant(0,61);src.actions.forEach((q,i)=>q[2]=i/60);
  const s=defaultSettings();s.limits.forEach(l=>l.current=0);
  const out=simulate(stitch([clip(src)]),s);
  out.actual.forEach(q=>q.slice(0,6).forEach(x=>near(x,0)));
  out.currents.flat().forEach(x=>near(x,0));
});

test('simulation uses manufacturer peak caps even with excessive user current limits',()=>{
  const src=constant(0,31);src.actions.forEach((q,i)=>q[2]=q[9]=i>0?2:0);
  const s=defaultSettings();s.mass=100;s.hold=0;s.limits.forEach(l=>{l.current=100;l.kt=100;});
  const out=simulate(stitch([clip(src)]),s);
  out.jointStats.forEach((d,a)=>assert.ok(d.maxTorque<=MOTOR_SPECS[a%6].peakTorque+1e-9));
  assert.ok(out.stats.saturation>0);
});

test('time integration converges for a current-limited moving elbow',()=>{
  const src=constant(0,91);src.actions.forEach((q,i)=>q[2]=.3*Math.sin(i/90*Math.PI));
  const s=defaultSettings();s.hold=0;s.limits[2].current=.1;
  const t=stitch([clip(src)]),normal=simulate(t,s),fine=simulate(t,s,()=>{},{substeps:16});
  assert.ok(Math.abs(normal.actual.at(-1)[2]-fine.actual.at(-1)[2])<.02);
});

test('ideal export carries model assumptions and reservations separately from ideal actions',()=>{
  const s=defaultSettings();s.limits[2].holdingCurrent=1;
  const out=createIdealExport([clip(constant(0))],{mode:'cut',blend:0},s);
  assert.equal(out.settings.simulation_model.version,3);
  assert.equal(out.settings.simulation_model.current_basis,'peak');
  assert.equal(out.settings.simulation_model.motor_specs[1].model,'RobStride 06');
  assert.equal(out.settings.simulation_model.motor_specs[1].peakTorque,36);
  assert.equal(out.settings.actuator.limits[2].holdingCurrent,1);
});
