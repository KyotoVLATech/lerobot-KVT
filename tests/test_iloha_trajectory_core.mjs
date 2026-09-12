import test from 'node:test';
import assert from 'node:assert/strict';
import {forward,LENGTHS,stitch,makeDemo,defaultReplaySettings,replayIntervals,unwrap,createTrajectorySettings,createPreview,defaultViewSettings,projectView,projectClipName,validateSpacing,sample} from '../tools/iloha_trajectory_web/core.mjs';

const near=(a,b,e=1e-8)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
const constant=(value,frames=61)=>({name:'fixture',fps:30,coordinates:'iloha',actions:Array.from({length:frames},()=>Array(14).fill(value))});
const clip=source=>({source,start:0,end:(source.actions.length-1)/source.fps,replay:{...defaultReplaySettings(),max_speedup:1}});

test('exported settings describe the previewed trajectory, speed settings included',()=>{
  const a=clip(constant(0,121)),b=clip(constant(1,121));a.start=1;a.end=3;b.start=.5;b.end=3.5;
  a.replay={base_speed:2,max_speedup:1,gripper_margin:10,speedup_distance:3,gripper_threshold:.2};
  const edit={mode:'linear',blend:1,fps:30};
  const trajectory=stitch([a,b],edit);
  const output=createTrajectorySettings([a,b],edit,trajectory);
  assert.equal(output.schema_version,3);
  assert.equal(output.speed_applied,true);
  // 1 (2 s at 2x) + 1 blend + 3 seconds at 30 FPS, inclusive endpoints.
  assert.equal(output.trajectory.frames,151);
  assert.equal(output.trajectory.frames,trajectory.actions.length);
  near(output.trajectory.duration,5);
  assert.equal(output.trajectory.fps,30);
  assert.equal(output.edit.fps,30);
  assert.deepEqual(output.trajectory.segments,trajectory.segments);
  assert.deepEqual(output.trajectory.boundaries,trajectory.boundaries);
  assert.deepEqual(output.clips[0].replay,a.replay);
  assert.equal(output.clips[0].dataset,'fixture');
  assert.equal(output.clips[0].start,1);
});

test('settings export refuses demo data that the robot cannot load',()=>{
  const demo=clip(makeDemo());
  const edit={mode:'cut',blend:0,fps:30};
  assert.throws(()=>createTrajectorySettings([demo],edit,stitch([demo],edit)),/実機/);
  assert.throws(()=>createTrajectorySettings([],edit,null),/データセット/);
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

test('left/right bases follow Unity right axis and stay 590 mm apart',()=>{
  const q=Array(14).fill(0),left=forward(q,0).points[0],right=forward(q,1).points[0];
  assert.deepEqual(left,[0,-.295,0]);assert.deepEqual(right,[0,.295,0]);near(right[1]-left[1],.59);
});


test('preview contains only ideal poses and full FK paths, with no simulated output',()=>{
  const src=constant(0,61);src.actions.forEach((q,i)=>q[0]=i/60);
  const trajectory=stitch([clip(src)]),preview=createPreview(trajectory);
  assert.equal(preview.ideal,trajectory.actions);
  near(preview.duration,trajectory.duration);
  near(preview.times.at(-1),trajectory.duration);
  assert.equal(preview.times.length,trajectory.actions.length);
  for(const key of ['actual','actualPaths','stats','currents','jointStats','modelVersion'])assert.equal(key in preview,false);
  for(let side=0;side<2;side++){
    assert.equal(preview.idealPaths[side].length,trajectory.actions.length);
    trajectory.actions.forEach((q,i)=>assert.deepEqual(preview.idealPaths[side][i],forward(q,side,.59).tip));
  }
  near(sample(preview.ideal,preview.fps,1)[0],.5);
  near(sample(preview.ideal,preview.fps,preview.duration)[0],1);
});

test('spacing changes only the geometry, not the ideal angles or timing',()=>{
  const t=stitch([clip(constant(.3))]),a=createPreview(t,.59),b=createPreview(t,.79);
  assert.deepEqual(a.ideal,b.ideal);assert.deepEqual(a.times,b.times);
  near(b.idealPaths[0][0][1]-a.idealPaths[0][0][1],-.1);
  near(b.idealPaths[1][0][1]-a.idealPaths[1][0][1],.1);
  assert.deepEqual(defaultViewSettings(),{spacing:.59});
  for(const value of [-1,4,NaN,Infinity,'0.59'])assert.throws(()=>validateSpacing(value));
});

test('projects retain view geometry while legacy simulator settings are discarded',()=>{
  assert.deepEqual(projectView({version:1,settings:{spacing:.7,mass:-10,limits:'obsolete'}}),{spacing:.7});
  assert.deepEqual(projectView({version:1}),{spacing:.59});
  assert.deepEqual(projectView({version:2,view:{spacing:.8}}),{spacing:.8});
  assert.throws(()=>projectView({version:3}));
  assert.throws(()=>projectView({version:2,view:{spacing:-1}}));
});

test('the exported robot settings can be loaded back into the editor',()=>{
  // trajectory_settings.json carries no view geometry and names clips "dataset".
  for(const version of [2,3])assert.deepEqual(projectView({schema_version:version}),{spacing:.59});
  assert.throws(()=>projectView({schema_version:1}),/形式/);
  assert.throws(()=>projectView({}),/形式/);
  assert.equal(projectClipName({dataset:'iloha-best'}),'iloha-best');
  assert.equal(projectClipName({name:'iloha-best'}),'iloha-best');
  for(const clip of [{},{name:''},{dataset:7},null])assert.throws(()=>projectClipName(clip),/データセット名/);
});

test('settings export carries edits but no removed simulator model',()=>{
  const c=clip(constant(0)),edit={mode:'cut',blend:0,fps:60};
  const output=createTrajectorySettings([c],edit,stitch([c],edit));
  assert.equal(output.edit.mode,'cut');
  assert.equal(output.edit.time_basis,'speed_adjusted_seconds');
  assert.equal(output.coordinates,'iloha');
  for(const key of ['actuator','simulation_model','actuator_limits_applied','actions'])assert.equal(key in output,false);
});
