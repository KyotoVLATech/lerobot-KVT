import {stitch, simulate} from './core.mjs';
self.onmessage = ({data}) => {
  try {
    const trajectory=stitch(data.clips,data.edit);
    const result=simulate(trajectory,data.settings,progress=>self.postMessage({progress,id:data.id}));
    self.postMessage({result,trajectory,id:data.id});
  } catch(error) { self.postMessage({error:error.message,id:data.id}); }
};
