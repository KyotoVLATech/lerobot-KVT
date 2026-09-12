import {stitch, createPreview} from './core.mjs';
self.onmessage = ({data}) => {
  try {
    const trajectory=stitch(data.clips,data.edit);
    const result=createPreview(trajectory,data.spacing);
    self.postMessage({result,trajectory,id:data.id});
  } catch(error) { self.postMessage({error:error.message,id:data.id}); }
};
