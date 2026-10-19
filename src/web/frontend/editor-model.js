/* Document operations shared by the editor and its interaction regression checks. */
(() => {
  'use strict';
  const copy = value => structuredClone(value);
  const uid = prefix => prefix + '-' + crypto.randomUUID();
  const bounds = elements => elements.length ? [Math.min(...elements.map(e=>e.bbox[0])), Math.min(...elements.map(e=>e.bbox[1])), Math.max(...elements.map(e=>e.bbox[2])), Math.max(...elements.map(e=>e.bbox[3]))] : [0,0,0,0];
  const rotate = (p, c, degrees) => {
    const angle = degrees * Math.PI / 180, x=p.x-c.x, y=p.y-c.y;
    return {x:c.x+x*Math.cos(angle)-y*Math.sin(angle), y:c.y+x*Math.sin(angle)+y*Math.cos(angle)};
  };
  function normalize(doc) {
    if (!doc) return;
    if (!Array.isArray(doc.layers) || !doc.layers.length) doc.layers = [{id:'main', name:'기본 레이어', visible:true, locked:false}];
    if (!doc.layers.some(layer=>layer.id===doc.active_layer)) doc.active_layer = doc.layers[0].id;
    doc.canvas ||= {grid:true, grid_size:10, grid_color:'#cdd2dc', background:'#ffffff'};
    if (!Array.isArray(doc.pages)) doc.pages = [];
    doc.active_page ||= 'page-1';
    doc.page_name ||= '페이지 1';
    for (const e of doc.elements) e.layer_id ||= doc.active_layer;
  }
  function layer(doc, e) { return (doc.layers||[]).find(l=>l.id===e.layer_id); }
  function visible(doc, e) { return !e.hidden && layer(doc,e)?.visible !== false; }
  function locked(doc, e) { return !!(e.locked || layer(doc,e)?.locked); }
  function expandGroup(doc, ids) {
    const groups=new Set(doc.elements.filter(e=>ids.includes(e.id)&&e.group_id).map(e=>e.group_id));
    return doc.elements.filter(e=>visible(doc,e)&&!locked(doc,e)&&(ids.includes(e.id)||groups.has(e.group_id))).map(e=>e.id);
  }
  function group(doc, ids) {
    const items=doc.elements.filter(e=>ids.includes(e.id)&&!locked(doc,e));
    if(items.length<2) throw new Error('두 개 이상의 편집 가능한 요소를 선택하세요.');
    const id=uid('group'); items.forEach(e=>e.group_id=id); return id;
  }
  function translate(e, dx, dy, original=e) {
    e.bbox=original.bbox.map((v,i)=>v+(i%2?dy:dx));
    for(const key of ['start','end']) if(original[key]) {
      const point=original[key],x=Array.isArray(point)?point[0]:point.x,y=Array.isArray(point)?point[1]:point.y;
      e[key]={x:x+dx,y:y+dy};
    }
    if(original.points) e.points=original.points.map(p=>({x:p.x+dx,y:p.y+dy}));
  }
  function boundary(e, toward, port) {
    const [x,y,r,b]=e.bbox,w=r-x,h=b-y,c={x:(x+r)/2,y:(y+b)/2};
    const local=rotate(toward,c,-(e.rotation||0));
    let dx=local.x-c.x,dy=local.y-c.y;
    if(port) {const p={top:[0,-1],right:[1,0],bottom:[0,1],left:[-1,0]}[port]; if(p)[dx,dy]=p;}
    if(!dx&&!dy)dy=1;
    let t;
    if(e.shape==='decision')t=.5/(Math.abs(dx)/w+Math.abs(dy)/h);
    else if(e.shape==='connector')t=1/Math.sqrt((dx/(w/2))**2+(dy/(h/2))**2);
    else t=.5/Math.max(Math.abs(dx)/w,Math.abs(dy)/h);
    if(e.shape==='terminator') {
      const rad=Math.min(w,h)/2, inset=w/2-rad, px=dx*t,py=dy*t;
      if(Math.abs(px)>inset) {
        const side=Math.sign(dx)*inset, a=dx*dx+dy*dy,bb=-2*dx*side,cc=side*side-rad*rad;
        t=(-bb+Math.sqrt(Math.max(0,bb*bb-4*a*cc)))/(2*a);
      }
    }
    return rotate({x:c.x+dx*t,y:c.y+dy*t},c,e.rotation||0);
  }
  function arrowBounds(e,size) {
    e.bbox=[Math.max(0,Math.min(e.start.x,e.end.x)),Math.max(0,Math.min(e.start.y,e.end.y)),Math.min(size.width,Math.max(e.start.x,e.end.x)+1),Math.min(size.height,Math.max(e.start.y,e.end.y)+1)];
  }
  function jumpPath(arrow, previous, mode='none') {
    const a=arrow.start,b=arrow.end,dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy);
    if(!length||mode==='none')return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
    const crossings=[];
    for(const other of previous){
      if(!other.start||!other.end||other.route&&other.route!=='straight')continue;
      const c=other.start,d=other.end,ex=d.x-c.x,ey=d.y-c.y,det=dx*ey-dy*ex;
      if(Math.abs(det)<1e-6)continue;
      const rx=c.x-a.x,ry=c.y-a.y,t=(rx*ey-ry*ex)/det,u=(rx*dy-ry*dx)/det;
      if(t>.05&&t<.95&&u>.05&&u<.95)crossings.push(t);
    }
    crossings.sort((x,y)=>x-y);
    let path=`M ${a.x} ${a.y}`,last=0;
    for(const t of crossings){const pad=Math.min(8/length,(t-last)/3,(1-t)/3),before=t-pad,after=t+pad;
      if(before<=last)continue;
      const p=v=>({x:a.x+dx*v,y:a.y+dy*v}),left=p(before),right=p(after),middle=p(t),normal={x:-dy/length,y:dx/length};
      path+=` L ${left.x} ${left.y}`;
      if(mode==='gap')path+=` M ${right.x} ${right.y}`;
      else if(mode==='sharp')path+=` L ${middle.x+normal.x*10} ${middle.y+normal.y*10} L ${right.x} ${right.y}`;
      else path+=` Q ${middle.x+normal.x*13} ${middle.y+normal.y*13} ${right.x} ${right.y}`;
      last=after;
    }
    return path+` L ${b.x} ${b.y}`;
  }
  function resize(bbox,handle,dx,dy,size,ratio=false) {
    let [x,y,r,b]=bbox, w=r-x,h=b-y;
    if(handle.includes('e'))r=Math.max(x+12,r+dx);
    if(handle.includes('w'))x=Math.min(r-12,x+dx);
    if(handle.includes('s'))b=Math.max(y+12,b+dy);
    if(handle.includes('n'))y=Math.min(b-12,y+dy);
    if(ratio){const scale=Math.max((r-x)/w,(b-y)/h); if(handle.includes('w'))x=r-w*scale;else r=x+w*scale;if(handle.includes('n'))y=b-h*scale;else b=y+h*scale;}
    const scale=Math.min(1,size.width/(r-x),size.height/(b-y));r=x+(r-x)*scale;b=y+(b-y)*scale;
    const tx=Math.max(-x,Math.min(size.width-r,0)),ty=Math.max(-y,Math.min(size.height-b,0));
    return [x+tx,y+ty,r+tx,b+ty];
  }
  function pageSnapshot(doc) {return copy({id:doc.active_page,name:doc.page_name,image:doc.image,elements:doc.elements,relations:doc.relations,warnings:doc.warnings,layers:doc.layers,active_layer:doc.active_layer,canvas:doc.canvas});}
  function switchPage(doc,id) {
    normalize(doc); if(id===doc.active_page)return;
    const index=doc.pages.findIndex(p=>p.id===id); if(index<0)throw new Error('페이지를 찾을 수 없습니다.');
    const next=doc.pages.splice(index,1)[0],previous=pageSnapshot(doc);doc.pages.push(previous);
    Object.assign(doc,{active_page:next.id,page_name:next.name,image:next.image,elements:next.elements,relations:next.relations,warnings:next.warnings||[],layers:next.layers,active_layer:next.active_layer,canvas:next.canvas});normalize(doc);
  }
  function addPage(doc,name) {normalize(doc);const id=uid('page');doc.pages.push({id,name:name||'새 페이지',image:{width:doc.image.width,height:doc.image.height},elements:[],relations:[],warnings:[],layers:[{id:'main',name:'기본 레이어',visible:true,locked:false}],active_layer:'main',canvas:copy(doc.canvas)});switchPage(doc,id);return id;}
  function layout(doc, spacing=80, horizontal=false) {
    const nodes=doc.elements.filter(e=>e.type==='Box'&&visible(doc,e)&&!locked(doc,e)), ids=new Set(nodes.map(e=>e.id));
    const levels=new Map(nodes.map(e=>[e.id,0])), indegree=new Map(nodes.map(e=>[e.id,0]));
    const edges=doc.relations.filter(r=>r.type==='flows_to'&&ids.has(r.source)&&ids.has(r.target));
    edges.forEach(r=>indegree.set(r.target,indegree.get(r.target)+1));
    const queue=nodes.filter(e=>indegree.get(e.id)===0).map(e=>e.id), seen=new Set();
    while(queue.length){const id=queue.shift();seen.add(id);for(const e of edges.filter(r=>r.source===id)){levels.set(e.target,Math.max(levels.get(e.target),levels.get(id)+1));indegree.set(e.target,indegree.get(e.target)-1);if(indegree.get(e.target)===0)queue.push(e.target);}}
    let cycleLevel=Math.max(0,...levels.values());for(const node of nodes)if(!seen.has(node.id))levels.set(node.id,cycleLevel++);
    const ranks=[...new Set(levels.values())].sort((a,b)=>a-b);let major=40;
    for(const rank of ranks){const row=nodes.filter(e=>levels.get(e.id)===rank);let minor=40,max=0;for(const e of row){const w=e.bbox[2]-e.bbox[0],h=e.bbox[3]-e.bbox[1];e.original_bbox ||= e.bbox.slice();translate(e,(horizontal?major:minor)-e.bbox[0],(horizontal?minor:major)-e.bbox[1]);minor+=(horizontal?h:w)+spacing;max=Math.max(max,horizontal?w:h);}major+=max+spacing;}
    const b=bounds(doc.elements),width=Math.max(doc.image.width,Math.ceil(b[2]+40)),height=Math.max(doc.image.height,Math.ceil(b[3]+40));
    if(width>40000||height>40000||width*height>40000000)throw new Error('자동 배치 결과가 캔버스 크기 제한을 넘습니다. 간격을 줄이세요.');
    doc.image.width=width;doc.image.height=height;
  }
  globalThis.RaduModel={copy,uid,bounds,rotate,normalize,visible,locked,expandGroup,group,translate,boundary,arrowBounds,jumpPath,resize,pageSnapshot,switchPage,addPage,layout};
})();
