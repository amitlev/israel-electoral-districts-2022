import numpy as np, pickle, json
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint, mapping
from shapely.ops import unary_union, transform, voronoi_diagram
from pyproj import Transformer
import engine as E
np.random.seed(1)
data=E.load(); isr=pickle.load(open('isr_itm.pkl','rb'))
T=Transformer.from_crs('EPSG:2039','EPSG:4326',always_xy=True)
BLOC4={'ימין':['מחל','ט'],'מרכז-שמאל':['פה','כן','אמת','מרצ','ל'],'חרדי':['שס','ג'],'ערבי':['ום','עם','ד']}
idx={p:i for i,p in enumerate(data['parties'])}

def vpolys(points):
    mn=points.min(0)-60000;mx=points.max(0)+60000
    ex=np.array([[mn[0],mn[1]],[mn[0],mx[1]],[mx[0],mn[1]],[mx[0],mx[1]]])
    vor=Voronoi(np.vstack([points,ex]));P=[]
    for i in range(len(points)):
        r=vor.regions[vor.point_region[i]];P.append(None if(not r or -1 in r) else Polygon([vor.vertices[j] for j in r]))
    return P
def ashape(points,maxR=14000):
    tri=Delaunay(points);k=[]
    for ia,ib,ic in tri.simplices:
        a,b,c=points[ia],points[ib],points[ic]
        ab=np.hypot(*(a-b));bc=np.hypot(*(b-c));ca=np.hypot(*(c-a));s=(ab+bc+ca)/2
        ar=(max(s*(s-ab)*(s-bc)*(s-ca),1e-9))**0.5;R=ab*bc*ca/(4*ar)
        if R<maxR:k.append(Polygon([a,b,c]))
    return unary_union(k)
def sample_in(cell,nn):
    minx,miny,maxx,maxy=cell.bounds;out=[];tries=0
    while len(out)<nn and tries<nn*50:
        x=np.random.uniform(minx,maxx);y=np.random.uniform(miny,maxy);tries+=1
        if cell.contains(Point(x,y)):out.append((x,y))
    return np.array(out) if out else np.array([[(minx+maxx)/2,(miny+maxy)/2]])
def largest_remainder(w,M):
    w=np.array(w,float);q=w/w.sum()*M;base=np.floor(q).astype(int)
    r=M-base.sum();frac=q-base
    for i in np.argsort(-frac)[:r]: base[i]+=1
    base[base<1]=1
    # trim back to M if min-clamp overshot
    while base.sum()>M:
        base[np.argmax(base)]-=1
    while base.sum()<M:
        base[np.argmin(base)]+=1
    return base

def split_cell(cell,dists,ws,dcent):
    k=len(dists);M=max(30,10*k)
    seeds=sample_in(cell,M)
    if len(seeds)<k: return {dists[int(np.argmax(ws))]:cell}
    samp=sample_in(cell,2500)
    for _ in range(14):
        lab=((samp[:,None,:]-seeds[None,:,:])**2).sum(2).argmin(1)
        for i in range(len(seeds)):
            if (lab==i).any(): seeds[i]=samp[lab==i].mean(0)
    M=len(seeds)
    caps=largest_remainder(ws,M)
    # cost seed->district = distance to district global centroid; expand columns by caps
    cent=np.array([dcent[d] for d in dists])
    cost=np.sqrt(((seeds[:,None,:]-cent[None,:,:])**2).sum(2))  # M x k
    cols=np.repeat(np.arange(k),caps)  # length M
    C=cost[:,cols]                     # M x M
    ri,ci=linear_sum_assignment(C)
    labels=np.empty(M,int)
    for r,c in zip(ri,ci): labels[r]=dists[cols[c]]
    vd=voronoi_diagram(MultiPoint([Point(*p) for p in seeds]),envelope=cell)
    sub={}
    for poly in vd.geoms:
        cc=poly.intersection(cell)
        if cc.is_empty: continue
        rp=cc.representative_point();pc=np.array([rp.x,rp.y])
        si=int(((seeds-pc)**2).sum(1).argmin())
        sub.setdefault(int(labels[si]),[]).append(cc)
    return {d:unary_union(v) for d,v in sub.items()}
def rnd(o,p=5):
    if isinstance(o,list):
        if o and isinstance(o[0],(int,float)):return [round(o[0],p),round(o[1],p)]
        return [rnd(x,p) for x in o]
    return o

def build(n):
    a=E.solve(data,n,alpha=0.0,do_repair=False)
    # district global centroids (voter-weighted)
    dcent={}
    for d in range(n):
        m=a==d
        if m.any(): dcent[d]=np.average(data['X'][m],axis=0,weights=data['W'][m])
    cities=np.unique(data['city']);cmap={c:i for i,c in enumerate(cities)}
    cid=np.array([cmap[c] for c in data['city']])
    cent=np.array([data['X'][cid==i].mean(0) for i in range(len(cities))])
    land=isr.union(ashape(cent)).buffer(200).buffer(-200)
    polys=vpolys(cent);dpieces={}
    for i in range(len(cities)):
        if polys[i] is None: continue
        cell=polys[i].intersection(land)
        if cell.is_empty or cell.geom_type not in ('Polygon','MultiPolygon'): continue
        m=cid==i;dloc=a[m];wloc=data['W'][m];uds=[int(x) for x in np.unique(dloc)]
        if len(uds)==1:
            dpieces.setdefault(uds[0],[]).append(cell); continue
        vs=[float(wloc[dloc==d].sum()) for d in uds]
        try:
            for d,g in split_cell(cell,uds,vs,dcent).items():
                if not g.is_empty: dpieces.setdefault(int(d),[]).append(g)
        except Exception as ex:
            dpieces.setdefault(uds[int(np.argmax(vs))],[]).append(cell)
    geoms={}
    for d,pcs in dpieces.items():
        g=unary_union(pcs)
        if not g.is_valid: g=g.buffer(0)
        geoms[d]=g
    geoms=cleanup_fragments(geoms)
    feats=[]
    for d in range(n):
        m=a==d
        if not m.any() or d not in geoms: continue
        pv=data['V'][m].sum(0);tot=int(pv.sum());wp=data['parties'][int(np.argmax(pv))]
        bl={b:int(sum(pv[idx[p]] for p in ps if p in idx)) for b,ps in BLOC4.items()}
        wb=max(bl,key=bl.get);srt=sorted(pv,reverse=True)
        pp={p:round(100*pv[idx[p]]/tot,1) for ps in BLOC4.values() for p in ps if p in idx and pv[idx[p]]>0}
        cs=np.unique(data['city'][m])
        cw=sorted(((c,int(data['W'][m&(data['city']==c)].sum())) for c in cs),key=lambda x:-x[1])
        names=[c for c,_ in cw]
        g=transform(lambda x,y:T.transform(x,y),geoms[d])
        gm=mapping(g);gm['coordinates']=rnd(gm['coordinates'])
        feats.append({"type":"Feature","properties":{"d":d,"v":tot,"wp":wp,"wb":wb,
            "m":round(100*(srt[0]-srt[1])/tot,1),"b":[round(100*bl[x]/tot) for x in BLOC4],
            "pp":pp,"s":names[:14],"sn":len(names)},"geometry":gm})
    # overlap check
    ov=(sum(g.area for g in geoms.values())-unary_union(list(geoms.values())).area)/unary_union(list(geoms.values())).area*100
    return {"type":"FeatureCollection","features":feats},ov

def cleanup_fragments(geoms, passes=3):
    from shapely.geometry import MultiPolygon
    for _ in range(passes):
        moved=0
        # snapshot list of district ids
        for d in list(geoms.keys()):
            g=geoms[d]
            if g.geom_type!='MultiPolygon': continue
            parts=sorted(list(g.geoms), key=lambda p:-p.area)
            main=parts[0]; frags=parts[1:]
            keep=[main]
            for fr in frags:
                # find neighbor district sharing longest boundary with fr
                best=None;blen=0
                for e,ge in geoms.items():
                    if e==d: continue
                    if ge.distance(fr)>1e-6: continue
                    inter=ge.buffer(2).intersection(fr.buffer(2)).length
                    if inter>blen: blen=inter;best=e
                if best is not None:
                    geoms[best]=unary_union([geoms[best],fr]).buffer(0)
                    moved+=1
                else:
                    keep.append(fr)  # no neighbor found, keep
            geoms[d]=unary_union(keep).buffer(0) if len(keep)>1 else keep[0]
        if moved==0: break
    return geoms

isr_ll=transform(lambda x,y:T.transform(x,y),isr.simplify(250));om=mapping(isr_ll);om['coordinates']=rnd(om['coordinates'])
o120,ov1=build(120);o215,ov2=build(215)
out={"n120":o120,"n215":o215,"outline":om}
json.dump(out,open('mapdata_new.json','w'),ensure_ascii=False,separators=(',',':'))
import os;print("bytes:",os.path.getsize('mapdata_new.json'),"overlap%:",round(ov1,4),round(ov2,4))
