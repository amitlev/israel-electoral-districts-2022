import csv, json, os, numpy as np
from scipy.spatial import cKDTree, Delaunay
from collections import deque, defaultdict
ROOT=os.path.dirname(os.path.abspath(__file__))
BASE=os.path.join(ROOT,"data")
CONF=json.load(open(os.path.join(ROOT,"elections.json"),encoding='utf-8'))
ENVELOPES=9999   # סמל ישוב of the double-envelope ("מעטפות חיצוניות") pseudo-settlement

def election(eid=None):
    """Config of one election from elections.json (default: the most recent one)."""
    eid=str(eid or CONF["default"]); el=dict(CONF["elections"][eid]); el["id"]=eid; return el

def names(eid=None): return {p["code"]:p["name"] for p in election(eid)["parties"]}
NAMES=names()

def load(eid=None):
    el=election(eid)
    coords={}
    for r in csv.DictReader(open(f"{BASE}/settlement_coords.csv",encoding='cp1255')):
        try: coords[int(r['סמל ישוב'])]=(float(r['אורדינטה מזרחית']),float(r['אורדינטה צפונית']))
        except: pass
    rows=list(csv.reader(open(os.path.join(ROOT,el["ballots"]),encoding='utf-8-sig')))
    hdr=rows[0]; ki=hdr.index('כשרים')
    # party columns = everything after 'כשרים' except per-election bookkeeping columns
    skip=set(el.get("non_party_columns",[]))
    pcols=[i for i in range(ki+1,len(hdr)) if hdr[i] not in skip]; parties=[hdr[i] for i in pcols]
    si=hdr.index('סמל ישוב'); ni=hdr.index('שם ישוב')
    X=[];W=[];V=[];CITY=[]
    st=dict(valid=0,located=0,envelopes=0,unlocated_other=0,ballots=0,mismatch_rows=0)
    for r in rows[1:]:
        if len(r)<len(hdr): continue
        sym=int(r[si]); w=int(r[ki]); v=[int(r[i] or 0) for i in pcols]
        st['valid']+=w
        if sum(v)!=w: st['mismatch_rows']+=1
        if sym not in coords or w<=0:
            st['envelopes' if sym==ENVELOPES else 'unlocated_other']+=w; continue
        st['located']+=w; st['ballots']+=1
        X.append(coords[sym]);W.append(w);CITY.append(r[ni]);V.append(v)
    if st['mismatch_rows']:
        raise ValueError(f"{el['ballots']}: {st['mismatch_rows']} rows where party columns != כשרים "
                         f"(a non-party column is probably missing from non_party_columns)")
    X=np.array(X,float);W=np.array(W,float);V=np.array(V,float)
    tot=V.sum(0);keep=tot/tot.sum()>=0.01
    Pk=V[:,keep];P=Pk/np.maximum(Pk.sum(1,keepdims=True),1)
    RIGHT=set(el["lean"]["right"]); LEFT=set(el["lean"]["left"])
    ci=np.array([1.0 if p in RIGHT else(-1.0 if p in LEFT else 0)for p in parties])
    score=(V*ci).sum(1)/np.maximum(V.sum(1),1)
    st['settlements']=len(set(CITY))
    return dict(X=X,W=W,V=V,P=P,parties=parties,city=np.array(CITY),score=score,election=el,stats=st)

def rcb(idx,k,data,alpha):
    if k==1: return [idx]
    X=data['X'][idx];W=data['W'][idx];sc=data['score'][idx]
    kL=k//2;kR=k-kL;frac=kL/k
    Xc=X-X.mean(0)
    # principal axis (compact cut)
    u=np.linalg.svd(Xc*W[:,None]**0.5,full_matrices=False)[2][0]
    proj=Xc@u
    proj=(proj-proj.mean())/(proj.std()+1e-9)
    scz=(sc-sc.mean())/(sc.std()+1e-9)
    key=proj+alpha*scz                    # alpha tilts cut toward political homogeneity
    order=np.argsort(key,kind='stable')
    cw=np.cumsum(W[order]);cut=min(max(np.searchsorted(cw,frac*W.sum())+1,1),len(idx)-1)
    return rcb(idx[order[:cut]],kL,data,alpha)+rcb(idx[order[cut:]],kR,data,alpha)

def repair(assign,data,n,cadj,cid):
    # reassign disconnected city-fragments to a bordering district (largest border)
    for d in range(n):
        cs=set(cid[assign==d].tolist())
        if len(cs)<=1: continue
        # components among cities of d
        comp={};ci=0
        for start in cs:
            if start in comp: continue
            dq=deque([start]);comp[start]=ci
            while dq:
                u=dq.popleft()
                for v in cadj[u]:
                    if v in cs and v not in comp: comp[v]=ci;dq.append(v)
            ci+=1
        if ci<=1: continue
        # keep the component with most voters; others -> neighbor district
        sizes=defaultdict(float)
        for c in cs: sizes[comp[c]]+=data['W'][(cid==c)&(assign==d)].sum()
        keep=max(sizes,key=sizes.get)
        for c in cs:
            if comp[c]==keep: continue
            nb=[assign[m] for v in cadj[c] for m in np.where(cid==v)[0] if assign[m]!=d]
            if nb:
                tgt=max(set(nb),key=nb.count)
                mask=(cid==c)&(assign==d); assign[mask]=tgt
    return assign

def city_adjacency(data):
    cities=np.unique(data['city']);cmap={c:i for i,c in enumerate(cities)}
    cid=np.array([cmap[c] for c in data['city']])
    cent=np.array([data['X'][cid==i].mean(0) for i in range(len(cities))])
    tri=Delaunay(cent);cadj=[set() for _ in cities]
    for s in tri.simplices:
        for a in s:
            for b in s:
                if a!=b:cadj[a].add(int(b))
    return cid,cadj

def solve(data,n,alpha=0.0,do_repair=True):
    parts=rcb(np.arange(len(data['W'])),n,data,alpha)
    assign=np.empty(len(data['W']),int)
    for d,p in enumerate(parts):assign[p]=d
    if do_repair:
        cid,cadj=city_adjacency(data);assign=repair(assign,data,n,cadj,cid)
    return assign

def metrics(data,assign,n,cid=None,cadj=None):
    W,V,parties=data['W'],data['V'],data['parties']
    wins={};loads=[];ws=[]
    for d in range(n):
        m=assign==d
        if not m.any(): continue
        vd=V[m].sum(0);t=vd.sum();loads.append(W[m].sum())
        k=int(np.argmax(vd));wins[parties[k]]=wins.get(parties[k],0)+1;ws.append(vd[k]/t)
    loads=np.array(loads)
    bad=0
    if cid is not None:
        for d in range(n):
            cs=set(cid[assign==d].tolist())
            if len(cs)<=1: continue
            l=list(cs);seen={l[0]};dq=deque(seen)
            while dq:
                u=dq.popleft()
                for v in cadj[u]:
                    if v in cs and v not in seen:seen.add(v);dq.append(v)
            if len(seen)<len(cs):bad+=1
    return dict(wins=wins,dev=(loads.max()-loads.min())/loads.mean()*100,ws=float(np.mean(ws)),noncontig=bad)

if __name__=="__main__":
    import sys
    data=load(sys.argv[1] if len(sys.argv)>1 else None);cid,cadj=city_adjacency(data)
    NAMES=names(data['election']['id'])
    print(data['election']['title'],"| located voters:",int(data['W'].sum()),"|",data['stats'])
    for n in [12,30,60,120]:
        for alpha in [0.0,0.5,1.0]:
            a=solve(data,n,alpha=alpha)
            m=metrics(data,a,n,cid,cadj)
            top=" ".join(f"{NAMES.get(p,p)}:{c}" for p,c in sorted(m['wins'].items(),key=lambda x:-x[1])[:4])
            print(f"n={n:3d} α={alpha:.1f}| size±={m['dev']:3.0f}% winShare={m['ws']:.1%} nonContig={m['noncontig']:2d}/{n} | {top}")
