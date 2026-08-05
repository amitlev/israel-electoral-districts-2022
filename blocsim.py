import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from bidi.algorithm import get_display
import engine as E
data=E.load()
def H(s): return get_display(s)
P=data['parties']

BLOC4={'ימין':['מחל','ט'],'מרכז-שמאל':['פה','כן','אמת','מרצ','ל'],
       'חרדי':['שס','ג'],'ערבי':['ום','עם','ד']}
BLOC5={'ימין':['מחל','ט'],'מרכז-שמאל':['פה','כן','אמת','מרצ'],
       'חרדי':['שס','ג'],'ערבי':['ום','עם','ד'],'ישראל ביתנו':['ל']}
COL={'ימין':'#1f6fb2','מרכז-שמאל':'#eb6834','חרדי':'#111111',
     'ערבי':'#1e8449','ישראל ביתנו':'#d4a017'}

def bloc_matrix(scheme):
    names=list(scheme); idx={p:i for i,p in enumerate(P)}
    M=np.zeros((len(data['V']),len(names)))
    for j,b in enumerate(names):
        for p in scheme[b]:
            if p in idx: M[:,j]+=data['V'][:,idx[p]]
    return names,M

def simulate(n,scheme,assign):
    names,M=bloc_matrix(scheme)
    seats={b:0 for b in names}; valid=data['V'].sum(); got=0
    dwin={}
    for d in range(n):
        vd=M[assign==d].sum(0); k=int(np.argmax(vd)); seats[names[k]]+=1; dwin[d]=names[k]
        got+=vd[k]
    return names,seats,dwin,100*got/valid

def voronoi_polys(points):
    mn=points.min(0)-50000;mx=points.max(0)+50000
    extra=np.array([[mn[0],mn[1]],[mn[0],mx[1]],[mx[0],mn[1]],[mx[0],mx[1]]])
    vor=Voronoi(np.vstack([points,extra]));polys=[]
    for i in range(len(points)):
        reg=vor.regions[vor.point_region[i]]
        polys.append(None if(not reg or -1 in reg) else Polygon([vor.vertices[j] for j in reg]))
    return polys

def make_map(n,scheme,fname,title):
    assign=E.solve(data,n,alpha=0.0,do_repair=False)
    names,seats,dwin,rep=simulate(n,scheme,assign)
    cities=np.unique(data['city']);cmap={c:i for i,c in enumerate(cities)}
    cid=np.array([cmap[c] for c in data['city']])
    cent=np.array([data['X'][cid==i].mean(0) for i in range(len(cities))])
    s_dist=np.empty(len(cities),int)
    for i in range(len(cities)):
        m=cid==i;ds=np.unique(assign[m]);wv=[data['W'][m&(assign==dd)].sum() for dd in ds]
        s_dist[i]=ds[int(np.argmax(wv))]
    polys=voronoi_polys(cent);hull=unary_union([Point(*p).buffer(9000) for p in cent])
    fig,ax=plt.subplots(figsize=(7.2,10.5),dpi=115);dp={}
    for i,poly in enumerate(polys):
        if poly is None: continue
        cell=poly.intersection(hull)
        if cell.is_empty: continue
        b=dwin[s_dist[i]]
        for g in (cell.geoms if isinstance(cell,MultiPolygon) else [cell]):
            xs,ys=g.exterior.xy;ax.fill(xs,ys,color=COL[b],ec='white',lw=0.15)
        dp.setdefault(int(s_dist[i]),[]).append(cell)
    for d,cells in dp.items():
        u=unary_union(cells)
        for g in (u.geoms if isinstance(u,MultiPolygon) else [u]):
            if not g.is_empty: xs,ys=g.exterior.xy;ax.plot(xs,ys,color='white',lw=0.6)
    order=sorted(names,key=lambda b:-seats[b])
    handles=[mpatches.Patch(color=COL[b],label=H(f"{b} ({seats[b]})")) for b in order]
    ax.legend(handles=handles,loc='lower left',fontsize=9,frameon=False,handlelength=1.1)
    ax.set_aspect('equal');ax.axis('off')
    ax.set_title(H(title),fontsize=12,fontfamily='DejaVu Sans')
    plt.tight_layout();plt.savefig(fname,bbox_inches='tight');plt.close()
    return seats,rep

for scheme,tag,label in [(BLOC4,'4bloc','4 גושים'),(BLOC5,'5bloc','5 גושים (ליברמן נישה)')]:
    for n in [120,215]:
        s,rep=make_map(n,scheme,f"bloc_{tag}_n{n}.png",f"{n} מחוזות · {label} · הגוש הזוכה בכל מחוז")
        print(f"[{tag} n={n}] rep={rep:.1f}%  "+"  ".join(f"{b}:{c}" for b,c in sorted(s.items(),key=lambda x:-x[1])))
