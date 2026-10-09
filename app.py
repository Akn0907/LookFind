import base64, io, os
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
app = FastAPI(title='LookFind API', version='0.5.0')
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')
ALLOWED = {'image/jpeg','image/png','image/webp'}

CATEGORY_MAP = {
    'top -shirt- t-shirt- blouse-': 'Üst',
    'outerwear -jacket- coat- blazer-': 'Dış Giyim',
    'bottom': 'Alt',
    'dress -one-piece-': 'Elbise',
    'shoe': 'Ayakkabı',
}
ORDER = {'Dış Giyim':0,'Üst':1,'Elbise':1,'Alt':2,'Ayakkabı':3}

def open_image(raw: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    except Exception as exc:
        raise HTTPException(400,'Geçerli bir fotoğraf seçin.') from exc

def jpeg_bytes(img: Image.Image, max_side=1200, max_bytes=490_000):
    img = img.copy(); img.thumbnail((max_side,max_side))
    for q in (88,80,72,64,56,48):
        out=io.BytesIO(); img.save(out,'JPEG',quality=q,optimize=True)
        if len(out.getvalue()) <= max_bytes: return out.getvalue()
    scale=(max_bytes/max(len(out.getvalue()),1))**0.5
    img=img.resize((max(1,int(img.width*scale)),max(1,int(img.height*scale))))
    out=io.BytesIO(); img.save(out,'JPEG',quality=55,optimize=True)
    return out.getvalue()

def normalize_product(item):
    price=item.get('price')
    if isinstance(price,dict):
        ptxt=price.get('value') or ''; pval=price.get('extracted_value'); currency=price.get('currency')
    else:
        ptxt=price or ''; pval=item.get('extracted_price'); currency=None
    return {'title':item.get('title') or 'Ürün','source':item.get('source') or 'Mağaza','link':item.get('link') or '',
            'thumbnail':item.get('thumbnail') or item.get('image') or '', 'price':ptxt,'price_value':pval,'currency':currency,
            'exact':bool(item.get('exact_matches'))}

def lens_search(img: Image.Image, serp_key: str, limit=8):
    payload=jpeg_bytes(img)
    up=requests.post('https://serpapi.com/image',files={'image':('lookfind.jpg',payload,'image/jpeg')},data={'api_key':serp_key},timeout=35)
    up.raise_for_status(); uj=up.json()
    if uj.get('error'): raise RuntimeError(uj['error'])
    image_id=uj.get('image_id')
    if not image_id: raise RuntimeError('SerpApi image_id döndürmedi.')
    r=requests.get('https://serpapi.com/search.json',params={'engine':'google_lens','image_id':image_id,'type':'products','country':'tr','hl':'tr','safe':'active','api_key':serp_key},timeout=50)
    r.raise_for_status(); data=r.json()
    if data.get('error'): raise RuntimeError(data['error'])
    raw=data.get('visual_matches') or data.get('exact_matches') or []
    arr=[normalize_product(x) for x in raw if x.get('link')]
    arr.sort(key=lambda x:(not bool(x['price']), not bool(x['exact'])))
    return arr[:limit]

def _find_prediction_lists(obj):
    """Workflow yanitinda YOLO prediction listelerini bul."""
    found=[]
    if isinstance(obj, dict):
        preds=obj.get('predictions')
        if isinstance(preds, list) and preds and all(isinstance(x, dict) for x in preds):
            if any(('class' in x or 'class_name' in x) and ('x' in x or 'bbox' in x) for x in preds):
                found.append(preds)
        for v in obj.values(): found.extend(_find_prediction_lists(v))
    elif isinstance(obj, list):
        for v in obj: found.extend(_find_prediction_lists(v))
    return found

def detect_clothes(img: Image.Image, rf_key: str, workflow_id: str):
    # Roboflow Serverless Workflow API. Workspace bu prototipte kullanicinin yayinladigi workspace'tir.
    workspace=os.getenv('ROBOFLOW_WORKSPACE','akin-ustabas').strip()
    if not workspace: raise RuntimeError('ROBOFLOW_WORKSPACE bos olamaz.')
    if not workflow_id: raise RuntimeError('ROBOFLOW_MODEL_ID / workflow ID bos olamaz.')
    payload=jpeg_bytes(img,max_side=1280,max_bytes=3_500_000)
    encoded=base64.b64encode(payload).decode('ascii')
    url=f'https://serverless.roboflow.com/{workspace}/workflows/{workflow_id}'
    body={'inputs':{'image':{'type':'base64','value':encoded}}}
    r=requests.post(url,json=body,headers={'Authorization':f'Bearer {rf_key}','Content-Type':'application/json'},timeout=90)
    if r.status_code in (401,403): raise RuntimeError('Roboflow API key veya Workflow erisimi reddedildi.')
    if r.status_code==404: raise RuntimeError(f'Roboflow Workflow bulunamadi: {workspace}/{workflow_id}')
    try: r.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f'Roboflow HTTP {r.status_code}: {r.text[:300]}') from exc
    result=r.json()
    lists=_find_prediction_lists(result)
    preds=max(lists,key=len) if lists else []
    candidates=[]
    for p in preds:
        raw_cls=str(p.get('class') or p.get('class_name') or '').strip().lower()
        cat=None
        if any(k in raw_cls for k in ('shoe','sneaker','boot','footwear')): cat='Ayakkabı'
        elif any(k in raw_cls for k in ('dress','one-piece','gown')): cat='Elbise'
        elif any(k in raw_cls for k in ('jacket','coat','blazer','outerwear')): cat='Dış Giyim'
        elif any(k in raw_cls for k in ('pants','trouser','jeans','shorts','skirt','bottom')): cat='Alt'
        elif any(k in raw_cls for k in ('shirt','t-shirt','tshirt','polo','top','blouse','sweater','sweatshirt','hoodie')): cat='Üst'
        conf=float(p.get('confidence') or p.get('score') or 0)
        if not cat or conf < 0.20: continue
        if all(k in p for k in ('x','y','width','height')):
            x,y,w,h=[float(p.get(k,0)) for k in ('x','y','width','height')]
            left=x-w/2; top=y-h/2; right=x+w/2; bottom=y+h/2
        elif isinstance(p.get('bbox'), dict):
            b=p['bbox']; left=float(b.get('x',0)); top=float(b.get('y',0)); w=float(b.get('width',0)); h=float(b.get('height',0)); right=left+w; bottom=top+h
        else: continue
        w=right-left; h=bottom-top
        if w<20 or h<20: continue
        pad=.06
        box=[max(0,int(left-w*pad)),max(0,int(top-h*pad)),min(img.width,int(right+w*pad)),min(img.height,int(bottom+h*pad))]
        if box[2]<=box[0] or box[3]<=box[1]: continue
        candidates.append({'category':cat,'confidence':conf,'box':box,'raw_class':raw_cls})
    # V0.4.2: kategori basina en guvenli tek kutu. Iki ayakkabidan en guvenli olanini aratir.
    best={}
    for c in candidates:
        if c['category'] not in best or c['confidence']>best[c['category']]['confidence']: best[c['category']]=c
    if 'Elbise' in best:
        best.pop('Üst',None); best.pop('Alt',None)
    return sorted(best.values(),key=lambda x:ORDER.get(x['category'],9))[:4]

@app.get('/')
def home(): return FileResponse(BASE_DIR/'static'/'index.html')

@app.get('/api/health')
def health():
    return {'ok':True,'serpapi_configured':bool(os.getenv('SERPAPI_API_KEY')),'roboflow_configured':bool(os.getenv('ROBOFLOW_API_KEY'))}

@app.post('/api/analyze')
async def analyze(image: UploadFile=File(...)):
    serp=os.getenv('SERPAPI_API_KEY','').strip(); rf=os.getenv('ROBOFLOW_API_KEY','').strip(); model=os.getenv('ROBOFLOW_MODEL_ID','object-detection-shop-anyting/fashion-items-ew8tn-instant-3').strip()
    if not serp or serp.startswith('BURAYA_'): raise HTTPException(500,'SERPAPI_API_KEY ayarlanmamış.')
    if not rf or rf.startswith('BURAYA_'): raise HTTPException(500,'ROBOFLOW_API_KEY ayarlanmamış. .env dosyasına Roboflow API key ekleyin.')
    if image.content_type not in ALLOWED: raise HTTPException(400,'Yalnızca JPG, PNG veya WebP kullanın.')
    raw=await image.read()
    if not raw or len(raw)>15*1024*1024: raise HTTPException(400,'Fotoğraf boş veya 15 MB sınırını aşıyor.')
    img=open_image(raw)
    try:
        detections=detect_clothes(img,rf,model)
    except Exception as exc:
        raise HTTPException(502,f'Kıyafet tespiti hatası: {exc}') from exc
    if not detections:
        raise HTTPException(422,'Bu fotoğrafta desteklenen kıyafet parçası tespit edilemedi. Kişinin ve kıyafetlerin net göründüğü başka bir fotoğraf deneyin.')
    groups=[]
    for d in detections:
        crop=img.crop(tuple(d['box']))
        try: products=lens_search(crop,serp,8)
        except Exception as exc: products=[]
        groups.append({'category':d['category'],'confidence':round(d['confidence']*100),'box':d['box'],'products':products})
    def valid_price(p):
        v=p.get('price_value')
        return isinstance(v,(int,float)) and v > 0 and p.get('link')

    # 1) En ucuz sepet: her kategori için fiyatı olan en ucuz görsel eşleşme.
    cheapest_cart=[]; cheapest_total=0.0
    for g in groups:
        priced=[p for p in g['products'] if valid_price(p)]
        pick=min(priced,key=lambda p:float(p['price_value'])) if priced else None
        if pick:
            cheapest_cart.append({'category':g['category'],**pick})
            cheapest_total += float(pick['price_value'])

    # 2) Tek platform sepetleri: bütün tespit edilen kategorilerin aynı source/mağazada
    # fiyatlı sonucu varsa, o mağaza için kategori başına en ucuz sonucu seç.
    # Source adı SerpApi'den geldiği için kullanıcıya olduğu gibi gösterilir.
    source_maps=[]
    for g in groups:
        m={}
        for p in g['products']:
            if not valid_price(p): continue
            source=(p.get('source') or '').strip()
            if not source: continue
            key=source.casefold()
            if key not in m or float(p['price_value']) < float(m[key]['price_value']):
                m[key]=p
        source_maps.append((g['category'],m))

    single_platform_carts=[]
    if source_maps:
        common=set(source_maps[0][1])
        for _,m in source_maps[1:]: common &= set(m)
        for key in common:
            items=[]; total=0.0
            for category,m in source_maps:
                p=m[key]; items.append({'category':category,**p}); total += float(p['price_value'])
            single_platform_carts.append({'platform':items[0]['source'],'items':items,'total_value':round(total,2)})
        single_platform_carts.sort(key=lambda c:c['total_value'])

    # Geriye dönük cart alanı da kalsın; V0.5 arayüzü yeni alanları kullanır.
    return {'detected_count':len(groups),'groups':groups,
            'cart':cheapest_cart,'total_value':round(cheapest_total,2),
            'cheapest_cart':cheapest_cart,'cheapest_total':round(cheapest_total,2),
            'single_platform_carts':single_platform_carts[:5],
            'currency':'TRY','model':model,
            'note':'V0.5: İki sepet oluşturulur: aynı mağazada tüm parçalar bulunursa Tek Platform; mağaza fark etmeksizin kategori başına en düşük fiyatlı sonuçlarla En Ucuz Sepet.'}
