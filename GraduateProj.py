import textwrap

try:
    import torch
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
import uuid
import random
import os


try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'

ENABLE_XFORMERS = globals().get('ENABLE_XFORMERS', True)
ENABLE_VAE_TILING = globals().get('ENABLE_VAE_TILING', True)
DISABLE_VAE_SLICING = globals().get('DISABLE_VAE_SLICING', True)
ENABLE_TORCH_COMPILE = globals().get('ENABLE_TORCH_COMPILE', False)
COMPILE_MODE = globals().get('COMPILE_MODE', 'reduce-overhead')
USE_HALF = globals().get('USE_HALF', (DEVICE == 'cuda'))

def _wrap_vae_decode_fp32(vae):
    try:
        orig = vae.decode
    except Exception:
        return
    def decode_fp32(z, *a, **kw):
        try:
            return orig(z.float(), *a, **kw)
        except Exception:
            return orig(z, *a, **kw)
    try:
        vae.decode = decode_fp32
    except Exception:
        pass

def _apply_cuda_half_optimizations(pipe):
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    if ENABLE_XFORMERS:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    if ENABLE_VAE_TILING:
        try: pipe.enable_vae_tiling()
        except Exception: pass
    if DISABLE_VAE_SLICING:
        try: pipe.disable_vae_slicing()
        except Exception:
            try: pipe.enable_vae_slicing()
            except Exception: pass

    try:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    try:
        import torch
        dtype = torch.float16 if USE_HALF else torch.float32
        device = 'cuda' if (USE_HALF and torch.cuda.is_available()) else 'cpu'
    except Exception:
        dtype = None; device = 'cpu'

    try: pipe.to(device)
    except Exception: pass

    try: pipe.unet.to(dtype)
    except Exception: pass
    for name in ('text_encoder','text_encoder_2'):
        te = getattr(pipe, name, None)
        if te is not None:
            try: te.to(dtype)
            except Exception: pass

    cn = getattr(pipe, 'controlnet', None)
    if cn is not None:
        try: cn.to(dtype)
        except Exception: pass

    try: pipe.vae.to(dtype)
    except Exception: pass
    try: _wrap_vae_decode_fp32(pipe.vae)
    except Exception: pass

    if USE_HALF and ENABLE_TORCH_COMPILE:
        try:
            pipe.unet = torch.compile(pipe.unet, mode=COMPILE_MODE, fullgraph=False)
            print('[v18l hotfix] UNet compiled')
        except Exception as e:
            print('[v18l hotfix] torch.compile skipped:', e)

    try:
        print('[v18l hotfix] device:', device, '| half:', USE_HALF)
        print('[v18l hotfix] dtypes -> unet:', str(pipe.unet.dtype), 'vae:', pipe.vae.dtype)
        print('[v18l hotfix] devices -> unet:', pipe.unet.device, 'vae:', pipe.vae.device)
    except Exception as _e:
        print('[v18l hotfix] dtype/device print skipped:', _e)
    return pipe

os.environ.setdefault("PYTHONUNBUFFERED","1")

import os, uuid, random, re, json, contextlib, inspect, unicodedata, itertools
from typing import List, Optional, Tuple, Dict
from collections import Counter

import numpy as np
from PIL import Image

import torch
def _v16_coerce_float32_cuda(pipe):
    try:
        pipe.to("cuda")
    except Exception:
        pass
    try:
        pipe.unet.to(torch.float32)
    except Exception:
        pass
    try:
        pipe.vae.to(torch.float32)
    except Exception:
        pass
    for name in ("text_encoder", "text_encoder_2"):
        try:
            te = getattr(pipe, name, None)
            if te is not None:
                te.to(torch.float32)
        except Exception:
            pass
    try:
        if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
            pipe.controlnet.to("cuda")
            pipe.controlnet.to(torch.float32)
    except Exception:
        pass
    return pipe

from skimage.metrics import structural_similarity as ssim

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor, pipeline as hf_pipeline
from diffusers import (
    StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline,
    DPMSolverMultistepScheduler, ControlNetModel
)
from diffusers import EulerAncestralDiscreteScheduler


GA_SCORE_ALPHA = globals().get("GA_SCORE_ALPHA", 0.50)
GA_SCORE_BETA  = globals().get("GA_SCORE_BETA",  0.50)

GA_POP_SIZE  = globals().get("GA_POP_SIZE", 15)
GA_ELITISM   = globals().get("GA_ELITISM", 2)
GA_MUT_RATE  = globals().get("GA_MUT_RATE", 0.15)


GA_MAX_GENS  = globals().get("GA_MAX_GENS", 12)
GA_PATIENCE  = globals().get("GA_PATIENCE", 4)


try:
    from controlnet_aux import HEDdetector
    HAVE_HED = True
except Exception:
    HEDdetector = None
    HAVE_HED = False
try:
    from controlnet_aux import PidiNetDetector
    HAVE_SOFTEDGE = True
except Exception:
    PidiNetDetector = None
    HAVE_SOFTEDGE = False

ORIGINAL_IMAGE_PATH = r"image/낫1.png"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_dir = OUTPUT_DIR

PROMPTBANK_PATH = r"C:\Users\admin\PycharmProjects\pythonProject6\promptbank\diffusiondb_prompts1.txt"
PROMPTBANK_TOPN = 80
PROMPTBANK_TOKENS_MAX = 60
PROMPTBANK_DEBUG = False

GUIDANCE_SCALE = 6.0
PROXY_W, PROXY_H, PROXY_EFF_STEPS = 512, 512, 16
FULL_W,  FULL_H,  FULL_EFF_STEPS  = 512, 512, 25

SSIM_THRESHOLD = 0.1

PROXY_W, PROXY_H = 640, 640
PROXY_STEPS = 12
PROXY_GUIDANCE = 4.5
PROXY_CN_END = 0.45
PROXY_USE_IPA = True
PROXY_DISABLE_TQDM = True
PROXY_SAVE_FAIL = False


CLIP_THRESHOLD = 80.0
MAX_RETRIES = 1
GA_TARGET_COUNT = 8
GA_RANDOM_SEED = None
GA_COMB_SAMPLE_LIMIT: Optional[int] = None
PROXY_STRICT = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
import torch
def _v16_coerce_float32_cuda(pipe):
    try:
        pipe.to("cuda")
    except Exception:
        pass
    try:
        pipe.unet.to(torch.float32)
    except Exception:
        pass
    try:
        pipe.vae.to(torch.float32)
    except Exception:
        pass
    for name in ("text_encoder", "text_encoder_2"):
        try:
            te = getattr(pipe, name, None)
            if te is not None:
                te.to(torch.float32)
        except Exception:
            pass
    try:
        if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
            pipe.controlnet.to("cuda")
            pipe.controlnet.to(torch.float32)
    except Exception:
        pass
    return pipe
import torch._dynamo
torch._dynamo.config.suppress_errors = True

PIPE_TXT2IMG = None
_CLIP_MODEL = _CLIP_TOK = _CLIP_IMG = None
_LLM_VLM = None
_PROMPTBANK_LINES: Optional[List[str]] = None

SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNET_SOFTEDGE_CANDIDATES = [
    "thibaud/controlnet-sdxl-1.0-softedge","diffusers/controlnet-softedge-sdxl-1.0",
]
CONTROLNET_HED_CANDIDATES = [
    "diffusers/controlnet-hed-sdxl-1.0","xinsir/controlnet-hed-sdxl-1.0","thibaud/controlnet-sdxl-1.0-hed",
]
CONTROLNET_CANNY_CANDIDATES = [
    "diffusers/controlnet-canny-sdxl-1.0","xinsir/controlnet-canny-sdxl-1.0","thibaud/controlnet-sdxl-1.0-canny",
]
IPADAPTER_REPO_CANDIDATES = [
    ("h94/IP-Adapter","models","ip-adapter_sdxl.safetensors"),
    ("h94/IP-Adapter","sdxl_models","ip-adapter_sdxl.safetensors"),
    ("h94/IP-Adapter","models","ip-adapter_sdxl.bin"),
]

_PREP_WORDS = {"of","with","and","by","for","to"}
BANNED_NOUN_ROOTS = {"dungeon","dragon","dragoon","face","faces","english","human","object","objects",
                     "fate","metaexotic","resolution","quality","photo","photograph","hydrant","bow"}
STOPWORDS = {"a","an","the","this","that","these","those","and","or","with","of","on","in","by","for","to",
             "from","at","as","into","is","are","be","being","been","it","its","his","her","their","your","our",
             "up","down","left","right","top","bottom","front","back","side","center","centre",
             "close","closeup","close-up","zoom","macro","angle","shot","view","there","person"}
BLOCKLIST_META = {"image","photo","picture","screenshot","render","rendering","graphic","concept art","icon",
                  "inventory item","ui icon","game art","asset","sprite","thumbnail","background","shadow",
                  "lighting","light","rim light","studio lighting","vray","octane","cycles","arnold","ue","ue5",
                  "unreal","unreal engine","unity","blender","maya","max","c4d","3ds","sdxl","stable diffusion",
                  "camera","bokeh","depth of field","hdr","uhd","fhd","ultra detail","ultra details","full body",
                  "rpg","inventory","item","asset","inspired","blizzard","disney","pixar","ghibli","nintendo","marvel","dc"}
GENERIC_WEAK = {"artifact","artifacts","object","device","equipment","gear","thing","stuff","tool"}
BLOCKLIST = {"text","logo","watermark","cancer","metaexotic"}
STOP_GEOM = {"cylinder","sphere","cube","cone","pyramid","torus","capsule","prism"}
_RE_HAS_DIGIT = re.compile(r"\d")
_RE_UNITS = re.compile(r"\b(mm|cm|m|inch|in|px|k)\b")
_RES_TERMS = {"4k","8k","16k"}
META_CONTAINS = {"image","photo","picture","screenshot","render","rendering","graphic","art","concept",
                 "icon","inventory","item","sprite","thumbnail","game","rpg","loot","style","stylized","design",
                 "artifact","artifacts","meta","exotic"}

LEX_STYLE_ROOTS = {"vector","clean","flat","outline","stylized","cartoon","painterly","minimal","silhouette","edges"}
LEX_LIGHT_ROOTS = {"bright","soft","softshadow","softshadows","softshadowed","rim","backlight","glow","shadow"}
LEX_MATERIAL_ROOTS = {"metal","iron","steel","bronze","brass","gold","silver","leather","wood","stone","glass","cloth"}
LEX_LAYOUT_ROOTS = {"center","centered","framing","focus","frontal"}
LEX_BG_ALLOWED = {"white background","dark background","plain background","clean background"}

OBJ_SYNONYM_MAP = {
    "book":{"book","tome","grimoire","spellbook","manuscript","codex","volume","cover"},
    "helmet":{"helmet","helm","headgear"},
    "sword":{"sword","blade","sabre","katana"},
    "shield":{"shield","buckler"},
    "ring":{"ring","band"},
    "staff":{"staff","rod","scepter","wand"},
    "potion":{"potion","vial","bottle","flask"},
    "scroll":{"scroll","parchment"},
}
OBJ_PARTS = {
    "_generic":["surface","rim","band","ornament","inlay","emblem","seal","crest","engraving","pattern","motif","trim"],
    "book":["cover","spine","pages","corner guards","clasp","binding","emboss","corner plates","bookmark"],
    "helmet":["shell","horns","straps","visor","cheek guards","rim","crest","nose guard","rivets"],
    "sword":["blade","hilt","guard","pommel","fuller","edge","scabbard"],
    "shield":["boss","rim","straps","face","studs","crest"],
    "ring":["band","setting","gem","inscription"],
    "staff":["shaft","head","ornament","gem","ring"],
    "potion":["glass","stopper","label","liquid"],
    "scroll":["parchment","seal","ribbon","handle"],
}
MATERIAL_WORDS = ["leather","wood","iron","steel","bronze","brass","gold","silver","stone","glass","cloth"]
COLOR_WORDS = ["red","orange","yellow","green","cyan","blue","purple","magenta","brown","black","gray","grey","white"]

TARGET_PROMPT_COUNT_MIN = 14
TARGET_PROMPT_COUNT_MAX = 15

def load_promptbank_lines(path: str) -> List[str]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path): return []
    lines: List[str] = []
    if os.path.isdir(abs_path):
        for root,_,files in os.walk(abs_path):
            for name in files:
                if not name.lower().endswith((".txt",".jsonl")): continue
                fp = os.path.join(root,name)
                try:
                    with open(fp,"r",encoding="utf-8") as f:
                        for ln in f:
                            ln=ln.strip()
                            if ln: lines.append(ln)
                except UnicodeDecodeError:
                    with open(fp,"r",encoding="utf-8",errors="ignore") as f:
                        for ln in f:
                            ln=ln.strip()
                            if ln: lines.append(ln)
    else:
        try:
            with open(abs_path,"r",encoding="utf-8") as f:
                for ln in f:
                    ln=ln.strip()
                    if ln: lines.append(ln)
        except UnicodeDecodeError:
            with open(abs_path,"r",encoding="utf-8",errors="ignore") as f:
                for ln in f:
                    ln=ln.strip()
                    if ln: lines.append(ln)
    return lines

def pb_search(lines: List[str], queries: List[str], topn: int = 100) -> List[str]:
    if not lines or not queries: return []
    q = " ".join([q for q in queries if q]).lower().split()
    q = list(dict.fromkeys(q))
    scored=[]
    for s in lines:
        t=s.lower()
        score=sum(1 for token in q if token and token in t)
        if score>0: scored.append((score,s))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [s for _,s in scored[:topn]]

def load_clip(device: str):
    global _CLIP_MODEL,_CLIP_TOK,_CLIP_IMG
    if _CLIP_MODEL is None:
        MODEL_ID="openai/clip-vit-base-patch32"
        _CLIP_MODEL=CLIPModel.from_pretrained(MODEL_ID).to(device)
        _CLIP_TOK=CLIPTokenizer.from_pretrained(MODEL_ID)
        _CLIP_IMG=CLIPImageProcessor.from_pretrained(MODEL_ID)
    return _CLIP_MODEL,(_CLIP_TOK,_CLIP_IMG)

def compare_with_transformers(p1: str, p2: str, device: str=DEVICE) -> float:
    load_clip(device)
    imgs=[Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")]
    batch=_CLIP_IMG(images=imgs, return_tensors="pt")
    px=batch["pixel_values"].to(device)
    with torch.no_grad():
        emb=_CLIP_MODEL.get_image_features(pixel_values=px)
    emb=emb/emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return float(torch.cosine_similarity(emb[0:1], emb[1:2]).item()*100.0)

def _clip_image_feat(pil: Image.Image) -> torch.Tensor:
    load_clip(DEVICE)
    px=_CLIP_IMG(images=pil, return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.no_grad():
        vf=_CLIP_MODEL.get_image_features(pixel_values=px)
    vf=vf/vf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return vf.squeeze(0)

def _clip_text_embeds(phrases: List[str]) -> Dict[str, torch.Tensor]:
    if not phrases: return {}
    load_clip(DEVICE)
    tok=_CLIP_TOK(phrases, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        tf=_CLIP_MODEL.get_text_features(
            input_ids=tok["input_ids"].to(DEVICE),
            attention_mask=tok["attention_mask"].to(DEVICE)
        )
    tf=tf/tf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return {p:e.detach().cpu() for p,e in zip(phrases, tf)}

def _clip_score_phrases(pil: Image.Image, phrases: List[str]) -> Dict[str,float]:
    if not phrases: return {}
    embeds=_clip_text_embeds(phrases)
    vf=_clip_image_feat(pil)
    out={}
    for p,e in embeds.items():
        e=e.to(vf.dtype)
        out[p]=float((e @ vf.cpu()) / (max(1e-6, torch.norm(e)*torch.norm(vf.cpu()))))
    return out

def _load_llava():
    global _LLM_VLM
    if _LLM_VLM is None:
        dev=0 if DEVICE=="cuda" else -1
        try:
            _LLM_VLM=hf_pipeline("image-text-to-text","llava-hf/llava-1.6-vicuna-7b-hf",device=dev)
        except Exception:
            _LLM_VLM=hf_pipeline("image-to-text","Salesforce/blip-image-captioning-large",device=dev)
    return _LLM_VLM

_LLaVA_SYS = (
    "You are tagging a stylized game OBJECT. Return compact JSON with short nouns and object parts only:\n"
    "{\"base\":[],\"shape\":[],\"material\":[],\"view\":\"\",\"lighting\":[],\"detail\":[],\"style\":[],\"background\":\"\",\"negative\":[]}\n"
    "Rules: NO people/places/meta; nouns 1–3 words; avoid 'there/person'."
)

def llava_extract_tags(pil: Image.Image) -> dict:
    vlm=_load_llava()
    try:
        task=getattr(vlm,"task","")
        if task=="image-text-to-text":
            out=vlm({"image":pil,"prompt":_LLaVA_SYS}, max_new_tokens=220, temperature=0.2, top_p=0.2)[0]["generated_text"]
        else:
            out=vlm(pil, max_new_tokens=60)[0]["generated_text"]
            kws=[t.strip(".,!?:;()[]{}'\"").lower() for t in out.split() if len(t)>2]
            return {"base":kws[:5],"shape":[],"material":[],"view":"","lighting":[],"detail":[],
                    "style":["digital illustration"],"background":"","negative":[]}
        m=re.search(r'\{.*\}', out, re.S)
        if not m: return {}
        return json.loads(m.group(0))
    except Exception:
        return {}

def llava_object_candidates_only(pil: Image.Image, k: int=10) -> List[str]:
    vlm=_load_llava()
    prompts=[
        "Name the OBJECT (not people). 1-2 word nouns only. CSV.",
        "Give object category nouns (1-2 words). CSV. No style/lighting.",
    ]
    out_tokens=[]
    for p in prompts:
        try:
            task=getattr(vlm,"task","")
            if task=="image-text-to-text":
                txt=vlm({"image":pil,"prompt":p}, max_new_tokens=50, temperature=0.2, top_p=0.2)[0]["generated_text"]
            else:
                txt=vlm(pil, max_new_tokens=40)[0]["generated_text"]
        except Exception:
            continue
        raw=re.split(r"[,\n/|]+", txt.lower())
        for t in raw:
            t=t.strip()
            if not t: continue
            if len(t.split())==0 or len(t.split())>2: continue
            out_tokens.append(t)
    return _clean_short_tokens(out_tokens)[:k]

def _alpha_bbox_crop(pil: Image.Image) -> Image.Image:
    if pil.mode!="RGBA": return pil
    a=np.array(pil.split()[-1]); ys,xs=np.where(a>0)
    if len(xs)==0 or len(ys)==0: return pil.convert("RGB")
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max()
    return pil.crop((x0,y0,x1+1,y1+1)).convert("RGB")

def _largest_contour_crop(pil: Image.Image, margin=0.06) -> Image.Image:
    try: import cv2
    except Exception: return pil
    arr=np.array(pil.convert("RGB")); gray=cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray=cv2.bilateralFilter(gray,7,25,25); edges=cv2.Canny(gray,60,160)
    cnts,_=cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return pil
    cnt=max(cnts, key=cv2.contourArea); x,y,w,h=cv2.boundingRect(cnt)
    if w*h<0.05*arr.shape[0]*arr.shape[1]: return pil
    dx=int(w*margin); dy=int(h*margin)
    x0=max(0,x-dx); y0=max(0,y-dy); x1=min(arr.shape[1],x+w+dx); y1=min(arr.shape[0],y+h+dy)
    return pil.crop((x0,y0,x1,y1))

def _pad_to_square(pil: Image.Image, bg="auto") -> Image.Image:
    w,h=pil.size; side=max(w,h)
    if bg=="auto":
        arr=np.array(pil if pil.mode=="RGB" else pil.convert("RGB"))
        border=np.concatenate([arr[0],arr[-1],arr[:,0],arr[:,-1]],axis=0)
        color=tuple(map(int, border.mean(axis=0)))
    else:
        color=(255,255,255) if bg=="white" else (0,0,0)
    canvas=Image.new("RGB",(side,side),color); canvas.paste(pil,((side-w)//2,(side-h)//2))
    return canvas

def prep_for_recognition(pil: Image.Image, target=768) -> Image.Image:
    im=pil
    if im.mode=="RGBA": im=_alpha_bbox_crop(im)
    base_w,base_h=im.size
    if min(base_w,base_h)<256:
        scale=max(2,int(256/min(base_w,base_h))+1)
        im=im.resize((base_w*scale, base_h*scale), Image.LANCZOS)
    im=_largest_contour_crop(im, margin=0.06); im=_pad_to_square(im, bg="auto")
    if max(im.size)<target: im=im.resize((target,target), Image.LANCZOS)
    return im.convert("RGB")

def _contains_any_word(s: str, vocab: set) -> bool:
    if not s: return False
    return any(w in vocab for w in re.findall(r"[a-z0-9]+", s.lower()))

def _is_ascii_word(s: str) -> bool:
    try:
        s_norm=unicodedata.normalize("NFKD", s)
        s_ascii=s_norm.encode("ascii","ignore").decode("ascii")
        return bool(s_ascii.strip())
    except Exception: return False

def _looks_adjective(s: str) -> bool:
    adj_bad={"detailed","polished","refined","precise","precise form","defined silhouette",
             "clean","simple","minimal","muscular","paladin","ultra","grain","grains"}
    if s in adj_bad: return True
    if s.endswith(("ed","ish")) and len(s.split())==1: return True
    return False

def _is_color_word(s: str) -> bool: return s in set(COLOR_WORDS)

def _is_garbage_token(t: str) -> bool:
    t=(t or "").strip().lower()
    if not t: return True
    if t in STOPWORDS: return True
    if not _is_ascii_word(t): return True
    if t in BLOCKLIST or t in BLOCKLIST_META: return True
    if _contains_any_word(t, META_CONTAINS): return True
    if _RE_HAS_DIGIT.search(t): return True
    if _RE_UNITS.search(t) or t in _RES_TERMS: return True
    if len(t)<=2 or len(t.split())>4: return True
    if _looks_adjective(t): return True
    return False

def _valid_descriptor_phrase(x: str) -> bool:
    if not x: return False
    x=x.strip().lower()
    if _is_garbage_token(x): return False
    toks=x.split()
    if len(toks)==0 or len(toks)>4: return False
    if any(t in _PREP_WORDS for t in toks): return False
    roots=_token_roots(x)
    if any(r in BANNED_NOUN_ROOTS for r in roots): return False
    return True

def _clean_short_tokens(lst):
    out,seen=[],set()
    for r in lst or []:
        r=(r or "").strip().lower()
        if _is_garbage_token(r): continue
        if r in seen: continue
        out.append(r); seen.add(r)
    return out

def _clean_tokens_from_lines(lines, max_tokens=60):
    tokens=[]
    for ln in lines or []:
        for p in re.split(r"[;,/]", (ln or "").lower()):
            t=p.strip()
            if _is_garbage_token(t): continue
            if len(t.split())>8: continue
            tokens.append(t)
    return _clean_short_tokens(tokens)[:max_tokens]

def extract_ngrams_from_lines(lines: List[str], max_ngrams: int=300) -> List[str]:
    cands=[]
    for ln in (lines or []):
        ln=(ln or "").strip().lower()
        if not ln: continue
        tokens=re.findall(r"[a-z0-9]+", ln)
        for n in (1,2,3):
            for i in range(0, max(0,len(tokens)-n+1)):
                phrase=" ".join(tokens[i:i+n]).strip()
                if not phrase: continue
                if _is_garbage_token(phrase): continue
                cands.append(phrase)
    return list(dict.fromkeys(cands))[:max_ngrams]

def _token_roots(s: str) -> List[str]:
    roots=[]
    for t in re.findall(r"[a-z0-9]+", s.lower()):
        roots.append(re.sub(r"(ing|ers|er|ed|s)$","",t))
    return roots

def _head_noun(s: str) -> str:
    toks=re.findall(r"[a-z0-9]+", s.lower())
    return toks[-1] if toks else ""

def _map_to_head_category(tok: str) -> str:
    t=tok.lower().strip()
    for head,syns in OBJ_SYNONYM_MAP.items():
        if t in syns: return head
    return t

def _valid_object_phrase(x: str) -> bool:
    if not x: return False
    x=x.strip().lower()
    if _is_garbage_token(x): return False
    if _is_color_word(x) or "background" in x: return False
    toks=x.split()
    if len(toks)==0 or len(toks)>3: return False
    if any(t in _PREP_WORDS for t in toks): return False
    return True

def pick_main_object_zero_shot(pil: Image.Image, llava_base_tokens: List[str], pb_lines: List[str]) -> str:
    cand_llava=[t for t in llava_object_candidates_only(pil,k=12) if _valid_object_phrase(t)]
    base_short=[]
    for t in llava_base_tokens or []:
        t=(t or "").strip().lower()
        if _valid_object_phrase(t): base_short.append(t)
    base_short=list(dict.fromkeys(base_short))
    pb_ngrams=[p for p in extract_ngrams_from_lines(pb_lines,max_ngrams=300) if _valid_object_phrase(p)]

    pools=[("llava_obj",cand_llava),("llava_base",base_short),("pb",pb_ngrams)]
    all_phrases,src=[],{}
    for tag,arr in pools:
        for p in arr:
            if p not in src: src[p]=tag; all_phrases.append(p)
    if not all_phrases: return ""

    load_clip(DEVICE)
    tok=_CLIP_TOK(all_phrases, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        tf=_CLIP_MODEL.get_text_features(input_ids=tok["input_ids"].to(DEVICE),
                                         attention_mask=tok["attention_mask"].to(DEVICE))
    tf=tf/tf.norm(dim=-1, keepdim=True).clamp(min=1e-6); vf=_clip_image_feat(pil).cpu()
    sims=(tf.cpu() @ (vf/(vf.norm()+1e-6))).numpy().tolist()
    bonuses={"llava_obj":0.08,"llava_base":0.04,"pb":0.0}
    scored=[(sc+bonuses.get(src.get(p,"pb"),0.0), p) for p,sc in zip(all_phrases,sims)]
    scored.sort(key=lambda x:x[0], reverse=True)
    for _,p in scored:
        cand=_map_to_head_category(p)
        if _valid_object_phrase(cand): return cand
    return _map_to_head_category(scored[0][1])

def _dominant_color_term(pil: Image.Image) -> str:
    arr=np.asarray(pil.convert("RGB")).astype(np.float32)/255.0
    mean=arr.reshape(-1,3).mean(0); r,g,b=mean; mx=float(mean.max()); diff=mx-float(mean.min())
    if mx<0.18: return "black"
    if diff<0.12 and mx>0.9: return "white"
    if diff<0.15: return "gray"
    if   mx==r: h=(60*((g-b)/(diff+1e-6))+360)%360
    elif mx==g: h=(60*((b-r)/(diff+1e-6))+120)%360
    else:       h=(60*((r-g)/(diff+1e-6))+240)%360
    if 255<=h<300 and mx<0.45: return "dark purple/slate"
    if   h>=330 or h<15:  return "red"
    if 15<=h<45:          return "orange"
    if 45<=h<70:          return "yellow"
    if 70<=h<165:         return "green"
    if 165<=h<210:        return "cyan"
    if 210<=h<255:        return "blue"
    return "purple"

def _background_hint(pil: Image.Image) -> str:
    if pil.mode=="RGBA":
        a=np.asarray(pil.split()[-1])
        edge=np.concatenate([a[:10,:].ravel(),a[-10:,:].ravel(),a[:, :10].ravel(),a[:, -10:].ravel()])
        if edge.mean()<10: return "transparent background"
    gray=np.asarray(pil.convert("L"),dtype=np.float32)
    border=np.concatenate([gray[:10,:].ravel(),gray[-10:,:].ravel(),gray[:, :10].ravel(),gray[:, -10:].ravel()])
    if border.mean()>200: return "white background"
    if border.mean()<40:  return "dark background"
    return "plain background"

def _dedup_redundant(phrases: List[str], embeds: Dict[str,torch.Tensor], sim_th: float=0.92) -> List[str]:
    out=[]
    for p in (phrases or []):
        p_l=(p or "").strip().lower()
        if not p_l: continue
        dup=False
        for q in out:
            q_l=q.lower()
            if p_l in q_l or q_l in p_l: dup=True; break
            if p in embeds and q in embeds:
                a,b=embeds[p],embeds[q]
                cs=float((a @ b)/max(1e-6,a.norm()*b.norm()))
                if cs>=sim_th: dup=True; break
        if not dup: out.append(p)
    return out

def _prune_headnoun_duplicates(main_obj: str, ordered: List[str], rel_scores: Dict[str,float]) -> List[str]:
    if not ordered: return ordered
    head=_head_noun(main_obj); head_root=re.sub(r"(ing|ers|er|ed|s)$","", head)
    main_roots=set(_token_roots(main_obj))
    filtered=[]
    for p in ordered:
        if p==main_obj: filtered.append(p); continue
        toks=re.findall(r"[a-z0-9]+", p.lower())
        if len(toks)==1 and re.sub(r"(ing|ers|er|ed|s)$","",toks[0])==head_root: continue
        filtered.append(p)
    filtered2=[]
    for p in filtered:
        if p==main_obj: filtered2.append(p); continue
        roots=set(_token_roots(p)); toks=re.findall(r"[a-z0-9]+", p.lower())
        if roots.issubset(main_roots) and 1<=len(toks)<=2: continue
        filtered2.append(p)
    same_group=[p for p in filtered2 if (p!=main_obj) and (len(set(_token_roots(p)) & main_roots)>0)]
    keep=set()
    if same_group:
        best=max(same_group, key=lambda x: rel_scores.get(x,0.0)); keep.add(best)
    final=[]
    for p in filtered2:
        if p==main_obj: final.append(p)
        elif p in same_group:
            if p in keep: final.append(p)
        else:
            final.append(p)
    return final

def _synthesize_object_attrs(main_head: str,
                             color_hint: Optional[str],
                             material_scores: Dict[str,float],
                             parts_dict: Dict[str,List[str]],
                             k_fill: int=10) -> List[str]:
    out=[]
    mats=sorted(material_scores.keys(), key=lambda m: material_scores[m], reverse=True)
    MAT_TEMPLATES=[("bronze","bronze fittings"),("brass","brass trim"),("gold","gold inlay"),
                   ("silver","silver inlay"),("iron","iron reinforcement"),("steel","steel reinforcement"),
                   ("leather","leather straps"),("wood","wooden panels"),("glass","glass inlay"),
                   ("cloth","cloth binding"),("stone","stone ornament")]
    mat_map={k:v for k,v in MAT_TEMPLATES}
    for m in mats:
        if m in mat_map and mat_map[m] not in out:
            out.append(mat_map[m])
        if len(out)>=3: break
    if color_hint and color_hint not in {"white","gray","grey"}:
        out.append(f"{color_hint} accents")
    parts=parts_dict.get(main_head, []) + parts_dict.get("_generic", [])
    parts=list(dict.fromkeys(parts))
    out += parts[:6]
    out += ["ornate","engraved emblem","embossed pattern","inlaid trim","gem setting"]
    clean,seen=[],set()
    for t in out:
        s=t.strip().lower()
        if s in seen: continue
        seen.add(s); clean.append(s)
    return clean[:k_fill]

def _apply_conflict_rules(cands: list) -> list:
    s=list(dict.fromkeys(cands))
    if ("flat shading" in s or "clean shading" in s) and "specular highlights" in s:
        s=[x for x in s if x!="specular highlights"]
    if "top-down" in s and any(("3/4" in x) or ("front view" in x) for x in s):
        s=[x for x in s if x!="top-down"]
    return s

def _attach_head_if_needed(token: str, head: str) -> str:
    t=token.lower().strip()
    if head in t: return t
    PART_HINT=set(sum(OBJ_PARTS.values(), []))
    if any(ph in t for ph in PART_HINT) or any(w in t for w in
        ["pattern","motif","engraving","inlay","trim","ornate","emboss","embossed","engraved",
         "gem","setting","strap","spine","cover","pages","clasp","binding","visor","horn"]):
        return t
    if t in MATERIAL_WORDS or t in COLOR_WORDS: return f"{t} {head}"
    return t

def _ensure_object_first(main_obj: str, ordered: List[str], cand_pool: List[str],
                         rel_scores: Dict[str,float], target_min: int, target_max: int,
                         bg_final: Optional[str], style_cap: int=2,
                         main_head: Optional[str]=None, pil_for_rec: Optional[Image.Image]=None) -> List[str]:
    mo=(main_obj or "").strip().lower(); main_head=main_head or _head_noun(mo)
    obj_idx=None; obj_phrase=None
    if mo:
        for i,p in enumerate(ordered):
            if re.search(r"\b"+re.escape(mo)+r"\b", p.lower()):
                obj_idx, obj_phrase = i, p; break
    if obj_phrase is None: obj_phrase = main_obj if main_obj else (ordered[0] if ordered else "")
    out=[obj_phrase] + [x for i,x in enumerate(ordered) if i!=obj_idx]

    def is_style_like(s:str)->bool:
        roots=set(_token_roots(s))
        return (len(roots & LEX_STYLE_ROOTS)>0) or (len(roots & LEX_LIGHT_ROOTS)>0)
    kept,style_cnt=[],0
    for p in out:
        if is_style_like(p):
            if style_cnt<style_cap: kept.append(p); style_cnt+=1
        else: kept.append(p)
    out=kept

    out=_dedup_redundant(out, _clip_text_embeds(out), sim_th=0.92)
    out=_prune_headnoun_duplicates(main_obj, out, rel_scores)

    if len(out)<target_min and pil_for_rec is not None:
        mat_scores=_score_materials_with_clip(pil_for_rec)
        color_hint=None
        try: color_hint=_dominant_color_term(pil_for_rec)
        except Exception: pass
        synth=_synthesize_object_attrs(main_head=main_head,color_hint=color_hint,
                                       material_scores=mat_scores, parts_dict=OBJ_PARTS,
                                       k_fill=(target_min-len(out)+6))
        for s in synth:
            if s not in out:
                out.append(s)
            if len(out)>=target_min: break

    if len(out)>target_max: out=out[:target_max]
    return out

def _score_materials_with_clip(pil: Image.Image) -> Dict[str, float]:
    variants=[]
    template_map={
        "leather":["leather straps","leather trim","leather binding","leather"],
        "wood":["wooden panels","wooden trim","wood texture","wood"],
        "iron":["iron reinforcement","iron parts","iron"],
        "steel":["steel reinforcement","steel parts","steel"],
        "bronze":["bronze fittings","bronze trim","bronze"],
        "brass":["brass trim","brass fittings","brass"],
        "gold":["gold inlay","gold details","gold"],
        "silver":["silver inlay","silver details","silver"],
        "stone":["stone ornament","stone parts","stone"],
        "glass":["glass inlay","glass parts","glass"],
        "cloth":["cloth binding","cloth wrap","cloth"],
    }
    for m, vs in template_map.items():
        variants += vs
    scores=_clip_score_phrases(pil, variants)
    out={}
    for m, vs in template_map.items():
        out[m]=max(scores.get(v,0.0) for v in vs)
    if out:
        mn=min(out.values()); mx=max(out.values())
        if mx>mn:
            for k in out: out[k]=(out[k]-mn)/(mx-mn)
    return out

def auto_generate_prompt_elements(image_path: str, want_count: int=15) -> List[str]:
    pil_raw=Image.open(image_path).convert("RGBA" if image_path.lower().endswith((".png",".webp")) else "RGB")
    pil_for_bg=pil_raw.convert("RGB")
    pil_for_rec=prep_for_recognition(pil_raw)

    bg_hint=_background_hint(pil_for_bg)

    tags=llava_extract_tags(pil_for_rec) or {}
    base_llava=_clean_short_tokens(tags.get("base", []))
    shapes_llava=_clean_short_tokens(tags.get("shape", []))
    mats_llava=_clean_short_tokens(tags.get("material", []))
    view_llava=(tags.get("view") or "").strip().lower()
    lights_llava=_clean_short_tokens(tags.get("lighting", []))
    details_llava=_clean_short_tokens(tags.get("detail", []))
    styles_llava=_clean_short_tokens(tags.get("style", []))
    bg_llava=(tags.get("background") or "").strip().lower()

    global _PROMPTBANK_LINES
    if _PROMPTBANK_LINES is None: _PROMPTBANK_LINES=load_promptbank_lines(PROMPTBANK_PATH)

    llava_objs=llava_object_candidates_only(pil_for_rec, k=10)
    pb_lines=pb_search(_PROMPTBANK_LINES or [], base_llava + llava_objs + ["object","item"], topn=PROMPTBANK_TOPN)
    main_obj=pick_main_object_zero_shot(pil_for_rec, base_llava, pb_lines) or (llava_objs[0] if llava_objs else "object")
    main_head=_map_to_head_category(_head_noun(main_obj))

    pb_tokens_raw=_clean_tokens_from_lines(pb_lines, max_tokens=PROMPTBANK_TOKENS_MAX)
    main_roots=set(_token_roots(main_head))
    def _keep_pb(t:str)->bool:
        if not _valid_descriptor_phrase(t): return False
        rs=set(_token_roots(t))
        if len(rs & main_roots)>0: return True
        if any(r in LEX_STYLE_ROOTS for r in rs): return True
        if any(r in LEX_LIGHT_ROOTS for r in rs): return True
        if any(r in LEX_MATERIAL_ROOTS for r in rs): return True
        if any(r in LEX_LAYOUT_ROOTS for r in rs): return True
        if t in LEX_BG_ALLOWED: return True
        return False
    pb_tokens=[t for t in pb_tokens_raw if _keep_pb(t)]

    def filt(lst): return [t for t in (lst or []) if _valid_descriptor_phrase(t)]
    base_pool=filt(list(dict.fromkeys(base_llava + pb_tokens)))
    shape_pool=filt(shapes_llava); mat_pool=filt(mats_llava)
    light_pool=filt(lights_llava); detail_pool=filt(details_llava)
    style_pool=[t for t in filt(styles_llava) if not any(x in t for x in ["edges","shading","gradient"])]
    view_pool=filt([view_llava] if view_llava else [])
    bg_final=bg_llava if bg_llava else bg_hint

    def attach_all(pool): return [_attach_head_if_needed(p, main_head) for p in pool]
    base_pool=attach_all(base_pool); shape_pool=attach_all(shape_pool)
    mat_pool=attach_all(mat_pool); light_pool=attach_all(light_pool)
    detail_pool=attach_all(detail_pool); style_pool=attach_all(style_pool); view_pool=attach_all(view_pool)

    phrases=list(dict.fromkeys([*base_pool,*shape_pool,*mat_pool,*light_pool,*detail_pool,*style_pool,*view_pool]))
    rel_scores=_clip_score_phrases(pil_for_rec, phrases) if phrases else {}
    text_embeds=_clip_text_embeds(phrases) if phrases else {}
    cand=[p for p in phrases if rel_scores.get(p,0.0)>=0.18]
    cand=_dedup_redundant(cand, text_embeds, sim_th=0.92)
    cand=[p for p in cand if not (_is_color_word(p) and p=="white")]

    buckets={"style":[],"shape":[],"material":[],"view":[],"lighting":[],"detail":[],"base":[],"background":[]}
    for p in cand:
        pl=p.lower()
        if "background" in pl: continue
        if any(w in pl for w in MATERIAL_WORDS): buckets["material"].append(p)
        elif any(w in pl for w in ["light","lighting","bright","soft shadows"]): buckets["lighting"].append(p)
        elif any(w in pl for w in ["curve","curved","angular","visor","plating","horn","strap","spine","binding","rim","cover","pages","clasp"]):
            buckets["detail"].append(p)
        elif any(w in pl for w in ["vector","clean","flat","outline","stylized"]): buckets["style"].append(p)
        elif any(w in pl for w in ["3/4","front","top-down","view"]): buckets["view"].append(p)
        else: buckets["base"].append(p)
    if bg_final and "background" in bg_final: buckets["background"]=[bg_final.lower()]

    def topk(keys, cap): return sorted(keys, key=lambda x: rel_scores.get(x,0.0), reverse=True)[:cap]

    ordered=[main_head]
    ordered+=topk(buckets["material"],2)
    ordered+=topk(buckets["detail"],4)
    ordered+=topk(buckets["base"],5)
    ordered+=topk(buckets["lighting"],1)
    ordered+=topk(buckets["style"],1)
    if buckets["background"]: ordered+=topk(buckets["background"],1)

    ordered=_apply_conflict_rules(ordered)
    ordered=_dedup_redundant(ordered, _clip_text_embeds(ordered), sim_th=0.92)
    ordered=_prune_headnoun_duplicates(main_head, ordered, rel_scores)

    ordered=_ensure_object_first(main_obj=main_head, ordered=ordered, cand_pool=cand, rel_scores=rel_scores,
                                 target_min=TARGET_PROMPT_COUNT_MIN, target_max=TARGET_PROMPT_COUNT_MAX,
                                 bg_final=bg_final if ("background" in (bg_final or "")) else "clean background",
                                 style_cap=2, main_head=main_head, pil_for_rec=pil_for_rec)

    print(f"[auto] elements={len(ordered)} (first='{ordered[0] if ordered else ''}') | main_obj='{main_head}'")
    return ordered

def _enable_memory_opts(pipe, use_fp16: bool):
    try:
        pass
    except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if use_fp16:
        pipe.unet.to(torch.float16)
        for n in ("text_encoder","text_encoder_2"):
            te=getattr(pipe,n,None)
            if te is not None: te.to(torch.float16)
        pipe.vae.to(torch.float32)
    else:
        pipe.unet.to(torch.float32); pipe.vae.to(torch.float32)
        for n in ("text_encoder","text_encoder_2"):
            te=getattr(pipe,n,None)
            if te is not None: te.to(torch.float32)

    pipe.to(DEVICE)
    return pipe

def _try_load_controlnet() -> Optional[ControlNetModel]:
    for mid in CONTROLNET_SOFTEDGE_CANDIDATES + CONTROLNET_HED_CANDIDATES + CONTROLNET_CANNY_CANDIDATES:
        try:
            cn=ControlNetModel.from_pretrained(mid, torch_dtype=torch.float32 if DEVICE=="cuda" else torch.float32)
            cn.to(DEVICE); print(f"[ControlNet] loaded: {mid}"); return cn
        except Exception: continue
    print("[ControlNet] WARNING: failed to load any ControlNet model."); return None

def _try_load_ip_adapter(pipe) -> bool:
    for repo,sub,weight in IPADAPTER_REPO_CANDIDATES:
        try:
            pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight)
            print(f"[IP-Adapter] loaded: {repo}/{sub}/{weight}")
            return True
        except Exception: continue
    print("[IP-Adapter] WARNING: failed to load IP-Adapter weights."); return False

def build_pipeline_sdxl_controlnet_ip(adapter_scale: float=0.72):
    use_fp16 = False
    controlnet = _try_load_controlnet()
    try:
        if controlnet is not None:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                SDXL_BASE_ID, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=True
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                SDXL_BASE_ID, torch_dtype=torch.float32, use_safetensors=True
            )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE_ID, torch_dtype=torch.float32, use_safetensors=True
        )
    pipe = _enable_memory_opts(pipe, use_fp16)
    ip_ok = _try_load_ip_adapter(pipe)
    if ip_ok:
        try:
            pipe.set_ip_adapter_scale(adapter_scale)
        except Exception:
            pass
    pipe = _v16_coerce_float32_cuda(pipe)
    try:
        pipe = _apply_cuda_half_optimizations(pipe)
    except Exception as _e:
        print('[v18l hotfix] optimize skipped:', _e)

    return pipe, (controlnet is not None), ip_ok

def build_pipeline_sdxl_txt2img(allow_nsfw: bool = False, force_device: str = None):
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    device = force_device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if (device == "cuda") else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE_ID,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass

    if hasattr(pipe, "enable_vae_tiling"):
        try: pipe.enable_vae_tiling()
        except Exception: pass
    if hasattr(pipe, "enable_vae_slicing"):
        try: pipe.enable_vae_slicing()
        except Exception: pass

    try: pipe.to(device)
    except Exception: pass
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass

    if allow_nsfw and hasattr(pipe, "safety_checker"):
        try: pipe.safety_checker = None
        except Exception: pass

    return pipe


def _make_controlnet_kwargs(pipe, control_image, cn_scale, cn_start, cn_end):
    if not isinstance(control_image, Image.Image): return {}
    try:
        sig=inspect.signature(pipe.__call__); params=sig.parameters
        key="image" if "image" in params else ("control_image" if "control_image" in params else None)
        if key is None: return {}
        return {key:control_image, "controlnet_conditioning_scale":cn_scale,
                "control_guidance_start":[cn_start], "control_guidance_end":[cn_end]}
    except Exception:
        return {"image":control_image, "controlnet_conditioning_scale":cn_scale,
                "control_guidance_start":[cn_start], "control_guidance_end":[cn_end]}

def make_control_image(pil: Image.Image, prefer_softedge=True):
    if prefer_softedge and HAVE_SOFTEDGE and PidiNetDetector is not None:
        try:
            pidi=PidiNetDetector.from_pretrained("lllyasviel/Annotators")
            arr=np.array(pil.convert("RGB")); edge=pidi(arr, safe=True)
            return Image.fromarray(edge).convert("RGB"), "softedge"
        except Exception: pass
    if HAVE_HED and HEDdetector is not None:
        try:
            hed=HEDdetector.from_pretrained("lllyasviel/Annotators")
            arr=np.array(pil.convert("RGB")); hed_img=hed(arr)
            return Image.fromarray(hed_img).convert("RGB"), "hed"
        except Exception: pass
    try:
        import cv2
        arr=np.array(pil.convert("RGB"))
        gray=cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(gray,100,200)
        return Image.fromarray(np.stack([edges]*3,axis=-1)), "canny"
    except Exception:
        return pil.convert("L").convert("RGB"), "gray"

def generate_image_txt2img_with_controls(
    pipe, prompt, out_path, ref_image, control_image, control_type,
    steps, w, h, guidance=GUIDANCE_SCALE, neg="low quality, blurry, dark",
    cn_scale=1.0, cn_start=0.0, cn_end=0.6, ipa_scale=0.7, has_cn=False, has_ipa=False):
    if max(w,h) >= 1024 and steps > 22:
        steps = 22
    extra={}
    if has_ipa and isinstance(ref_image, Image.Image):
        try:
            extra["ip_adapter_image"]=ref_image
            try: pipe.set_ip_adapter_scale(ipa_scale)
            except Exception: pass
        except Exception: pass
    cn_kwargs={}
    if has_cn and isinstance(control_image, Image.Image):
        cn_kwargs=_make_controlnet_kwargs(pipe, control_image, cn_scale, cn_start, cn_end)
    use_bf16 = False
    cm = torch.autocast("cuda", dtype=torch.float32) if use_bf16 else contextlib.nullcontext()
    with torch.no_grad(), cm:
        out=pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=steps, guidance_scale=guidance,
                 width=w, height=h, output_type="pil", return_dict=True, **cn_kwargs, **extra)
    out.images[0].save(out_path)

def compute_ssim_pair(img1: str, img2: str) -> float:
    im1=Image.open(img1).convert("RGB"); im2=Image.open(img2).convert("RGB")
    if im2.size!=im1.size: im2=im2.resize(im1.size, Image.LANCZOS)
    a=np.asarray(im1); b=np.asarray(im2)
    try:    return float(ssim(a,b,channel_axis=2, data_range=255))
    except TypeError: return float(ssim(a,b,multichannel=True, data_range=255))

def is_near_black(img_path: str, mean_thresh: float = 3.0, std_thresh: float = 3.0) -> bool:
    try:
        arr = np.asarray(Image.open(img_path).convert("RGB"))
    except Exception:
        return True
    return (arr.mean() <= mean_thresh) and (arr.std() <= std_thresh)

def generate_with_screening_fullflow(
    prompt: str,
    original_path: str,
    out_path: str,
    pipe,
    ref_pil: Optional[Image.Image],
    control_img: Optional[Image.Image],
    control_kind: str,
    has_cn: bool,
    has_ipa: bool,
    stage_name: str,
    width: int,
    height: int,
    steps: int,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    max_retries: int = MAX_RETRIES,
) -> Tuple[bool, float, Dict]:
    last_ssim = last_clip = None
    for attempt in range(1, max_retries + 1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        generate_image_txt2img_with_controls(
            pipe=pipe, prompt=prompt, out_path=tmp_path,
            ref_image=ref_pil, control_image=(control_img if has_cn else None), control_type=control_kind,
            steps=steps, w=width, h=height, guidance=GUIDANCE_SCALE, neg="low quality, blurry, dark",
            cn_scale=1.15, cn_start=0.0, cn_end=0.6, ipa_scale=0.72, has_cn=has_cn, has_ipa=has_ipa)
        if is_near_black(tmp_path):
            try: os.remove(tmp_path)
            except: pass
            continue
        try:
            ssim_score = compute_ssim_pair(original_path, tmp_path)
        except Exception:
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue
        if ssim_score < ssim_threshold:
            try: os.remove(tmp_path)
            except: pass
            last_ssim = ssim_score
            continue
        try:
            clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception:
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue
        if clip_score < clip_threshold:
            try: os.remove(tmp_path)
            except: pass
            last_clip = clip_score
            continue
        try:
            os.replace(tmp_path, out_path)
        except Exception:
            Image.open(tmp_path).save(out_path)
            try: os.remove(tmp_path)
            except: pass
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": out_path}
    return False, 0.0, {"attempts": max_retries, "last_ssim": last_ssim, "last_clip": last_clip}


def score_prompt_set(elements: List[str], pipe, ref_pil, control_img, control_kind, has_cn, has_ipa) -> Tuple[float, Optional[str]]:
    prompt = ", ".join(elements)
    _log("[Set] Elements: " + prompt)

    proxy_path = os.path.join(out_dir, f"proxy_{uuid.uuid4().hex}.png")
    ok_p, clip_p, info_p = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, proxy_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="proxy", width=PROXY_W, height=PROXY_H, steps=PROXY_EFF_STEPS)
    try:
        if ok_p:
            _log(f"[Set] proxy -> PASS (SSIM={info_p.get('ssim')}, CLIP={info_p.get('clip')})")
        else:
            _log("[Set] proxy -> FAIL")
    finally:
        try: os.remove(proxy_path)
        except: pass

    if (not ok_p) and PROXY_STRICT:
        return 0.0, None

    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.png")
    ok_f, clip_f, info_f = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, out_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="full", width=FULL_W, height=FULL_H, steps=FULL_EFF_STEPS)
    if not ok_f:
        _log("[Set] full -> FAIL")
        return 0.0, None

    _log(f"[Set] full -> PASS (SSIM={info_f.get('ssim')}, CLIP={info_f.get('clip')}) saved={os.path.basename(out_path)}")
    return float(info_f.get('clip', 0.0)), out_path

def ga_population_optimize(
    elements: List[str],
    k: int,
    protected_index: int = 0,
    seed: Optional[int] = GA_RANDOM_SEED,
    pop_size: int = 18,
    generations: int = 1,
    elite_frac: float = 0.2,
    mut_prob: float = 0.25,
    pipe=None, ref_pil=None, control_img=None, control_kind="", has_cn=False, has_ipa=False
) -> Tuple[List[str], List[Tuple[str, float]]]:
    rng = random.Random(seed)
    n = len(elements)
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}]")
    if protected_index < 0 or protected_index >= n:
        raise ValueError("protected_index out of range")

    def _repair(ind):
        s = set(ind)
        s.add(protected_index)
        while len(s) < k:
            cand = rng.randrange(0, n)
            if cand not in s:
                s.add(cand)
        out = sorted(list(s))
        if len(out) > k:
            out = [protected_index] + [i for i in out if i != protected_index][:k-1]
            out = sorted(out)
        return tuple(out)

    base_pool = [i for i in range(n) if i != protected_index]
    def _rand_ind():
        return _repair(tuple(sorted([protected_index] + rng.sample(base_pool, k-1))))
    population = [_rand_ind() for _ in range(pop_size)]

    saved_records: List[Tuple[str, float]] = []

    def fitness(ind):
        subset = [elements[i] for i in ind]
        score, img_path = score_prompt_set(subset, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa)
        if img_path is not None:
            saved_records.append((img_path, score))
        return score

    def tournament_select(candidates, fit_vals, tsize=3):
        picks = rng.sample(range(len(candidates)), min(tsize, len(candidates)))
        best = max(picks, key=lambda idx: fit_vals[idx])
        return candidates[best]

    def crossover(p1, p2):
        core1 = [i for i in p1 if i != protected_index]
        core2 = [i for i in p2 if i != protected_index]
        cut = rng.randrange(1, k)
        child = [protected_index]
        pool = core1[:cut] + [i for i in core2 if i not in core1[:cut]]
        for i in pool:
            if len(child) >= k: break
            if i not in child:
                child.append(i)
        if len(child) < k:
            for i in range(n):
                if len(child) >= k: break
                if i not in child:
                    child.append(i)
        return _repair(tuple(sorted(child)))

    def mutate(ind):
        if rng.random() > mut_prob: 
            return ind
        core = [i for i in ind if i != protected_index]
        if not core:
            return ind
        drop = rng.choice(core)
        pool = [i for i in range(n) if i not in ind]
        if not pool:
            return ind
        add = rng.choice(pool)
        new = [i for i in ind if i != drop] + [add]
        return _repair(tuple(sorted(new)))

    elite_k = max(1, int(pop_size * elite_frac))

    for gen in range(1, generations+1):
        suffix = "th"
        if gen % 100 not in (11,12,13):
            if gen % 10 == 1: suffix = "st"
            elif gen % 10 == 2: suffix = "nd"
            elif gen % 10 == 3: suffix = "rd"
        gen_folder_name = f"{gen}{suffix}_generation"
        gen_folder_path = os.path.join(OUTPUT_DIR, gen_folder_name)
        os.makedirs(gen_folder_path, exist_ok=True)
        global out_dir
        out_dir = gen_folder_path

        fit_vals = [fitness(ind) for ind in population]
        best_idx = max(range(len(population)), key=lambda i: fit_vals[i])
        best_fit = fit_vals[best_idx]
        passes = [s for s in fit_vals if s > 0]
        fails  = [s for s in fit_vals if s <= 0]
        pass_rate = (len(passes) / max(1, len(fit_vals))) * 100.0
        mean_all  = (sum(fit_vals) / max(1, len(fit_vals)))
        mean_pass = (sum(passes) / max(1, len(passes))) if passes else 0.0
        mean_fail = (sum(fails)  / max(1, len(fails)))  if fails  else 0.0
        _log(f"[GA] Gen {gen}/{generations} | best={best_fit:.2f} | mean={mean_all:.2f} | "
             f"pass={pass_rate:.1f}% | mean_pass={mean_pass:.2f} | mean_fail={mean_fail:.2f} | pop={len(fit_vals)}")

        elite_order = sorted(range(len(population)), key=lambda i: fit_vals[i], reverse=True)[:elite_k]
        elites = [population[i] for i in elite_order]

        children = []
        while len(children) < (pop_size - elite_k):
            p1 = tournament_select(population, fit_vals, tsize=3)
            p2 = tournament_select(population, fit_vals, tsize=3)
            c = crossover(p1, p2)
            c = mutate(c)
            children.append(c)

        population = elites + children

    final_fit = [fitness(ind) for ind in population]
    best_idx = max(range(len(population)), key=lambda i: final_fit[i])
    best_subset_idx = population[best_idx]
    kept = [elements[i] for i in best_subset_idx]
    return kept, saved_records

def compile_sentence_prompt(main_obj: str, elems: List[str]) -> str:
    mat=next((e for e in elems if any(r in LEX_MATERIAL_ROOTS for r in _token_roots(e))), None)
    style=next((e for e in elems if "style" in e or "vector" in e or "stylized" in e), "clean vector-like style")
    light_bits=[]
    if any("bright" in e for e in elems): light_bits.append("bright lighting")
    if any("soft shadows" in e for e in elems): light_bits.append("soft shadows")
    bg=next((e for e in elems if "background" in e), "clean background")
    bits=[f"A stylized game icon of a {main_obj}"]
    if mat: bits.append(mat)
    if style: bits.append(style)
    bits+=light_bits; bits+=["sharp focus", bg]
    return ", ".join(dict.fromkeys([b.strip().strip(",") for b in bits if b]))


def propose_addon_descriptor(kept: List[str], all_pool: List[str], main_obj: str) -> Optional[str]:
    kept_set={k.lower() for k in kept}; main_roots=set(_token_roots(main_obj or ""))
    def _is_obj_syn(p:str)->bool:
        roots=set(_token_roots(p)); toks=re.findall(r"[a-z0-9]+", p.lower())
        return (len(roots & main_roots)>=1) and (1<=len(toks)<=2)
    pri=["detail","lighting","style","material","view","background","base"]
    buckets={b:[] for b in pri}
    for p in all_pool or []:
        pl=p.lower().strip()
        if not pl or pl in kept_set: continue
        if _is_garbage_token(pl) or _is_obj_syn(pl): continue
        if any(w in pl for w in MATERIAL_WORDS): buckets["material"].append(p)
        elif any(w in pl for w in ["light","lighting","bright","soft shadows"]): buckets["lighting"].append(p)
        elif any(w in pl for w in ["vector","clean","flat","outline","stylized"]): buckets["style"].append(p)
        elif any(w in pl for w in ["curve","curved","angular","visor","plating","u-shaped","horn","strap","spine","binding","rim"]): buckets["detail"].append(p)
        elif any(w in pl for w in ["3/4","front","top-down","view"]): buckets["view"].append(p)
        elif "background" in pl: buckets["background"].append(p)
        else: buckets["base"].append(p)
    for b in pri:
        if buckets[b]:
            cand=buckets[b][0]
            return _attach_head_if_needed(cand, _head_noun(main_obj))
    for f in ["engraved emblem","inlaid trim","gem setting"]:
        if f not in kept_set: return f
    return None
import sys, os, time
os.environ.setdefault("PYTHONUNBUFFERED","1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

VERBOSE_LOG = True

def _log(msg: str):
    if VERBOSE_LOG:
        try:
            print(msg, flush=True)
        except Exception:
            print(msg)

def _ssim_score(a: str, b: str) -> float:
    try:
        return compute_ssim(a,b)
    except Exception:
        try:
            return compute_ssim_pair(a,b)
        except Exception as e:
            _log(f"[LOG] {e}")
            raise

def generate_image_with_screening(
    prompt: str,
    original_path: str,
    output_path: str,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
    width: int = FULL_W,
    height: int = FULL_H,
    effective_steps: int = FULL_EFF_STEPS,
    stage_name: str = "full",
    neg: str = "low quality, blurry, dark",
    pipe_override=None) -> Tuple[bool, float, dict]:
    proxy_guidance = GUIDANCE_SCALE
    try:
        if stage_name.lower() == 'proxy':
            proxy_guidance = min(GUIDANCE_SCALE, 4.0)
    except Exception:
        pass
    global PIPE_TXT2IMG
    pipe = pipe_override or PIPE_TXT2IMG
    last_ssim = None
    last_clip = None
    preview = (prompt[:300] + ("..." if len(prompt) > 300 else ""))
    for attempt in range(1, max_retries+1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        _log(f"[{stage_name}] Attempt {attempt}/1")
        _log(f"[{stage_name}] Prompt: {preview}")
        try:
            with torch.no_grad():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    num_inference_steps=effective_steps,
                    guidance_scale=proxy_guidance,
                    width=width, height=height,
                    output_type="pil",
                    return_dict=True,
                )
            out.images[0].save(tmp_path)
        except Exception as e:
            _log(f"[{stage_name}] generation error: {e}")
            continue

        try:
            if is_near_black(tmp_path):
                _log(f"[{stage_name}] -> near-black detected, retrying")
                try: os.remove(tmp_path)
                except: pass
                continue
        except Exception:
            pass

        try:
            ssim_score = _ssim_score(original_path, tmp_path)
        except Exception as e:
            _log(f"[{stage_name}] -> SSIM error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue

        try:
            clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception as e:
            _log(f"[{stage_name}] -> CLIP error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue

        last_ssim = ssim_score
        last_clip = clip_score
        _log(f"[{stage_name}] Metrics: SSIM={ssim_score:.4f} (≥ {ssim_threshold}) | CLIP={clip_score:.2f}% (≥ {clip_threshold}%)")

        if ssim_score < ssim_threshold or clip_score < clip_threshold:
            _log(f"[{stage_name}] -> FAIL (threshold not met)")
            try: os.remove(tmp_path)
            except: pass
            continue

        try:
            os.replace(tmp_path, output_path)
        except Exception:
            Image.open(tmp_path).save(output_path)
            try: os.remove(tmp_path)
            except: pass

        _log(f"[{stage_name}] -> PASS, saved: {os.path.basename(output_path)}")
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": output_path}

    _log(f"[{stage_name}] Single attempt FAILED | last SSIM={last_ssim} | last CLIP={last_clip}")
    return False, 0.0, {"attempts": max_retries, "last_ssim": last_ssim, "last_clip": last_clip}


def generate_with_screening_fullflow(
    prompt: str,
    original_path: str,
    out_path: str,
    pipe,
    ref_pil: Optional[Image.Image],
    control_img: Optional[Image.Image],
    control_kind: str,
    has_cn: bool,
    has_ipa: bool,
    stage_name: str,
    width: int,
    height: int,
    steps: int,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    max_retries: int = MAX_RETRIES,
) -> Tuple[bool, float, Dict]:
    last_ssim = None
    last_clip = None
    preview = (prompt[:300] + ("..." if len(prompt) > 300 else ""))
    for attempt in range(1, max_retries+1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        _log(f"[{stage_name}] Attempt {attempt}/1")
        _log(f"[{stage_name}] Prompt: {preview}")
        generate_image_txt2img_with_controls(
            pipe=pipe, prompt=prompt, out_path=tmp_path,
            ref_image=ref_pil, control_image=(control_img if has_cn else None), control_type=control_kind,
            steps=steps, w=width, h=height, guidance=GUIDANCE_SCALE, neg="low quality, blurry, dark",
            cn_scale=1.15, cn_start=0.0, cn_end=0.6, ipa_scale=0.72, has_cn=has_cn, has_ipa=has_ipa)

        try:
            if is_near_black(tmp_path):
                _log(f"[{stage_name}] -> near-black detected, retrying")
                try: os.remove(tmp_path)
                except: pass
                continue
        except Exception:
            pass

        try:
            ssim_score = _ssim_score(original_path, tmp_path)
        except Exception as e:
            _log(f"[{stage_name}] -> SSIM error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue

        try:
            clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception as e:
            _log(f"[{stage_name}] -> CLIP error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue

        last_ssim = ssim_score
        last_clip = clip_score
        _log(f"[{stage_name}] Metrics: SSIM={ssim_score:.4f} (≥ {ssim_threshold}) | CLIP={clip_score:.2f}% (≥ {clip_threshold}%)")

        if ssim_score < ssim_threshold or clip_score < clip_threshold:
            _log(f"[{stage_name}] -> FAIL (threshold not met)")
            try: os.remove(tmp_path)
            except: pass
            continue

        try:
            os.replace(tmp_path, out_path)
        except Exception:
            Image.open(tmp_path).save(out_path)
            try: os.remove(tmp_path)
            except: pass
        _log(f"[{stage_name}] -> PASS, saved: {os.path.basename(out_path)}")
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": out_path}

    _log(f"[{stage_name}] Single attempt FAILED | last SSIM={last_ssim} | last CLIP={last_clip}")
    return False, 0.0, {"attempts": max_retries, "last_ssim": last_ssim, "last_clip": last_clip}



def score_prompt_set(elements: List[str], pipe, ref_pil, control_img, control_kind, has_cn, has_ipa) -> Tuple[float, Optional[str]]:
    prompt = ", ".join(elements)
    _log("[Set] Elements: " + prompt)

    proxy_path = os.path.join(out_dir, f"proxy_{uuid.uuid4().hex}.png")
    ok_p, clip_p, info_p = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, proxy_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="proxy", width=PROXY_W, height=PROXY_H, steps=PROXY_EFF_STEPS)
    try:
        if ok_p:
            _log(f"[Set] proxy -> PASS (SSIM={info_p.get('ssim')}, CLIP={info_p.get('clip')})")
        else:
            _log("[Set] proxy -> FAIL")
    finally:
        try: os.remove(proxy_path)
        except: pass

    if (not ok_p) and PROXY_STRICT:
        return 0.0, None

    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.png")
    ok_f, clip_f, info_f = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, out_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="full", width=FULL_W, height=FULL_H, steps=FULL_EFF_STEPS)
    if not ok_f:
        _log("[Set] full -> FAIL")
        return 0.0, None

    _log(f"[Set] full -> PASS (SSIM={info_f.get('ssim')}, CLIP={info_f.get('clip')}) saved={os.path.basename(out_path)}")
    return float(info_f.get('clip', 0.0)), out_path

def compute_ssim(a_path: str, b_path: str) -> float:
    return _ssim_score(a_path, b_path)

def compute_ssim_pair(a_path: str, b_path: str) -> float:
    return _ssim_score(a_path, b_path)

print("[v7] SSIM shims active: compute_ssim / compute_ssim_pair -> _ssim_score", flush=True)

import sys, os
os.environ.setdefault("PYTHONUNBUFFERED","1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

V10_VERBOSE = True
def _v10_log(msg: str):
    if V10_VERBOSE:
        try:
            print(msg, flush=True)
        except Exception:
            print(msg)

def _v10_local_ssim(a_path: str, b_path: str) -> float:
    from PIL import Image
    import numpy as _np
    try:
        from skimage.metrics import structural_similarity as _ssim_fn
        im1 = Image.open(a_path).convert("RGB")
        im2 = Image.open(b_path).convert("RGB")
        if im2.size != im1.size:
            im2 = im2.resize(im1.size, Image.LANCZOS)
        A = _np.asarray(im1)
        B = _np.asarray(im2)
        try:
            return float(_ssim_fn(A, B, channel_axis=2, data_range=255))
        except TypeError:
            return float(_ssim_fn(A, B, multichannel=True, data_range=255))
    except Exception:
        im1 = Image.open(a_path).convert("RGB")
        im2 = Image.open(b_path).convert("RGB")
        if im2.size != im1.size:
            im2 = im2.resize(im1.size, Image.LANCZOS)
        A = _np.asarray(im1).astype(_np.float32)
        B = _np.asarray(im2).astype(_np.float32)
        mse = _np.mean((A - B) ** 2)
        norm = _np.mean((A - 127.5) ** 2) + 1e-6
        score = max(0.0, 1.0 - (mse / (norm * 4.0)))
        return float(score)

def generate_image_with_screening(
    prompt: str,
    original_path: str,
    output_path: str,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True,
    width: int = FULL_W,
    height: int = FULL_H,
    effective_steps: int = FULL_EFF_STEPS,
    stage_name: str = "full",
    neg: str = "low quality, blurry, dark", pipe_override=None):
    global PIPE_TXT2IMG
    pipe = pipe_override or PIPE_TXT2IMG
    proxy_guidance = GUIDANCE_SCALE
    try:
        if stage_name.lower() == 'proxy':
            proxy_guidance = min(GUIDANCE_SCALE, 4.0)
    except Exception:
        pass
    last_ssim = None
    last_clip = None
    preview = (prompt[:300] + ("..." if len(prompt) > 300 else ""))
    for attempt in range(1, max_retries+1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        _v10_log(f"[{stage_name}] Attempt {attempt}/1")
        _v10_log(f"[{stage_name}] Prompt: {preview}")
        try:
            with torch.no_grad():
                out = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    num_inference_steps=effective_steps,
                    guidance_scale=proxy_guidance,
                    width=width, height=height,
                    output_type="pil",
                    return_dict=True,
                )
            out.images[0].save(tmp_path)
        except Exception as e:
            _v10_log(f"[{stage_name}] generation error: {e}")
            continue

        try:
            if is_near_black(tmp_path):
                _v10_log(f"[{stage_name}] -> near-black detected, retrying")
                try: os.remove(tmp_path)
                except: pass
                continue
        except Exception:
            pass

        try:
            ssim_score = _v10_local_ssim(original_path, tmp_path)
        except Exception as e:
            _v10_log(f"[{stage_name}] -> SSIM error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue

        try:
            clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception as e:
            _v10_log(f"[{stage_name}] -> CLIP error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue

        last_ssim = ssim_score
        last_clip = clip_score
        _v10_log(f"[{stage_name}] Metrics: SSIM={ssim_score:.4f} (≥ {ssim_threshold}) | CLIP={clip_score:.2f}% (≥ {clip_threshold}%)")

        if ssim_score < ssim_threshold or clip_score < clip_threshold:
            _v10_log(f"[{stage_name}] -> FAIL (threshold not met)")
            try: os.remove(tmp_path)
            except: pass
            continue

        try:
            os.replace(tmp_path, output_path)
        except Exception:
            Image.open(tmp_path).save(output_path)
            try: os.remove(tmp_path)
            except: pass
        _v10_log(f"[{stage_name}] -> PASS, saved: {os.path.basename(output_path)}")
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": output_path}

    _v10_log(f"[{stage_name}] Single attempt FAILED | last SSIM={last_ssim} | last CLIP={last_clip}")
    return False, 0.0, {"attempts": max_retries, "last_ssim": last_ssim, "last_clip": last_clip}

def generate_with_screening_fullflow(
    prompt: str,
    original_path: str,
    out_path: str,
    pipe,
    ref_pil: Optional[Image.Image],
    control_img: Optional[Image.Image],
    control_kind: str,
    has_cn: bool,
    has_ipa: bool,
    stage_name: str,
    width: int,
    height: int,
    steps: int,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
    max_retries: int = MAX_RETRIES,
):
    last_ssim = None
    last_clip = None
    preview = (prompt[:300] + ("..." if len(prompt) > 300 else ""))
    for attempt in range(1, max_retries+1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        _v10_log(f"[{stage_name}] Attempt {attempt}/1")
        _v10_log(f"[{stage_name}] Prompt: {preview}")
        generate_image_txt2img_with_controls(
            pipe=pipe, prompt=prompt, out_path=tmp_path,
            ref_image=ref_pil, control_image=(control_img if has_cn else None), control_type=control_kind,
            steps=steps, w=width, h=height, guidance=GUIDANCE_SCALE, neg="low quality, blurry, dark",
            cn_scale=1.15, cn_start=0.0, cn_end=0.6, ipa_scale=0.72, has_cn=has_cn, has_ipa=has_ipa)

        try:
            if is_near_black(tmp_path):
                _v10_log(f"[{stage_name}] -> near-black detected, retrying")
                try: os.remove(tmp_path)
                except: pass
                continue
        except Exception:
            pass

        try:
            ssim_score = _v10_local_ssim(original_path, tmp_path)
        except Exception as e:
            _v10_log(f"[{stage_name}] -> SSIM error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue

        try:
            clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception as e:
            _v10_log(f"[{stage_name}] -> CLIP error: {e}")
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue

        last_ssim = ssim_score
        last_clip = clip_score
        _v10_log(f"[{stage_name}] Metrics: SSIM={ssim_score:.4f} (≥ {ssim_threshold}) | CLIP={clip_score:.2f}% (≥ {clip_threshold}%)")

        if ssim_score < ssim_threshold or clip_score < clip_threshold:
            _v10_log(f"[{stage_name}] -> FAIL (threshold not met)")
            try: os.remove(tmp_path)
            except: pass
            continue

        try:
            os.replace(tmp_path, out_path)
        except Exception:
            Image.open(tmp_path).save(out_path)
            try: os.remove(tmp_path)
            except: pass
        _v10_log(f"[{stage_name}] -> PASS, saved: {os.path.basename(out_path)}")
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": out_path}

    _v10_log(f"[{stage_name}] Single attempt FAILED | last SSIM={last_ssim} | last CLIP={last_clip}")
    return False, 0.0, {"attempts": max_retries, "last_ssim": last_ssim, "last_clip": last_clip}

def score_prompt_set(elements: List[str], pipe, ref_pil, control_img, control_kind, has_cn, has_ipa):
    prompt = ", ".join(elements)
    _v10_log("[Set] Elements: " + prompt)

    proxy_path = os.path.join(out_dir, f"proxy_{uuid.uuid4().hex}.png")
    ok_p, clip_p, info_p = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, proxy_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="proxy", width=PROXY_W, height=PROXY_H, steps=PROXY_EFF_STEPS)
    if not ok_p:
        _v10_log("[Set] proxy -> FAIL")
        return 0.0, None
    else:
        if isinstance(info_p, dict):
            _v10_log(f"[Set] proxy -> PASS (SSIM={info_p.get('ssim')}, CLIP={info_p.get('clip')})")
    try: os.remove(proxy_path)
    except: pass

    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.png")
    ok_f, clip_f, info_f = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, out_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="full", width=FULL_W, height=FULL_H, steps=FULL_EFF_STEPS)
    if not ok_f:
        _v10_log("[Set] full -> FAIL")
        return 0.0, None
    else:
        if isinstance(info_f, dict):
            _v10_log(f"[Set] full -> PASS (SSIM={info_f.get('ssim')}, CLIP={info_f.get('clip')}) | saved={info_f.get('path')}")
    return float(clip_f), out_path

def _v14_ordinal(n: int) -> str:
    n_abs = abs(int(n))
    if 11 <= (n_abs % 100) <= 13: suf = "th"
    else:
        suf = {1:"st",2:"nd",3:"rd"}.get(n_abs % 10, "th")
    return f"{n}{suf}"

def _v14_gen_dir(gen_idx: int) -> str:
    sub = f"{_v14_ordinal(gen_idx)}_generation"
    out_dir = os.path.join(OUTPUT_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _v12_eval_set(elements: List[str], pipe, ref_pil, control_img, control_kind, has_cn, has_ipa, out_dir: str):
    prompt = ", ".join(elements)
    proxy_path = os.path.join(out_dir, f"proxy_{uuid.uuid4().hex}.png")
    ok_p, clip_p, info_p = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, proxy_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="proxy", width=PROXY_W, height=PROXY_H, steps=PROXY_EFF_STEPS)
    if not ok_p:
        try: os.remove(proxy_path)
        except: pass
        return False, 0.0, 0.0, None
    try: os.remove(proxy_path)
    except: pass

    out_path = os.path.join(out_dir, f"{uuid.uuid4().hex}.png")
    ok_f, clip_f, info_f = generate_with_screening_fullflow(
        prompt, ORIGINAL_IMAGE_PATH, out_path, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa,
        stage_name="full", width=FULL_W, height=FULL_H, steps=FULL_EFF_STEPS)
    if not ok_f:
        return False, 0.0, 0.0, None
    ssim_val = float(info_f.get('ssim', 0.0)) if isinstance(info_f, dict) else 0.0
    clip_val = float(info_f.get('clip', 0.0)) if isinstance(info_f, dict) else float(clip_f)
    return True, ssim_val, clip_val, info_f.get('path') if isinstance(info_f, dict) else out_path


def _ga_repair_subset(indices, k, protected_index, n):
    s = set(indices)
    if protected_index is not None:
        s.add(protected_index)
    if len(s) > k:
        extra = [i for i in s if i != protected_index]
        rng.shuffle(extra)
        for i in extra:
            if len(s) <= k: break
            s.remove(i)
    elif len(s) < k:
        pool = [i for i in range(n) if i not in s]
        rng.shuffle(pool)
        for i in pool:
            if len(s) >= k: break
            s.add(i)
    return tuple(sorted(s))

def _ga_random_individual(n, k, protected_index):
    others = [i for i in range(n) if i != protected_index]
    pick = rng.sample(others, k-1) if k-1 <= len(others) else others
    return _ga_repair_subset([protected_index, *pick], k, protected_index, n)

def _ga_crossover(p1, p2, k, protected_index, n):
    a, b = set(p1), set(p2)
    child = set([protected_index] if protected_index is not None else [])
    for i in range(n):
        if i==protected_index: continue
        if i in a and i in b:
            if rng.random()<0.5: child.add(i)
        elif i in a:
            if rng.random()<0.5: child.add(i)
        elif i in b:
            if rng.random()<0.5: child.add(i)
    return _ga_repair_subset(child, k, protected_index, n)

def _ga_mutate(ind, mut_rate, k, protected_index, n):
    s=set(ind)
    if rng.random() < mut_rate:
        cand_remove=[i for i in s if i!=protected_index]
        if cand_remove:
            s.remove(rng.choice(cand_remove))
            pool=[i for i in range(n) if i not in s]
            if pool: s.add(rng.choice(pool))
    return _ga_repair_subset(s, k, protected_index, n)

def _ga_eval_individual(ind, elements, cache, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa):
    key = tuple(sorted(ind))
    if key in cache:
        return cache[key]
    prompt = ", ".join(elements[i] for i in key)
    img_path = os.path.join(OUTPUT_DIR, f"ga_{uuid.uuid4().hex}.png")
    ok, clip_val, info = generate_image_with_screening(
        prompt=prompt,
        original_path=ORIGINAL_IMAGE_PATH,
        output_path=img_path,
        ssim_threshold=SSIM_THRESHOLD,
        clip_threshold=CLIP_THRESHOLD,
        max_retries=1,
        width=FULL_W, height=FULL_H, effective_steps=FULL_EFF_STEPS,
        stage_name="full",
    )
    ssim_val = info.get("ssim", 0.0) if ok else info.get("last_ssim", 0.0) or 0.0
    fitness = GA_SCORE_ALPHA*clip_val + GA_SCORE_BETA*ssim_val
    cache[key] = {"fitness": fitness, "clip": clip_val, "ssim": ssim_val, "ok": ok, "path": img_path, "prompt": prompt}
    print(f"[GA-EVAL] fit={fitness:.2f} | CLIP={clip_val:.2f}% | SSIM={ssim_val:.4f} | ok={ok} | {os.path.basename(img_path)}", flush=True)
    print(f"[GA-EVAL] prompt: {prompt}", flush=True)
    return cache[key]

def ga_evolutionary_optimize(
    elements: List[str], k: int, protected_index: int,
    pipe, ref_pil, control_img, control_kind, has_cn: bool, has_ipa: bool,
    pop_size: int = GA_POP_SIZE, elitism: int = GA_ELITISM, mut_rate: float = GA_MUT_RATE,
    max_gens: int = GA_MAX_GENS, patience: int = GA_PATIENCE, seed: Optional[int] = None,
):
    global rng
    import secrets
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    n = len(elements)
    assert 1 <= k <= n, "k must be within 1..n"
    population = [_ga_random_individual(n, k, protected_index) for _ in range(pop_size)]
    cache = {}
    best_fit = -1e18
    best_ind = None
    no_imp = 0
    saved_records = []

    for gen in range(1, max_gens+1):
        print(f"[GA] === Generation {gen}/{max_gens} | pop={len(population)} | k={k} ===", flush=True)
        scores = []
        for ind in population:
            res = _ga_eval_individual(ind, elements, cache, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa)
            scores.append((res["fitness"], ind, res))
        scores.sort(key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind, _ in scores[:elitism]]
        top_res = [res for _,_,res in scores[:elitism]]
        top_res.sort(key=lambda r: r["fitness"], reverse=True)
        if top_res:
            saved_records.append((top_res[0]["path"], top_res[0]["clip"]))
        if scores[0][0] > best_fit + 1e-6:
            best_fit = scores[0][0]
            best_ind = scores[0][1]
            no_imp = 0
        else:
            no_imp += 1
        print(f"[GA] best fit={best_fit:.2f} (no_imp={no_imp}/{patience}) | best set idx={best_ind}", flush=True)

        if no_imp >= patience and gen >= 3:
            print("[GA] Early stop: no improvement", flush=True)
            break

        next_pop = elites[:]
        while len(next_pop) < pop_size:
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            child = _ga_crossover(p1, p2, k, protected_index, n)
            child = _ga_mutate(child, mut_rate, k, protected_index, n)
            next_pop.append(child)
        population = next_pop

    _ = _ga_eval_individual(best_ind, elements, cache, pipe, ref_pil, control_img, control_kind, has_cn, has_ipa)
    best_elems = [elements[i] for i in best_ind]
    return best_elems, saved_records

def _safe_swap_first_element(final_prompt: str, new_object: str):
    if not isinstance(final_prompt, str):
        try:
            final_prompt = str(final_prompt)
        except Exception:
            return final_prompt
    new_object = (new_object or "").strip()
    if not new_object:
        return final_prompt
    parts = [p.strip() for p in final_prompt.split(",")]
    if not parts:
        return new_object
    parts[0] = new_object
    return ", ".join([p for p in parts if p])
def generate_final_image_with_toggle(
    prompt: str,
    original_path: str,
    out_path: str,
    width: int,
    height: int,
    steps: int,
    use_cn: bool,
    use_ipa: bool,
    pipe_cn,
    pipe_txt2img,
    ref_pil=None,
    control_img=None,
    control_kind=None,
    ssim_threshold: float = SSIM_THRESHOLD,
    clip_threshold: float = CLIP_THRESHOLD,
):
    if use_cn:
        ok, clip, info = generate_with_screening_fullflow(
            prompt=prompt,
            original_path=original_path,
            out_path=out_path,
            pipe=pipe_cn,
            ref_pil=ref_pil if use_ipa else None,
            control_img=control_img if use_cn else None,
            control_kind=control_kind if use_cn else None,
            has_cn=use_cn,
            has_ipa=use_ipa,
            stage_name="final",
            width=width,
            height=height,
            steps=steps,
        )
        return ok, clip, info
    else:
        ok = generate_image_with_screening(
            prompt=prompt,
            original_path=original_path,
            output_path=out_path,
            ssim_threshold=ssim_threshold,
            clip_threshold=clip_threshold,
            max_retries=1,
            width=width,
            height=height,
            effective_steps=steps,
            stage_name="final-prompt-only",
            pipe_override=pipe_txt2img,
        )
        return (ok if isinstance(ok, bool) else bool(ok)), None, {}

if __name__ == "__main__":
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        _ = load_clip(DEVICE)
    except Exception as e:
        try:
            print(f"[boot] CLIP load warn: {e}")
        except Exception:
            pass

    PIPE_TXT2IMG, has_cn, has_ipa = build_pipeline_sdxl_controlnet_ip(adapter_scale=0.72)

    _apply_cuda_half_optimizations(PIPE_TXT2IMG)
    ref_pil = Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")
    control_img, control_kind = make_control_image(ref_pil, prefer_softedge=True)

    prompt_elements = auto_generate_prompt_elements(ORIGINAL_IMAGE_PATH, want_count=15)
    print("\n[Auto Prompt Elements]")
    for i, e in enumerate(prompt_elements, 1):
        print(f"{i:02d}. {e}")

    k = 9 if len(prompt_elements) >= 8 else max(2, len(prompt_elements))
    elite_frac = max(1, GA_ELITISM) / float(GA_POP_SIZE)
    print(f"[GA-ENTRY] n={len(prompt_elements)} | k={k} | pop={GA_POP_SIZE} | gens={GA_MAX_GENS} | elite_frac={elite_frac:.2f} | mut={GA_MUT_RATE}")

    best, saved_records = ga_population_optimize(
        elements=prompt_elements,
        k=k,
        protected_index=0,
        seed=GA_RANDOM_SEED,
        pop_size=GA_POP_SIZE,
        generations=GA_MAX_GENS,
        elite_frac=elite_frac,
        mut_prob=GA_MUT_RATE,
        pipe=PIPE_TXT2IMG, ref_pil=ref_pil,
        control_img=control_img, control_kind=control_kind,
        has_cn=has_cn, has_ipa=has_ipa
    )

    print("\n[GA Final Elements]")
    for e in best:
        print(f"- {e}")

    try:
        if isinstance(best, (list, tuple)) and len(best) < 9 and len(best) > 0:
            _pool = list(best[1:] if len(best) > 1 else best)
            while len(best) < 9 and _pool:
                best = list(best) + [_pool[(len(best)) % len(_pool)]]
        _resp = 'n'
    except Exception:
        _resp = 'n'
    if _resp == 'y':
        try:
            _new = input("Enter new object (e.g., 'sword'): ").strip()
            if _new:
                print(f"[swap] replacing '{best[0]}' -> '{_new}'")
                best[0] = _new
        except Exception:
            pass
    final_prompt = ", ".join(best)
    final_out = os.path.join(OUTPUT_DIR, f"ga_best_{uuid.uuid4().hex}.png")


PIPE_TXT2IMG_PURE = build_pipeline_sdxl_txt2img(allow_nsfw=False, force_device=None)

try:
    _swap_yn = input("Swap the 1st element (index 0)? (y/n): ").strip().lower()
except Exception:
    _swap_yn = "n"

_swapped = False
_new_obj = None
if _swap_yn == "y":
    try:
        _new_obj = input("Enter new object for #0: ").strip()
    except Exception:
        _new_obj = ""
    def _safe_swap_first_element(final_prompt: str, new_object: str):
        if not isinstance(final_prompt, str):
            final_prompt = str(final_prompt)
        parts = [p.strip() for p in final_prompt.split(",")]
        if not parts:
            return new_object
        parts[0] = new_object.strip()
        return ", ".join([p for p in parts if p])

    if _new_obj:
        final_prompt = _safe_swap_first_element(final_prompt, _new_obj)
        _swapped = True
    else:
        print('[swap] No object entered. Keeping original prompt; screening path will run.')
        _swapped = False

if _swapped:
    try:
        out = PIPE_TXT2IMG_PURE(
            prompt=final_prompt,
            width=FULL_W,
            height=FULL_H,
            num_inference_steps=FULL_EFF_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )
        img = out.images[0]
        img.save(final_out)
        print(f"[final-swapped:no-screen] saved: {final_out}")
    except Exception as e:
        print("[final-swapped] generation error:", e)
else:
    ok, clip, info = generate_final_image_with_toggle(
        prompt=final_prompt,
        original_path=ORIGINAL_IMAGE_PATH,
        out_path=final_out,
        width=FULL_W,
        height=FULL_H,
        steps=FULL_EFF_STEPS,
        use_cn=True,
        use_ipa=True,
        pipe_cn=PIPE_TXT2IMG,
        pipe_txt2img=PIPE_TXT2IMG_PURE,
        ref_pil=ref_pil,
        control_img=control_img,
        control_kind=control_kind,
        ssim_threshold=SSIM_THRESHOLD,
        clip_threshold=CLIP_THRESHOLD,
    )

try:
    import torch, os
    from contextlib import contextmanager
except Exception as _e:
    print("[speedfix] import error:", _e)

def _apply_cuda_half_optimizations(pipe):
    try:
        if pipe is None:
            return
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("[speedfix] CUDA not available; skip half-optimizations")
            return
        dev = torch.device("cuda:0")
        for mname in ("unet", "vae"):
            m = getattr(pipe, mname, None)
            if m is not None:
                m.to(device=dev, dtype=torch.float16)
        for mname in ("text_encoder", "text_encoder_2"):
            m = getattr(pipe, mname, None)
            if m is not None:
                try: m.to(device=dev, dtype=torch.float16)
                except Exception: m.to(device=dev)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as _e:
            print("[speedfix] xformers not enabled:", _e)
        try:
            if hasattr(pipe, "vae"):
                pipe.vae.enable_tiling()
                try:
                    pipe.vae.disable_slicing()
                except Exception:
                    pass
        except Exception as _e:
            print("[speedfix] vae tile/slice warn:", _e)
        print("[speedfix] pipeline moved to cuda fp16")
    except Exception as _e:
        print("[speedfix] optimize exception:", _e)

try:
    _orig_build = build_pipeline_sdxl_controlnet_ip
    def build_pipeline_sdxl_controlnet_ip(*args, **kwargs):
        pipe, device = _orig_build(*args, **kwargs)
        _apply_cuda_half_optimizations(pipe)
        return pipe, device
    print("[speedfix] build_pipeline_sdxl_controlnet_ip wrapped")
except Exception as _e:
    print("[speedfix] wrap builder skipped:", _e)

try:
    _orig_generate_image_with_screening = generate_image_with_screening
    def generate_image_with_screening(*args, pipe_override=None, **kwargs):
        stage_name = kwargs.get("stage_name", "full")
        proxy_steps_env = os.environ.get("PROXY_STEPS")
        proxy_g_env = os.environ.get("PROXY_GUIDANCE")
        disable_pb = os.environ.get("PROXY_DISABLE_TQDM", "1") != "0"
        if stage_name == "proxy":
            if "effective_steps" in kwargs:
                try:
                    steps = int(kwargs.get("effective_steps", 16))
                    if proxy_steps_env is not None:
                        steps = min(steps, int(proxy_steps_env))
                    else:
                        steps = min(steps, 12)
                    kwargs["effective_steps"] = steps
                except Exception:
                    kwargs["effective_steps"] = 12
            _old_g = globals().get("GUIDANCE_SCALE", 6.0)
            try:
                new_g = float(proxy_g_env) if proxy_g_env is not None else 4.5
            except Exception:
                new_g = 4.5
            globals()["GUIDANCE_SCALE"] = new_g
            if disable_pb:
                os.environ["DIFFUSERS_DISABLE_PROGRESS_BARS"] = "1"
            try:
                res = _orig_generate_image_with_screening(*args, **kwargs, pipe_override=pipe_override)
            finally:
                globals()["GUIDANCE_SCALE"] = _old_g
                if disable_pb:
                    os.environ.pop("DIFFUSERS_DISABLE_PROGRESS_BARS", None)
            return res
        else:
            return _orig_generate_image_with_screening(*args, **kwargs, pipe_override=pipe_override)
    print("[speedfix] proxy turbo wrapper active")
except Exception as _e:
    print("[speedfix] wrap screening skipped:", _e)

try:
    import torch, os, contextlib, uuid, inspect
    from PIL import Image
except Exception:
    pass

_ORIG_EMB_PATH = None
_ORIG_EMB_VEC = None
def compare_with_transformers(p1: str, p2: str, device: str=DEVICE) -> float:
    load_clip(device)
    global _ORIG_EMB_PATH, _ORIG_EMB_VEC
    if (_ORIG_EMB_PATH != p1) or (_ORIG_EMB_VEC is None):
        im1 = Image.open(p1).convert("RGB")
        _ORIG_EMB_VEC = _clip_image_feat(im1).cpu()
        _ORIG_EMB_PATH = p1
    im2 = Image.open(p2).convert("RGB")
    v2 = _clip_image_feat(im2).cpu()
    a = _ORIG_EMB_VEC / max(1e-6, _ORIG_EMB_VEC.norm())
    b = v2 / max(1e-6, v2.norm())
    return float((a @ b).item() * 100.0)

_generate_image_txt2img_with_controls_orig = globals().get("generate_image_txt2img_with_controls")

def generate_image_txt2img_with_controls(
    pipe, prompt, out_path, ref_image, control_image, control_type,
    steps, w, h, guidance=GUIDANCE_SCALE, neg="low quality, blurry, dark",
    cn_scale=0.3, cn_start=0.0, cn_end=0.2, ipa_scale=0.0, has_cn=False, has_ipa=False):
    use_amp = torch.cuda.is_available()
    cm = torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16) if use_amp else contextlib.nullcontext()
    if max(w,h) >= 1024 and steps > 22:
        steps = 22
    extra = {}
    cn_kwargs = {}
    if has_cn and isinstance(control_image, Image.Image):
        try:
            sig = inspect.signature(pipe.__call__); params = sig.parameters
            key = "image" if "image" in params else ("control_image" if "control_image" in params else None)
            if key:
                cn_kwargs = {key: control_image, "controlnet_conditioning_scale": cn_scale,
                             "control_guidance_start": [cn_start], "control_guidance_end": [cn_end]}
        except Exception:
            pass
    with torch.no_grad(), cm:
        try:
            if has_ipa:
                import inspect as _inspect
                _sig = _inspect.signature(pipe.__call__)
                if 'image' in _sig.parameters:
                    _ipa_img = ref_image
                    if _ipa_img is None:
                        try:
                            from PIL import Image as _I
                            _orig_path = globals().get('ORIGINAL_IMAGE_PATH', None)
                            if _orig_path:
                                _ipa_img = _I.open(_orig_path).convert('RGB')
                        except Exception:
                            _ipa_img = None
                    if _ipa_img is not None:
                        if not isinstance(extra, dict):
                            extra = {}
                        extra['image'] = _ipa_img
        except Exception as _e_inj:
            print('[rootfix] warn:', _e_inj)
        out = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=int(steps),
                   guidance_scale=float(guidance), width=int(w), height=int(h),
                   output_type="pil", return_dict=True, **cn_kwargs, **extra)
    out.images[0].save(out_path)

_generate_with_screening_fullflow_orig = globals().get("generate_with_screening_fullflow")

def generate_with_screening_fullflow(
    prompt: str, original_path: str, out_path: str, pipe, ref_pil, control_img, control_kind,
    has_cn: bool, has_ipa: bool, stage_name: str, width: int, height: int, steps: int,
    ssim_threshold: float = SSIM_THRESHOLD, clip_threshold: float = CLIP_THRESHOLD, max_retries: int = MAX_RETRIES,
):
    local_has_cn, local_has_ipa = has_cn, has_ipa
    local_guidance, local_steps = GUIDANCE_SCALE, steps
    if str(stage_name).lower() == "proxy":
        local_has_cn = True
        local_has_ipa = False
        try: env_steps = int(os.environ.get("PROXY_STEPS", "12"))
        except: env_steps = 12
        local_steps = min(int(steps), env_steps, 12)
        try: local_guidance = float(os.environ.get("PROXY_GUIDANCE", "4.5"))
        except: local_guidance = 4.5

    last_ssim = last_clip = None
    for attempt in range(1, int(max_retries) + 1):
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.png")
        generate_image_txt2img_with_controls(pipe, prompt, tmp_path, ref_pil,
                                             (control_img if local_has_cn else None), control_kind,
                                             local_steps, width, height, local_guidance,
                                             "low quality, blurry, dark",
                                             0.3, 0.0, 0.2, 0.0, local_has_cn, local_has_ipa)
        if is_near_black(tmp_path):
            try: os.remove(tmp_path)
            except: pass
            continue
        try: ssim_score = compute_ssim_pair(original_path, tmp_path)
        except Exception:
            try: os.remove(tmp_path)
            except: pass
            last_ssim = None
            continue
        if ssim_score < ssim_threshold:
            try: os.remove(tmp_path)
            except: pass
            last_ssim = ssim_score
            continue
        try: clip_score = compare_with_transformers(original_path, tmp_path, device=DEVICE)
        except Exception:
            try: os.remove(tmp_path)
            except: pass
            last_clip = None
            continue
        if clip_score < clip_threshold:
            try: os.remove(tmp_path)
            except: pass
            last_clip = clip_score
            continue
        try: os.replace(tmp_path, out_path)
        except Exception:
            Image.open(tmp_path).save(out_path)
            try: os.remove(tmp_path)
            except: pass
        return True, float(clip_score), {"attempt": attempt, "ssim": float(ssim_score), "clip": float(clip_score), "path": out_path}
    return False, 0.0, {"attempts": int(max_retries), "last_ssim": last_ssim, "last_clip": last_clip}