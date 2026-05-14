"""skim-compressor plugin v0.4"""
from __future__ import annotations
import json, logging, os, re, subprocess, tempfile, threading, time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
logger = logging.getLogger(__name__)
_MODE = os.environ.get("HERMES_CONTEXT_COMPRESSION", "shadow").lower()
_THRESHOLDS = {"file_read_lines": int(os.environ.get("SKIM_THRESHOLD_FILE_LINES","300")),
    "git_diff_lines": int(os.environ.get("SKIM_THRESHOLD_GIT_LINES","50")),
    "test_output_lines": int(os.environ.get("SKIM_THRESHOLD_TEST_LINES","50")),
    "token_limit": int(os.environ.get("SKIM_THRESHOLD_TOKENS","8000"))}
_SKIM_BIN = os.environ.get("SKIM_BIN","/usr/local/bin/skim")
_TRACE_DIR = Path(os.path.expanduser("~/.hermes/skim-traces")); _TRACE_DIR.mkdir(parents=True, exist_ok=True)
_stats_lock = threading.Lock()
_stats = {"total_outputs":0,"compressed":0,"compression_rejected":0,"total_original_tokens":0,
    "total_compressed_tokens":0,"missing_critical_info":0,"full_context_requested":0,"symbol_guard_failures":0}

# Per-type gating (v0.4): allows on/shadow/off per compression type
# HERMES_CONTEXT_COMPRESSION_ON_TYPES: comma-separated types to compress in on mode
# HERMES_CONTEXT_COMPRESSION_SHADOW_TYPES: comma-separated types to shadow-log
# HERMES_CONTEXT_COMPRESSION_OFF_TYPES: comma-separated types to skip entirely
# Types: file_read, git, test, generic
_ON_TYPES = set(t.strip() for t in os.environ.get("HERMES_CONTEXT_COMPRESSION_ON_TYPES", "file_read").split(",") if t.strip())
_SHADOW_TYPES = set(t.strip() for t in os.environ.get("HERMES_CONTEXT_COMPRESSION_SHADOW_TYPES", "git,test,generic").split(",") if t.strip())
_OFF_TYPES = set(t.strip() for t in os.environ.get("HERMES_CONTEXT_COMPRESSION_OFF_TYPES", "").split(",") if t.strip())

def _get_effective_mode(category: str) -> str:
    """Return the effective mode for a given compression category."""
    if category in _OFF_TYPES: return "off"
    if _MODE == "on":
        if category in _ON_TYPES: return "on"
        if category in _SHADOW_TYPES: return "shadow"
        return "off"
    if _MODE == "shadow":
        return "shadow"
    return "off"

_FILE_READ_PATTERNS = [r"^cat\s+",r"^head\s+",r"^tail\s+",r"^sed\s+",r"^less\s+",r"^more\s+"]
_GIT_PATTERNS = [r"^git\s+diff",r"^git\s+show",r"^git\s+log",r"^git\s+status",r"^git\s+stash\s+list"]
_TEST_PATTERNS = [r"^pytest",r"^python\s+-m\s+pytest",r"^npm\s+test",r"^yarn\s+test",
    r"^cargo\s+test",r"^cargo\s+nextest",r"^go\s+test",r"^vitest",r"^jest"]
_SYMBOL_PATTERNS = {"python":[(r"^\s*class\s+(\w+)","class"),(r"^\s*(?:async\s+)?def\s+(\w+)","function"),
    (r"^import\s+(\w+)","import"),(r"^from\s+(\w+)\s+import","import"),
    (r"^([A-Z_][A-Z0-9_]*)\s*=","constant"),(r"^if\s+__name__\s*==","main")],
    "typescript":[(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)","class"),
    (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)","function"),
    (r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=","symbol"),
    (r"^\s*export\s+(?:default\s+)?","export"),(r"^\s*import\s+","import"),
    (r"^\s*(?:export\s+)?interface\s+(\w+)","interface"),(r"^\s*(?:export\s+)?type\s+(\w+)","type")],
    "rust":[(r"^\s*(?:pub\s+)?fn\s+(\w+)","function"),(r"^\s*(?:pub\s+)?struct\s+(\w+)","struct"),
    (r"^\s*(?:pub\s+)?enum\s+(\w+)","enum"),(r"^\s*(?:pub\s+)?trait\s+(\w+)","trait"),
    (r"^\s*impl\s+","impl"),(r"^\s*mod\s+(\w+)","mod"),(r"^\s*use\s+(\w+)","use")]}
def _detect_language(fn:str)->str:
    return {".py":"python",".pyw":"python",".ts":"typescript",".tsx":"typescript",
        ".js":"typescript",".jsx":"typescript",".rs":"rust"}.get(Path(fn).suffix.lower(),"python")
def _strip_line_numbers(text:str)->str:
    return "\n".join(line[re.match(r"^\s*\d+\|",line).end():] if re.match(r"^\s*\d+\|",line) else line for line in text.split("\n"))
def _extract_symbols(text:str,lang:str)->Set[str]:
    syms=set(); pats=_SYMBOL_PATTERNS.get(lang,_SYMBOL_PATTERNS["python"]); clean=_strip_line_numbers(text)
    for line in clean.split("\n"):
        for pat,kind in pats:
            m=re.match(pat,line)
            if m: syms.add(f"{kind}:{m.group(1) if m.lastindex else kind}"); break
    return syms
def _symbol_recall(raw:Set[str],comp:Set[str])->float: return len(raw&comp)/len(raw) if raw else 1.0
def _est(text:str)->int: return len(text)//4
def _lines(text:str)->int: return text.count("\n")+(1 if text and not text.endswith("\n") else 0)
def _classify(cmd:str)->str:
    # Check all commands in chain (split on &&, ;, |)
    # Use the LAST meaningful command for classification
    segments = re.split(r'\s*&&\s*|\s*;\s*', cmd.strip())
    last_cmd = segments[-1].strip() if segments else cmd.strip()
    # Also check the full command for patterns
    for c in [last_cmd, cmd.strip()]:
        for p in _FILE_READ_PATTERNS:
            if re.match(p,c): return "file_read"
        for p in _GIT_PATTERNS:
            if re.match(p,c): return "git"
        for p in _TEST_PATTERNS:
            if re.match(p,c): return "test"
    return "generic"
def _should_compress(cmd:str,out:str)->bool:
    if not out or len(out)<500: return False
    cat=_classify(cmd); ln=_lines(out); tok=_est(out)
    if tok<_THRESHOLDS["token_limit"]//2: return False
    k={"file_read":"file_read_lines","git":"git_diff_lines","test":"test_output_lines"}.get(cat)
    if k and ln>=_THRESHOLDS[k]: return True
    if tok>=_THRESHOLDS["token_limit"]: return True
    return False
def _fname_from_cmd(cmd:str)->str:
    for p in cmd.split("|")[0].strip().split()[1:]:
        if p.startswith("-"): continue
        if "/" in p or "." in p or p.startswith("~"): return p
    return "output.txt"
_GIT_KEEP=[r"^diff --git ",r"^index ",r"^new file ",r"^deleted file ",
    r"^rename from ",r"^rename to ",r"^--- ",r"^\+\+\+ ",r"^@@ ",r"^\+",r"^-"]
_GIT_IMP=["ERROR","TODO","FIXME","SECURITY","BREAKING","WARNING","BUG","HACK","XXX"]
def _compress_git(raw:str)->str:
    lines=raw.split("\n"); out=[]; cbuf=[]; hunk=False
    for line in lines:
        is_hdr=any(re.match(p,line) for p in _GIT_KEEP[:10])
        is_chg=line.startswith("+") or line.startswith("-")
        is_imp=any(p in line for p in _GIT_IMP)
        if is_hdr or is_imp:
            if cbuf: out.extend(cbuf); cbuf=[]
            out.append(line)
            if line.startswith("@@"): hunk=True
            continue
        if is_chg:
            if cbuf: out.extend(cbuf); cbuf=[]
            out.append(line); continue
        if hunk: cbuf.append(line)
        if len(cbuf)>3: cbuf.pop(0)
    if cbuf: out.extend(cbuf)
    return "\n".join(out)
def _run_skim_file(content:str,fn:str="output.txt")->Optional[str]:
    try:
        bn=os.path.basename(fn) if fn else "output.txt"
        with tempfile.NamedTemporaryFile(mode="w",suffix=f"_{bn}",delete=False) as f: f.write(content); tp=f.name
        r=subprocess.run([_SKIM_BIN,tp,"--mode","structure"],capture_output=True,text=True,timeout=15)
        os.unlink(tp)
        if r.returncode==0 and r.stdout.strip(): return r.stdout
    except Exception as e: logger.debug(f"skim file failed: {e}")
    return None
def _run_skim_generic(content:str)->Optional[str]:
    try:
        r=subprocess.run([_SKIM_BIN,"-","--filename","output.txt","--mode","structure"],
            input=content,capture_output=True,text=True,timeout=15)
        if r.returncode==0 and r.stdout.strip(): return r.stdout
    except Exception as e: logger.debug(f"skim generic failed: {e}")
    return None
def _compress(cmd:str,out:str,fn_hint:str="")->Tuple[Optional[str],dict]:
    g={"quality_guard_passed":True,"missing_symbols":[],"symbol_recall":1.0,
       "compressor_backend":"none","compression_rejected":False,"reject_reason":""}
    cat=_classify(cmd)
    if cat=="git":
        g["compressor_backend"]="conservative_diff"; c=_compress_git(out)
        if _est(c)>=_est(out): g["compression_rejected"]=True; g["reject_reason"]="expanded_output"; return None,g
        return c,g
    if cat=="file_read":
        fn=fn_hint or _fname_from_cmd(cmd); lang=_detect_language(fn)
        g["compressor_backend"]=f"skim_structure_{lang}"; raw_sym=_extract_symbols(out,lang)
        c=_run_skim_file(out,fn)
        if c:
            comp_sym=_extract_symbols(c,lang); rec=_symbol_recall(raw_sym,comp_sym)
            g["symbol_recall"]=round(rec,4); g["missing_symbols"]=sorted(raw_sym-comp_sym)[:20]
            if rec<0.98:
                g["quality_guard_passed"]=False; g["compression_rejected"]=True
                g["reject_reason"]=f"symbol_recall_{rec:.3f}"
                with _stats_lock: _stats["symbol_guard_failures"]+=1
                return None,g
            if _est(c)>=_est(out): g["compression_rejected"]=True; g["reject_reason"]="expanded_output"; return None,g
            return c,g
        g["compressor_backend"]="skim_structure_failed"; return None,g
    g["compressor_backend"]="skim_generic"; c=_run_skim_generic(out)
    if c:
        if _est(c)>=_est(out): g["compression_rejected"]=True; g["reject_reason"]="expanded_output"; return None,g
        return c,g
    return None,g
def _save_trace(tid,cmd,orig,comp,g):
    try:
        rt=_est(orig); ct=_est(comp) if comp else rt; acc=comp is not None and not g.get("compression_rejected",False)
        t={"timestamp":time.time(),"mode":_MODE,"command":cmd,"raw_tokens":rt,"raw_lines":_lines(orig),
           "compressed_tokens":ct,"compressed_lines":_lines(comp) if comp else _lines(orig),
           "compression_ratio":round(1-ct/max(rt,1),3),"compression_accepted":acc,
           "compression_rejected":g.get("compression_rejected",False),"reject_reason":g.get("reject_reason",""),
           "quality_guard_passed":g.get("quality_guard_passed",True),"symbol_recall":g.get("symbol_recall"),
           "missing_symbols":g.get("missing_symbols",[]),"compressor_backend":g.get("compressor_backend","none"),
           "original_preview":orig[:500],"compressed_preview":comp[:500] if comp else None}
        (_TRACE_DIR/f"{tid}.json").write_text(json.dumps(t,indent=2,ensure_ascii=False))
    except Exception as e: logger.debug(f"trace save failed: {e}")
def _on_transform(tool_name="",args=None,result=None,task_id="",session_id="",tool_call_id="",**_):
    if not isinstance(result,(str,dict)): return None
    cat = _classify(args.get("command","") if tool_name=="terminal" and isinstance(args,dict) else tool_name)
    eff_mode = _get_effective_mode(cat)
    if eff_mode == "off": return None
    ot,cmd=None,""; _r=None
    if isinstance(result,str):
        try: _r=json.loads(result)
        except: _r=None
    elif isinstance(result,dict): _r=result
    if tool_name=="terminal" and isinstance(args,dict):
        cmd=args.get("command","")
        if _r and isinstance(_r,dict): ot=_r.get("output","")
        elif isinstance(result,str): ot=result
    elif tool_name=="execute_code":
        if _r and isinstance(_r,dict): ot=_r.get("output","")
    elif tool_name=="read_file":
        if _r and isinstance(_r,dict): ot=_r.get("content","")
    if not ot or not ot.strip(): return None
    with _stats_lock: _stats["total_outputs"]+=1; _stats["total_original_tokens"]+=_est(ot)
    if not _should_compress(cmd or tool_name,ot): return None
    tid=f"{int(time.time())}_{tool_call_id or 'unknown'}"
    fn=args.get("path","") if tool_name=="read_file" and isinstance(args,dict) else ""
    comp,guard=_compress(cmd or f"cat {fn or 'output.txt'}",ot,fn)
    _save_trace(tid,cmd or tool_name,ot,comp,guard)
    if eff_mode=="shadow":
        with _stats_lock:
            _stats["compressed"]+=1
            if comp and not guard.get("compression_rejected"): _stats["total_compressed_tokens"]+=_est(comp)
            else: _stats["total_compressed_tokens"]+=_est(ot)
            if guard.get("compression_rejected"): _stats["compression_rejected"]+=1
        return None
    if _MODE=="on" and comp and not guard.get("compression_rejected"):
        with _stats_lock: _stats["compressed"]+=1; _stats["total_compressed_tokens"]+=_est(comp)
        ot2=_est(ot); ct=_est(comp); r=round((1-ct/max(ot2,1))*100,1)
        w=f"[Compressed by Skim v0.4] {ot2}->{ct}T ({r}%) backend={guard.get('compressor_backend','?')}\nfull_context://tool-output/{tid}\n\n---\n\n{comp}"
        if isinstance(result,dict): result["output"]=w; result.get("content") and result.__setitem__("content",w); return json.dumps(result,ensure_ascii=False)
        return w
    return None
def _handle_slash(args:str)->str:
    global _MODE; parts=args.strip().split(); sub=parts[0] if parts else "status"
    if sub=="status":
        with _stats_lock:
            r=round((1-_stats["total_compressed_tokens"]/max(_stats["total_original_tokens"],1))*100,1) if _stats["total_original_tokens"]>0 else 0
            return f"Skim v0.4 | mode={_MODE} | proc={_stats['total_outputs']} | comp={_stats['compressed']} | rej={_stats['compression_rejected']} | sym_fail={_stats['symbol_guard_failures']} | red={r}%"
    if sub=="mode" and len(parts)>=2: _MODE=parts[1].lower(); return f"Mode: {_MODE}"
    return f"Usage: /skim status|mode|reset|traces | mode={_MODE}"
def register(ctx):
    ctx.register_hook("transform_tool_result",_on_transform)
    ctx.register_command("skim",handler=_handle_slash,description="Skim v0.4")
    logger.info(f"skim-compressor v0.4 loaded (mode={_MODE})")
