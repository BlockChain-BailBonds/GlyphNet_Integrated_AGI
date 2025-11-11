# Codex — javascript

### Ωfn_def

```javascript
function name(args) { body }
```

### Ωclass_def

```javascript
class Name { }
```

### Ωif_cond

```javascript
if (cond) { }
```

### Ωelse

```javascript
else { }
```

### Ωloop_for

```javascript
for (const x of xs) {}
```

### Ωreturn

```javascript
return v;
```

### Ωprint

```javascript
console.log(v)
```

### Ωassign

```javascript
x = y;
```

### Ωadd

```javascript
a + b
```

### Ωsub

```javascript
a - b
```

### Ωmul

```javascript
a * b
```

### Ωdiv

```javascript
a / b
```

### Ωeq

```javascript
a === b
```

### Ωneq

```javascript
a !== b
```

### Ωlt

```javascript
a < b
```

### Ωgt

```javascript
a > b
```

### Ωle

```javascript
a <= b
```

### Ωge

```javascript
a >= b
```

### Ωapl_rho

```javascript
/* reshape: use typed arrays or helper */
```

### Ωapl_iota

```javascript
Array.from
```

### Ωapl_floor

```javascript
Math.floor
```

### Ωapl_ceiling

```javascript
Math.ceil
```

### Ωapl_gradeup

```javascript
[...arr.keys()].sort((i,j)=>arr[i]-arr[j])
```

### Ωapl_gradedown

```javascript
[...arr.keys()].sort((i,j)=>arr[j]-arr[i])
```

### Ωapl_reduce

```javascript
Array.prototype.reduce
```

### Ωapl_scan

```javascript
custom scan
```

### Ωapl_enclose

```javascript
[x]
```

### Ωapl_disclose

```javascript
x[0]
```

### Ωapl_ravel

```javascript
arr.flat ? arr.flat() : arr.reduce((a,v)=>a.concat(v), [])
```

### Ωapl_take

```javascript
arr.slice(0,n)
```

### Ωapl_drop

```javascript
arr.slice(n)
```

### Ωapl_transpose

```javascript
transpose helper
```

### Ωapl_outer

```javascript
nested map
```

### α

```javascript
for (let {0}=0; {0}<{1}; {0}++) {{
  {2}
}}
```

### β

```javascript
function {0}({1}) {{
  {2}
}}
```

### γ

```javascript
import {0} from '{1}';
```

### δ

```javascript
if ({0}) {{
  {1}
}}
```

### ε

```javascript
console.log({0});
```

### □

```javascript
{0}
```

### Ωjs_async_fetch

```javascript
const {0} = await fetch('{1}').then(r => r.{2}())
```

### Ωjs_class_ctor

```javascript
class {0} {{
  constructor({1}) {{ {2} }}
}}
```

### Ωjs_export_default

```javascript
export default {0}
```

### Ωjs_for_of

```javascript
for (const {0} of {1}) {{ {2} }}
```

### Ωjs_import_named

```javascript
import {{ {0} }} from '{1}'
```

### Ωjs_template_log

```javascript
console.log(`${0}`)
```

### Ωcli_echo

```javascript
#!/usr/bin/env node
console.log(process.argv.slice(2).join(" "));

```

### Ωhttp_server_8000

```javascript
const http = require('http');
http.createServer((req, res) => {{
    res.writeHead(200, {{{{'Content-Type': 'text/plain'}}}});
    res.end('ok');
}}).listen(8000);

```

### Ωhttp_get_print

```javascript
fetch({0}).then(r=>r.text()).then(t=>console.log(t.slice(0,200)));

```

### Ωsum_numbers

```javascript
#!/usr/bin/env node
console.log(process.argv.slice(2).reduce((a,x)=>a+parseInt(x,10),0));

```

### Ωjson_roundtrip

```javascript
const fs=require('fs'); const data=fs.readFileSync(0,'utf8');
console.log(JSON.stringify(JSON.parse(data)));

```

### Ωfile_cat

```javascript
const fs=require('fs'); console.log(fs.readFileSync({0},'utf8'));
```

### Ωtimer_tick_5

```javascript
setInterval(()=>console.log("tick"), 5000);
```

### Ω3_stdin_upper_stdout

```javascript
const fs = require('fs');
const data = fs.readFileSync(0,'utf8');
process.stdout.write(data.toUpperCase());

```

### Ω3_file_grep_count

```javascript
const fs=require('fs');
const path={0}; const pat=new RegExp({1});
let n=0;
fs.readFileSync(path,'utf8').split(/\r?\n/).forEach(l=>{ if(pat.test(l)) n++; });
console.log(n);

```

### Ω3_http_json_key_print

```javascript
(async()=>{
  const res = await fetch({0});
  const obj = await res.json();
  console.log(obj[{1}]);
})();

```

### Ω3_csv_col_sum

```javascript
const fs=require('fs'); const path={0}; const col=parseInt({1},10);
let s=0; fs.readFileSync(path,'utf8').trim().split(/\r?\n/).forEach(line=>{
  const r=line.split(',');
  const v=parseFloat(r[col]); if(!Number.isNaN(v)) s+=v;
});
console.log(s);

```

### Ω3_replace_inplace

```javascript
const fs=require('fs'); const path={0}; const pat=new RegExp({1},'g'); const repl={2};
const s=fs.readFileSync(path,'utf8'); fs.writeFileSync(path, s.replace(pat, repl));

```

### Ω3_jsonl_filter_map_write

```javascript
const fs=require('fs'); const [inp,key,val,outp] = [{0},{1},{2},{3}];
const out = fs.createWriteStream(outp); 
fs.readFileSync(inp,'utf8').split(/\r?\n/).forEach(l=>{
  if(!l.trim()) return;
  const o=JSON.parse(l);
  if(String(o[key])===String(val)) out.write(JSON.stringify(o)+"\n");
});
out.end();

```

### Ω3_stdin_unique_count

```javascript
const fs=require('fs'); 
const s = fs.readFileSync(0,'utf8').split(/\r?\n/);
console.log(new Set(s).size);

```

### Ω3_markdown_toc

```javascript
const fs=require('fs');
const lines = fs.readFileSync(0,'utf8').split(/\r?\n/);
for(const l of lines){
  const m = l.match(/^(#{1,6})\s+(.*)$/);
  if(m){ const level=m[1].length; const title=m[2].trim();
    console.log("  ".repeat(level-1)+"- "+title); }
}

```

### Ωenv_get_print

```javascript
console.log(process.env[{{0}}] !== undefined ? process.env[{{0}}] : {{1}});

```

### Ωenv_set_current

```javascript
process.env[{{0}}] = {{1}}
```

### Ωenv_unset

```javascript
delete process.env[{{0}}]
```

### Ωenv_list

```javascript
Object.entries(process.env).forEach(([k,v]) => console.log(k + "=" + v));

```

### Ωenv_cwd_print

```javascript
console.log(process.cwd())
```

### Ωenv_chdir

```javascript
process.chdir({{0}})
```

### Ωenv_path_prepend

```javascript
const seg = {{0}};
process.env.PATH = seg + (process.env.PATH ? ":" + process.env.PATH : "");

```

### Ωenv_guard_required

```javascript
if (process.env[{{0}}] === undefined) {{ console.error("Missing required env: " + {{0}}); process.exit(1); }}

```

### Ωenv_load_dotenv_min

```javascript
const fs=require('fs'); const path={{0}};
fs.readFileSync(path,'utf8').split(/\r?\n/).forEach(l=>{{
    const line=l.trim(); if(!line || line.startsWith('#')) return;
    const i=line.indexOf('='); if(i<0) return;
    const k=line.slice(0,i).trim(); let v=line.slice(i+1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1,-1);
    process.env[k]=v;
}});

```

### Ωenv_export_file

```javascript
const fs=require('fs'); const out={{0}};
const lines = Object.keys(process.env).sort().map(k => k + "=" + String(process.env[k]).replace(/\n/g,"\\n"));
fs.writeFileSync(out, lines.join("\n") + "\n");

```

### Ωlib_base_js

```javascript
// ---- Ωlib_base_js: Node (no external deps) ----
const fs = require('fs');
const http = require('http'); const https = require('https');
const {{ execFileSync, spawnSync }} = require('child_process');
const crypto = require('crypto');

// CLI parse
function cliParse(argv = process.argv.slice(2)){{
  const opts = {{}}; const pos = [];
  for(let i=0;i<argv.length;i++){{
    const a = argv[i];
    if(a.startsWith('--')){{
      const eq = a.indexOf('=');
      if(eq>2){{ opts[a.slice(2,eq)] = a.slice(eq+1); continue; }}
      const k = a.slice(2);
      if(i+1<argv.length && !argv[i+1].startsWith('-')){{ opts[k]=argv[++i]; }} else {{ opts[k]='true'; }}
      continue;
    }}
    if(a.startsWith('-') && a.length>1){{
      const k = a[1];
      if(i+1<argv.length && !argv[i+1].startsWith('-')){{ opts[k]=argv[++i]; }} else {{ opts[k]='true'; }}
      continue;
    }}
    pos.push(a);
  }}
  return {{opts, pos}};
}}

// JSON log
function log(level, msg, fields={{}}){{
  const rec = Object.assign({{ts: new Date().toISOString(), level, msg}}, fields);
  process.stderr.write(JSON.stringify(rec)+'\n');
}}

// FS
const readText = p => fs.readFileSync(p,'utf8');
const writeText = (p,s)=>{{ fs.mkdirSync(require('path').dirname(p), {{recursive:true}}); fs.writeFileSync(p,s,'utf8'); }};
const mkdirP = d => fs.mkdirSync(d,{{recursive:true}});
const exists = p => fs.existsSync(p);

// JSON
const jsonLoad = p => JSON.parse(readText(p));
const jsonDump = (p,obj)=> writeText(p, JSON.stringify(obj,null,2));

// SHA256
function sha256File(p){{
  const h = crypto.createHash('sha256');
  const data = fs.readFileSync(p);
  h.update(data);
  return h.digest('hex');
}}
const sha256Str = s => crypto.createHash('sha256').update(Buffer.from(s,'utf8')).digest('hex');

// HTTP GET (supports http/https)
function httpGet(url){{
  return new Promise((resolve,reject)=>{{
    const lib = url.startsWith('https') ? https : http;
    lib.get(url, res => {{
      if (res.statusCode && res.statusCode >= 400) {{ reject(new Error('HTTP '+res.statusCode)); return; }}
      let data=''; res.setEncoding('utf8'); res.on('data', c=> data+=c); res.on('end', ()=> resolve(data));
    }}).on('error', reject);
  }});
}}

// Retry
async function retry(fn, attempts=3, delay=500){{
  let d=delay; let last;
  for(let i=0;i<attempts;i++){{
    try{{ return await fn(); }} catch(e){{ last=e; await new Promise(r=>setTimeout(r,d)); d*=2; }}
  }}
  throw last;
}}

module.exports = {{ cliParse, log, readText, writeText, mkdirP, exists, jsonLoad, jsonDump, sha256File, sha256Str, httpGet, retry }};

```

### Ωcx_sign_hmac_sha256_js

```javascript
const crypto = require('crypto'); const secret={0}; const msg={1};
const sig = crypto.createHmac('sha256', Buffer.from(secret,'utf8')).update(Buffer.from(msg,'utf8')).digest('hex');
console.log(sig);

```
