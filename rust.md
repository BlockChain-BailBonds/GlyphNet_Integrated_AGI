# Codex — rust

### α

```rust
for {0} in 0..{1} {{
{2}
}}
```

### γ

```rust
fn main(){{
{0}
}}
```

### ε

```rust
println!("{{}}", {0});
```

### □

```rust
{0}
```

### Ωrs_fn

```rust
fn {0}({1}) -> {2} {{ {3} }}
```

### Ωrs_match_result

```rust
match {0} {{ Ok({1}) => {2}, Err({3}) => {4} }}
```

### Ωrs_vec

```rust
let mut {0}: Vec<{1}> = Vec::new();
```

### Ωcli_echo

```rust
use std::env;
fn main(){
    let args: Vec<String> = env::args().skip(1).collect();
    println!("{}", args.join(" "));
}

```

### Ωsum_numbers

```rust
use std::env;
fn main(){
    let mut s = 0i64;
    for a in env::args().skip(1){
        if let Ok(v) = a.parse::<i64>() { s += v; }
    }
    println!("{}", s);
}

```

### Ωfile_cat

```rust
use std::fs;
fn main(){
    let p = std::env::args().nth(1).expect("path");
    let s = fs::read_to_string(p).expect("read");
    print!("{}", s);
}

```

### Ωrs_stdin_count_lines

```rust
use std::io::{self, Read};
fn main(){
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    println!("{}", s.lines().count());
}

```

### Ωenv_get_print

```rust
use std::env;
fn main(){{
    let v = env::var({0}).unwrap_or({1}.to_string());
    println!("{{}}", v);
}}

```

### Ωenv_set_current

```rust
use std::env;
fn main(){{ env::set_var({0}, {1}); }}

```

### Ωenv_unset

```rust
use std::env;
fn main(){{ env::remove_var({0}); }}

```

### Ωenv_list

```rust
fn main(){{ for (k,v) in std::env::vars() {{ println!("{{}}={{}}", k, v); }} }}

```

### Ωenv_cwd_print

```rust
fn main(){{ println!("{{}}", std::env::current_dir().unwrap().display()); }}

```

### Ωenv_chdir

```rust
fn main(){{ std::env::set_current_dir({0}).unwrap(); }}

```

### Ωenv_path_prepend

```rust
use std::env;
fn main(){{
    let seg = {{0}};
    let p = env::var("PATH").unwrap_or_default();
    let newp = if p.is_empty() {{ seg.to_string() }} else {{ format!("{{}}:{{}}", seg, p) }};
    env::set_var("PATH", newp);
}}

```

### Ωenv_guard_required

```rust
use std::env; use std::process::exit;
fn main(){{ if env::var({0}).ok().is_none(){{ eprintln!("Missing required env: {{}}", {0}); exit(1); }} }}

```

### Ωlib_base_rust

```rust
// ---- Ωlib_base_rust: std-only subset (no external crates) ----
use std::env;
use std::fs;
use std::io::{{self, Read}};
use std::time::{{SystemTime, UNIX_EPOCH}};
use std::process::{{Command, Stdio}};

// CLI parse
pub fn cli_parse(args: Option<Vec<String>>) -> (std::collections::HashMap<String,String>, Vec<String>) {{
    let argv = args.unwrap_or_else(|| env::args().skip(1).collect());
    let mut opts = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let mut i = 0;
    while i < argv.len() {{
        let a = &argv[i];
        if a.starts_with("--") {{
            if let Some(eq) = a.find('=') {{
                opts.insert(a[2..eq].to_string(), a[eq+1..].to_string()); i+=1; continue;
            }}
            let k = a[2..].to_string();
            if i+1 < argv.len() && !argv[i+1].starts_with('-') {{ opts.insert(k, argv[i+1].clone()); i+=2; continue; }}
            opts.insert(k, "true".to_string()); i+=1; continue;
        }} else if a.starts_with('-') && a.len()>1 {{
            let k = a[1..2].to_string();
            if i+1 < argv.len() && !argv[i+1].starts_with('-') {{ opts.insert(k, argv[i+1].clone()); i+=2; continue; }}
            opts.insert(k, "true".to_string()); i+=1; continue;
        }} else {{
            pos.push(a.clone()); i+=1; continue;
        }}
    }}
    (opts, pos)
}}

// Log (plain line with epoch seconds)
pub fn log(level: &str, msg: &str){{
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    eprintln!("{{{{"ts":{{}},"level":"{{}}","msg":"{{}}"}}}}", ts, level, msg.replace('"',"\""));
}}

pub fn read_text(p: &str) -> io::Result<String> {{ fs::read_to_string(p) }}
pub fn write_text(p: &str, s: &str) -> io::Result<()> {{ fs::write(p, s) }}

```
