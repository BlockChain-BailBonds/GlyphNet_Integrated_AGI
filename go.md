# Codex — go

### α

```go
for {0} := 0; {0} < {1}; {0}++ {{
{2}
}}
```

### β

```go
func {0}({1}) {2} {{
{3}
}}
```

### γ

```go
package main
import "fmt"
func main() {{
{0}
}}
```

### ε

```go
fmt.Println({0})
```

### □

```go
{0}
```

### Ωgo_err_check

```go
if err != nil {{ {0} }}
```

### Ωgo_func

```go
func {0}({1}) {2} {{ {3} }}
```

### Ωgo_http_get

```go
resp, err := http.Get("{0}")
```

### Ωcli_echo

```go
package main
import (
    "fmt"
    "os"
    "strings"
)
func main(){{
    fmt.Println(strings.Join(os.Args[1:], " "))
}}

```

### Ωhttp_server_8000

```go
package main
import (
    "net/http"
    "log"
)
func main(){{
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request){{
        w.Write([]byte("ok"))
    }})
    log.Fatal(http.ListenAndServe(":8000", nil))
}}

```

### Ωhttp_get_print

```go
package main
import (
    "fmt"
    "io"
    "net/http"
)
func main(){{
    resp, err := http.Get({0})
    if err != nil {{{{ panic(err) }}}}
    defer resp.Body.Close()
    b, _ := io.ReadAll(resp.Body)
    if len(b) > 200 {{{{ fmt.Println(string(b[:200])) }}}} else {{{{ fmt.Println(string(b)) }}}}
}}

```

### Ωsum_numbers

```go
package main
import (
    "fmt"
    "os"
    "strconv"
)
func main(){{
    s := 0
    for _, a := range os.Args[1:] {{
        if v, err := strconv.Atoi(a); err == nil {{ s += v }}
    }}
    fmt.Println(s)
}}

```

### Ωfile_cat

```go
package main
import (
    "fmt"
    "os"
)
func main(){{
    b, _ := os.ReadFile({0})
    fmt.Print(string(b))
}}

```

### Ω3_http_json_key_print

```go
package main
import ("encoding/json"; "io"; "net/http"; "fmt")
func main(){{
    resp, err := http.Get({0}); if err!=nil {{ fmt.Println(err); return }}
    defer resp.Body.Close()
    b, _ := io.ReadAll(resp.Body)
    var m map[string]any
    if err := json.Unmarshal(b, &m); err!=nil {{ fmt.Println(err); return }}
    k := {1}
    if v, ok := m[k]; ok {{ fmt.Println(v) }} else {{ fmt.Println("<nil>") }}
}}

```

### Ωenv_get_print

```go
package main
import ("fmt"; "os")
func main(){{
    v := os.Getenv({0})
    if v == "" {{ fmt.Println({1}) }} else {{ fmt.Println(v) }}
}}

```

### Ωenv_set_current

```go
package main
import ("os")
func main(){{
    if err := os.Setenv({0}, {1}); err != nil {{{{ panic(err) }}}}
}}

```

### Ωenv_unset

```go
package main
import ("os")
func main(){{ _ = os.Unsetenv({0}) }}

```

### Ωenv_list

```go
package main
import ("fmt"; "os")
func main(){{ for _, e := range os.Environ() {{{{ fmt.Println(e) }}}} }}

```

### Ωenv_cwd_print

```go
package main
import ("fmt"; "os")
func main(){{ d, _ := os.Getwd(); fmt.Println(d) }}

```

### Ωenv_chdir

```go
package main
import "os"
func main(){{ _ = os.Chdir({0}) }}

```

### Ωenv_path_prepend

```go
package main
import ("os")
func main(){{
    seg := {{0}}
    p := os.Getenv("PATH")
    if p != "" {{ p = seg + ":" + p }} else {{ p = seg }}
    _ = os.Setenv("PATH", p)
}}

```

### Ωenv_guard_required

```go
package main
import ("fmt"; "os")
func main(){{
    if os.Getenv({0}) == "" {{ fmt.Fprintln(os.Stderr, "Missing required env: "+{0}); os.Exit(1) }}
}}

```

### Ωlib_base_go

```go
// ---- Ωlib_base_go: stdlib helpers ----
package util

import (
    "bufio"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "errors"
    "io"
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"
    "os/exec"
)

// CLI parse: returns map opts and positional slice
func CliParse(args []string) (map[string]string, []string) {{
    opts := map[string]string{{}}
    pos := []string{{}}
    for i := 0; i < len(args); i++ {{
        a := args[i]
        if strings.HasPrefix(a, "--") {{
            if eq := strings.Index(a, "="); eq > 2 {{
                opts[a[2:eq]] = a[eq+1:]
                continue
            }}
            k := a[2:]
            if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {{
                opts[k] = args[i+1]; i++; continue
            }}
            opts[k] = "true"; continue
        }}
        if strings.HasPrefix(a, "-") && len(a)>1 {{
            k := a[1:2]
            if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {{
                opts[k] = args[i+1]; i++; continue
            }}
            opts[k] = "true"; continue
        }}
        pos = append(pos, a)
    }}
    return opts, pos
}}

// JSON log to stderr
func Log(level, msg string, fields map[string]any) {{
    rec := map[string]any{{"ts": time.Now().UTC().Format(time.RFC3339), "level": level, "msg": msg}}
    for k,v := range fields {{ rec[k]=v }}
    enc := json.NewEncoder(os.Stderr); enc.Encode(rec)
}}

// FS
func ReadText(p string) (string, error) {{
    b, err := os.ReadFile(p); if err!=nil {{ return "", err }}
    return string(b), nil
}}
func WriteText(p, s string) error {{
    if err := os.MkdirAll(filepath.Dir(p), 0o755); err!=nil {{ return err }}
    return os.WriteFile(p, []byte(s), 0o644)
}}
func MkdirP(d string) error {{ return os.MkdirAll(d, 0o755) }}
func Exists(p string) bool {{ _, err := os.Stat(p); return err==nil }}

// JSON helpers
func JSONLoad(p string, v any) error {{
    s, err := ReadText(p); if err!=nil {{ return err }}
    return json.Unmarshal([]byte(s), v)
}}
func JSONDump(p string, v any) error {{
    b, err := json.MarshalIndent(v,"","  "); if err!=nil {{ return err }}
    return WriteText(p, string(b))
}}

// SHA256
func SHA256File(p string) (string, error) {{
    f, err := os.Open(p); if err!=nil {{ return "", err }}
    defer f.Close()
    h := sha256.New()
    if _, err := io.Copy(h, bufio.NewReader(f)); err!=nil {{ return "", err }}
    return hex.EncodeToString(h.Sum(nil)), nil
}}

// HTTP GET text
func HTTPGet(url string, timeout time.Duration) (string, error) {{
    c := &http.Client{{ Timeout: timeout }}
    resp, err := c.Get(url); if err!=nil {{ return "", err }}
    defer resp.Body.Close()
    if resp.StatusCode >= 400 {{ return "", errors.New(resp.Status) }}
    b, err := io.ReadAll(resp.Body); if err!=nil {{ return "", err }}
    return string(b), nil
}}

// Retry
func Retry(fn func() error, attempts int, delay time.Duration) error {{
    var last error
    d := delay
    for i:=0;i<attempts;i++{{
        if err := fn(); err==nil {{ return nil }} else {{ last = err; time.Sleep(d); d*=2 }}
    }}
    return last
}}

// Exec capture
func RunCapture(cmd string, args ...string) (int, string, string) {{
    c := exec.Command(cmd, args...)
    out, errOut := &strings.Builder{{}}, &strings.Builder{{}}
    c.Stdout, c.Stderr = out, errOut
    err := c.Run()
    code := 0; if err!=nil {{ if ee,ok := err.(*exec.ExitError); ok {{ code = ee.ExitCode() }} else {{ code = -1 }} }}
    return code, out.String(), errOut.String()
}}

```
