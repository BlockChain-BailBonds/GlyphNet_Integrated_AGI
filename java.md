# Codex — java

### α

```java
for (int {0}=0; {0}<{1}; {0}++) {{ {2} }}
```

### γ

```java
public class Main {{ public static void main(String[] args) {{ {0} }} }}
```

### ε

```java
System.out.println({0});
```

### □

```java
{0}
```

### Ωjava_class_method

```java
class {0} {{ {1} {2}({3}) {{ {4} }} }}
```

### Ωjava_list

```java
List<{0}> {1} = new ArrayList<>();
```

### Ωjava_try_catch

```java
try {{ {0} }} catch ({1} e) {{ {2} }}
```

### Ωcli_echo

```java
public class Main {{
    public static void main(String[] args){{
        System.out.println(String.join(" ", args));
    }}
}}

```

### Ωhttp_server_8000

```java
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
public class Main {{
    public static void main(String[] args) throws Exception {{
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/", new HttpHandler(){{
            public void handle(HttpExchange ex) throws IOException {{
                byte[] bytes = "ok".getBytes();
                ex.sendResponseHeaders(200, bytes.length);
                try(OutputStream os = ex.getResponseBody()){{ os.write(bytes); }}
            }}
        }});
        server.start();
    }}
}}

```

### Ωjava_readfile_print

```java
import java.nio.file.*; import java.io.*; 
public class Main {{ 
    public static void main(String[] args) throws Exception {{ 
        System.out.print(Files.readString(Path.of(args[0]))); 
    }} 
}}

```

### Ωandr_activity_java_min

```java
package {0};

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {{{{
    @Override
    protected void onCreate(Bundle savedInstanceState) {{{{
        super.onCreate(savedInstanceState);
        TextView tv = new TextView(this);
        tv.setText("Hello, Android!");
        setContentView(tv);
    }}}}
}}}}

```

### Ωlib_base_java

```java
// ---- Ωlib_base_java: std-only subset (Java 11+) ----
import java.io.*; import java.nio.file.*; import java.time.*; import java.security.*; import java.net.*;
import java.net.http.*; import java.util.*; 

public class Lib {{
    public static Map<String,String> cliParse(String[] argv){{
        Map<String,String> opts = new HashMap<>(); List<String> pos = new ArrayList<>();
        for(int i=0;i<argv.length;i++){{
            String a = argv[i];
            if(a.startsWith("--")){{
                int eq = a.indexOf("=");
                if(eq>2){{ opts.put(a.substring(2,eq), a.substring(eq+1)); continue; }}
                String k = a.substring(2);
                if(i+1<argv.length && !argv[i+1].startsWith("-")){{ opts.put(k, argv[++i]); }} else {{ opts.put(k,"true"); }}
                continue;
            }}
            if(a.startsWith("-") && a.length()>1){{
                String k = a.substring(1,2);
                if(i+1<argv.length && !argv[i+1].startsWith("-")){{ opts.put(k, argv[++i]); }} else {{ opts.put(k,"true"); }}
                continue;
            }}
            pos.add(a);
        }}
        // NOTE: positions not returned here; extend as needed
        return opts;
    }}

    public static void log(String level, String msg){{
        System.err.println("{{"ts":""+Instant.now().toString()+"","level":""+level+"","msg":""+msg.replace(""","\"")+""}}");
    }}

    public static String readText(String p) throws IOException {{ return Files.readString(Path.of(p)); }}
    public static void writeText(String p, String s) throws IOException {{ Files.createDirectories(Path.of(p).getParent()); Files.writeString(Path.of(p), s); }}

    public static String sha256File(String p) throws Exception {{
        MessageDigest d = MessageDigest.getInstance("SHA-256");
        try(InputStream in = Files.newInputStream(Path.of(p))){{
            byte[] buf = new byte[65536]; int r;
            while((r=in.read(buf))!=-1){{ d.update(buf,0,r); }}
        }}
        byte[] dig = d.digest(); StringBuilder sb = new StringBuilder();
        for(byte b: dig){{ sb.append(String.format("%02x", b)); }}
        return sb.toString();
    }}

    public static String httpGet(String url) throws Exception {{
        HttpClient c = HttpClient.newBuilder().connectTimeout(java.time.Duration.ofSeconds(15)).build();
        HttpRequest req = HttpRequest.newBuilder(URI.create(url)).GET().build();
        HttpResponse<String> resp = c.send(req, HttpResponse.BodyHandlers.ofString());
        if(resp.statusCode()>=400) throw new IOException("HTTP "+resp.statusCode());
        return resp.body();
    }}
}}

```
