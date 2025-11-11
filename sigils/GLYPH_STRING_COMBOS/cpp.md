# Codex — cpp

### α

```cpp
for (int {0}=0; {0}<{1}; ++{0}){{
{2}
}}
```

### γ

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){{
{0}
return 0;
}}
```

### ε

```cpp
cout << {0} << std::endl;
```

### □

```cpp
{0}
```

### Ωcpp_for_range

```cpp
for (auto& {0} : {1}) {{ {2} }}
```

### Ωcpp_func

```cpp
{0} {1}({2}) {{ {3} }}
```

### Ωcpp_vector

```cpp
#include <vector>
std::vector<{0}> {1};
```

### Ωcli_echo

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){{
    for(int i=1;i<argc;i++){{ if(i>1) cout<<" "; cout<<argv[i]; }}
    cout<<endl;
    return 0;
}}

```

### Ωsum_numbers

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){{
    long long s=0; for(int i=1;i<argc;i++) s += atoll(argv[i]);
    cout<<s<<endl; return 0;
}}

```

### Ωfile_cat

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){{
    if(argc<2) return 1;
    ifstream in(argv[1]); cout<<in.rdbuf(); return 0;
}}

```

### Ωcpp_sort_numbers_stdin

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){{
    vector<long long> a; long long x;
    while (cin>>x) a.push_back(x);
    sort(a.begin(), a.end());
    for (auto &v: a) cout<<v<<"\n";
    return 0;
}}

```
