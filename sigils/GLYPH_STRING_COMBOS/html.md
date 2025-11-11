# Codex — html

### btn

```html
<button>{0}</button>
```

### div

```html
<div>{0}</div>
```

### h

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{0}</title>
</head>
<body>
{1}
</body>
</html>
```

### p

```html
<p>{0}</p>
```

### Ωhtml_form

```html
<form action="{0}" method="{1}">
  {2}
</form>
```

### Ωhtml_input

```html
<input type="{0}" name="{1}" value="{2}">
```

### Ωhtml_link_styles

```html
<link rel="stylesheet" href="{0}">
```

### Ωhtml_meta_viewport

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

### Ωhtml_script_module

```html
<script type="module" src="{0}"></script>
```

### Ωhtml_counter_app

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Counter</title>
  <style>
    body {{{{ font-family: system-ui, sans-serif; padding: 2rem; }}}}
    button {{{{ font-size: 1.25rem; padding: .5rem 1rem; }}}}
    #v {{{{ font-weight: bold; margin-left: .5rem; }}}}
  </style>
</head>
<body>
  <button id="b">Increment</button><span id="v">0</span>
  <script>
    let n=0; document.getElementById('b').onclick=()=>{{{{ n++; document.getElementById('v').textContent=n; }}}};
  </script>
</body>
</html>

```
