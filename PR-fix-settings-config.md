# /settings: no crash from a second `import config` in handle_input

Branch: `fix/settings-config`, from `main`. Stands alone.

## The problem

Typing `/settings` raises an unhandled exception:

```
File "main.py", line 584, in handle_input
    if not config.MONITOR_ENABLED:
UnboundLocalError: cannot access local variable 'config' where it is not associated with a value
```

`handle_input()` has `import config` inside its `/camera` branch (`main.py:1504`). An import inside a function makes the
name local to the whole function, so any use of `config` earlier in `handle_input` fails, at least until `/camera` has
run once. `/settings` (line 584) is the first command to hit it. `config` is already imported at the top of `main.py`
(line 11).

## The fix

```diff
     if text == "/camera" or text.startswith("/camera "):
         import camera
-        import config
         import tools
```

## Test

`tests/test_main_config_scope.py` parses `main.py` (without importing it), and fails if any function imports or assigns
`config` again. That catches the next one too.

```bash
python -m unittest discover tests
```
