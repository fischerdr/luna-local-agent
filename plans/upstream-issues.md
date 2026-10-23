# Issues found on main (for issues or small PRs against main)

Found while installing from zero for the containers work. Each one is on
main as of `f27ef43` (2026-10-22), stands alone, and doesn't need the
containers branch. Where there's a patch, it's against main.

---

## 1. Speech input fails on every fresh install: `open() got an unexpected keyword argument 'metadata_errors'`

**What happens.** Every spoken turn fails; typing still works. The status line says
`Error: open() got an unexpected keyword argument 'metadata_errors'`, and the log has:

```
File ".../faster_whisper/audio.py", line 46, in decode_audio
    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

**Why.** `speech.transcribe()` hands the recording to faster-whisper, which decodes
it with PyAV. faster-whisper 1.2.1, the latest, passes `metadata_errors="ignore"` and
asks only for `av>=11`. **PyAV 19.0.0 (2026-09-29) removed that argument.**
`installer/requirements.txt` pins faster-whisper (`>=1.1`) but not PyAV, so any install
made since late September gets PyAV 19 and breaks. Any Python is affected.

**Reproduce.** On a fresh install, `pip show av` reports 19.x. Then press Home and say
anything.

**Fix.** PyAV 18.1 is the last release that takes the argument; it has wheels up to
Python 3.14.

```diff
--- a/installer/requirements.txt
+++ b/installer/requirements.txt
@@
 faster-whisper>=1.1
+av>=11,<19              # faster-whisper 1.2 still passes av.open(metadata_errors=), gone in PyAV 19
```

An existing install mends with `./update.sh`, or `.venv/bin/pip install 'av<19'`. Lift
the cap once faster-whisper stops passing the argument.

**Test.** Check that requirements.txt caps `av` below 19, and that if faster-whisper's
`audio.py` passes `metadata_errors`, the installed PyAV is below 19.

---

## 2. `/settings` crashes: `cannot access local variable 'config' where it is not associated with a value`

**What happens.** Typing `/settings` raises an unhandled exception in the event loop:

```
File "main.py", line 584, in handle_input
    if not config.MONITOR_ENABLED:
UnboundLocalError: cannot access local variable 'config' where it is not associated with a value
```

**Why.** `handle_input()` (`main.py:349`) has `import config` inside its `/camera`
branch (`main.py:1504`, from 2026-10-09). An import inside a function makes the name
local to the **whole** function, so every earlier use of `config` in `handle_input`
fails, at least until `/camera` has run once. `/settings` (`main.py:584`, added in
`30fe513` on 2026-10-21) is the first command to hit it. `config` is already
imported at the top of `main.py` (line 11).

**Reproduce.** Start Luna and type `/settings`.

**Fix.**

```diff
--- a/main.py
+++ b/main.py
@@ def handle_input(text):
     if text == "/camera" or text.startswith("/camera "):
         import camera
-        import config
         import tools
```

**Test.** Parse `main.py` with `ast`, and fail if any function imports or assigns
`config` again. That catches the next one too.

---

## 3. The wake word never loads on Linux (openwakeword falls back to 0.4.0)

**What happens.** With the `wakeword` extra installed, `/wake` reports an error like
`AudioFeatures.__init__() got an unexpected keyword argument 'wakeword_models'`, and
open mode starts on any speech instead of her name. It works on macOS and Windows.

**Why.**
- `installer/install.py:46` installs `openwakeword` with no version.
- openwakeword 0.5.0 and newer require `tflite-runtime >=2.8; platform_system == "Linux"`.
- `tflite-runtime` (Google's old TensorFlow Lite package, now `ai-edge-litert`) only has
  wheels for Python **3.6-3.9**. So on Linux with any Python Luna supports (3.10+), pip
  can't install 0.5 or 0.6, and quietly settles on **0.4.0** (April 2023), the newest
  release without that dependency.
- `wakeword.py` is written for the 0.5+ API:
  - line 112, `openwakeword.utils.download_models()`, doesn't exist in 0.4.0 (the
    `except Exception: pass` hides that);
  - line 116, `Model(wakeword_models=..., inference_framework="onnx")`: 0.4.0's
    signature is `Model(wakeword_model_paths=...)`, with no `inference_framework`.

Luna already asks for the onnx framework, which needs no tflite at all. tflite is only
a packaging dependency here.

**Reproduce.** On Linux, `./installer/install.sh --extras wakeword`, then
`pip show openwakeword` reports 0.4.0, and `/wake` shows the error.

**Fix (proposed, still to be tested).**
- `installer/install.py`, the `wakeword` extra: install `openwakeword==0.6.0` with
  `--no-deps`, then what it needs at run time: `onnxruntime scipy scikit-learn tqdm
  requests`. `python_env()` installs extras in one pip call, so this one needs its own
  call.
- `wakeword.py`: when the installed openwakeword is older than 0.5, say so plainly
  ("openwakeword 0.4 is installed; 0.5+ is needed: ...") instead of the raw
  `TypeError`, and don't silently swallow a failed `download_models()`.
- `docs/controls.md` (Wake Word): a line on this until it's fixed upstream.
- Upstream, at `dscripka/openWakeWord`: make `tflite-runtime` an optional extra, or move
  it to `ai-edge-litert`.

---

## 4. The host voice won't install on Python 3.13

**What happens.** With Python 3.13 as the installer's Python, the voice step ends
with "voice server packages failed".

**Why.** `installer/install.sh` accepts Python 3.10-3.13 (`pick()` tries 3.12, 3.11,
3.13, 3.10 in that order, and its fallback check is `<= (3,13)`). The host voice's venv
is made from that same Python, and kokoro-reader needs `kokoro` and `misaki` 0.9.4,
both `requires-python >=3.10,<3.13`. A machine whose newest Python is 3.13 (and that
has no 3.12 or 3.11) gets a Luna that installs and a voice that doesn't.

**Fix (either).**
- Make the voice's venv from Python 3.10-3.12 when one is there, even if Luna's is 3.13.
- Or say it up front: "Kokoro needs Python 3.10-3.12; with only 3.13 here the voice
  can't be installed" (and point at `--voice url:` for a voice server elsewhere).

---

## 5. The Live2D runtime isn't ignored where the installer saves it

**What happens.** After `./installer/install.sh --extras live2d`, `git status` shows
`portrait/viewer/vendor/live2dcubismcore.min.js` as untracked, so it's easy to commit
by accident. Live2D's own file, under their licence.

**Why.** The installer saves it to `portrait/viewer/vendor/` (`installer/install.py`,
`extras_files()`), but `.gitignore` line 110 names `portrait/vendor/live2dcubismcore.min.js`.

**Fix.**

```diff
--- a/.gitignore
+++ b/.gitignore
@@
-portrait/vendor/live2dcubismcore.min.js
+portrait/viewer/vendor/live2dcubismcore.min.js
```

(Unless something else does write to `portrait/vendor/`; I couldn't find anything.)

---

## 6. Two health-check tests fail when the machine's swap is more than half full

**What happens.** `tests/test_system_tools.py`'s `Health.test_all_quiet` (line 100)
and `Health.test_away_from_linux` (line 111) fail on a busy machine:
`AssertionError: ... Needs a look: more than half the swap is in use.`

**Why.** Both tests mock the RAM, disks, load, CPU temperature and GPUs, but not
swap. `tools/system.py:171`, `_swap()`, reads the real `/proc/meminfo`, and at line
283 health() warns when more than half of it is in use. So a test that expects
"Nothing looks wrong." depends on the test machine's swap.

**Fix.** In both tests, add
`mock.patch.object(system, "_swap", return_value=None)` beside the other patches.

---

## 7. A broken docs link: `devices.md` → configuration's "Who she's talking to"

`docs/devices.md:71` links to `configuration.md#whos-she-talking-to`, but the heading
`## Who she's talking to` gets the anchor `#who-shes-talking-to`.

```diff
--- a/docs/devices.md
+++ b/docs/devices.md
@@
-(see [configuration](configuration.md#whos-she-talking-to)).
+(see [configuration](configuration.md#who-shes-talking-to)).
```
