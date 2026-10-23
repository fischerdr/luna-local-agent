# requirements: PyAV below 19, which broke speech input

Branch: `fix/av-19`, from `main`. Stands alone.

## The problem

On any install made since late September, every spoken turn fails; typing still works. The status line says
`Error: open() got an unexpected keyword argument 'metadata_errors'`, and the log has:

```
File ".../faster_whisper/audio.py", line 46, in decode_audio
    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

`speech.transcribe()` hands the recording to faster-whisper, which decodes it with PyAV. faster-whisper 1.2.1 (the
latest) passes `metadata_errors="ignore"` and asks only for `av>=11`. PyAV 19.0.0 (2026-09-29) removed that argument.
`installer/requirements.txt` doesn't pin PyAV, so a fresh install gets 19. Any Python, any OS.

## The fix

```diff
 faster-whisper>=1.1
+av>=11,<19              # faster-whisper 1.2 still passes av.open(metadata_errors=), gone in PyAV 19
```

PyAV 18.1 is the last release that takes the argument, with wheels up to Python 3.14. The cap can be lifted once
faster-whisper stops passing it.

An existing install is mended by `./update.sh`, or by `.venv/bin/pip install 'av<19'`.

## Test

`tests/test_requirements_av.py` checks two things:

- `requirements.txt` caps `av` below 19;
- if the installed faster-whisper's `audio.py` passes `metadata_errors`, the installed PyAV is below 19.

The second check is skipped where faster-whisper isn't installed.

```bash
python -m unittest discover tests
```
