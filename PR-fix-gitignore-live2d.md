# .gitignore: the Live2D runtime where the installer saves it

Branch: `fix/gitignore-live2d`, from `main`. Stands alone.

## The problem

After `./installer/install.sh --extras live2d`, `git status` shows `portrait/viewer/vendor/live2dcubismcore.min.js` as
untracked. That's Live2D's own file, under their licence, and it's easy to commit by accident.

The installer saves it to `portrait/viewer/vendor/` (`installer/install.py`, `extras_files()`), but `.gitignore`
names `portrait/vendor/live2dcubismcore.min.js`.

## The fix

```diff
-portrait/vendor/live2dcubismcore.min.js
+portrait/viewer/vendor/live2dcubismcore.min.js
```

I couldn't find anything that writes to `portrait/vendor/`. If something does, keep both lines.

## Test

```bash
git check-ignore -v portrait/viewer/vendor/live2dcubismcore.min.js
```
