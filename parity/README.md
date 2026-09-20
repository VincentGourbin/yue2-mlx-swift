# parity/

Tier-1 fixtures (weight-free, always green), generated from the **upstream** `yue2`
Python classes only — never a reimplementation. Regenerate with:

```
.venv-ref/bin/python Scripts/reference/tiny_fixtures.py
```

Do **not** edit any file here by hand (safetensors metadata carries `seed` and
`upstream_sha`; a hand edit breaks that provenance and the determinism check in
`Scripts/check-fixtures.sh`). `sampling.safetensors` stores each config's `distribution()`
output sparsely (`{cfg}_finite_indices` / `{cfg}_finite_values`): every other position of
the `[184704]` vocab is exactly `-inf`.
