# Summarise inference usage

Implement `summarise_usage(records)` in `usage.py`. Each input record contains
`model` (a string), `prompt_tokens` (a non-negative integer), and
`completion_tokens` (a non-negative integer). Assume these fields and types are
present; validate that neither token count is negative.

Return a dictionary keyed by model. Each value must contain `prompt_tokens`,
`completion_tokens`, and `total_tokens`, summed across that model's records.
Return `{}` for empty input. Raise `ValueError` for a negative token count.
Do not modify the input records. Use only Python's standard library.

For example, two records for `model-a` with token counts `(12, 3)` and `(8, 7)`
produce:

```python
{"model-a": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
```

Run `python3 -m unittest -v` to verify the implementation. Keep the existing tests
unchanged. These are synthetic usage records, not a billing calculation.
