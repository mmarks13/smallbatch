# complaint-review-priority evidence

Evaluation decisions: **120**

> Decision agreement measures fidelity to the supplied decisions, not correctness.

## Candidates

| candidate | status | runtime | p50 ms | peak RSS | owned bytes |
|---|---:|---:|---:|---:|---:|
| tfidf | completed | tfidf | 1.1858 | 804626432 | 27212093 |

### tfidf metrics

```json
{
  "n": 120,
  "valid_n": 120,
  "invalid_rate": 0.0,
  "exact": 0.5417,
  "exact_ci": [
    0.4526,
    0.6281
  ],
  "within_one": 0.8667,
  "within_one_ci": [
    0.7944,
    0.9162
  ],
  "mae": 0.6,
  "absolute_error_histogram": {
    "0": 65,
    "1": 39,
    "2": 15,
    "3": 1
  },
  "p90_absolute_error": 2,
  "max_absolute_error": 3,
  "mean_signed_error": -0.4333,
  "pearson_r": 0.373,
  "spearman_rho": 0.367
}
```
| bge-small | completed | setfit | 42.321 | 1794359296 | 134520754 |

### bge-small metrics

```json
{
  "n": 120,
  "valid_n": 120,
  "invalid_rate": 0.0,
  "exact": 0.5667,
  "exact_ci": [
    0.4773,
    0.6519
  ],
  "within_one": 0.8833,
  "within_one_ci": [
    0.8137,
    0.9292
  ],
  "mae": 0.55,
  "absolute_error_histogram": {
    "0": 68,
    "1": 38,
    "2": 14
  },
  "p90_absolute_error": 2,
  "max_absolute_error": 2,
  "mean_signed_error": -0.1,
  "pearson_r": 0.4526,
  "spearman_rho": 0.4714
}
```
| qwen3-06b | completed | lora | 781.3839 | 5330980864 | 56310842 |

### qwen3-06b metrics

```json
{
  "n": 120,
  "valid_n": 120,
  "invalid_rate": 0.0,
  "exact": 0.6333,
  "exact_ci": [
    0.5442,
    0.7142
  ],
  "within_one": 0.9167,
  "within_one_ci": [
    0.8534,
    0.9541
  ],
  "mae": 0.45,
  "absolute_error_histogram": {
    "0": 76,
    "1": 34,
    "2": 10
  },
  "p90_absolute_error": 1,
  "max_absolute_error": 2,
  "mean_signed_error": 0.2667,
  "pearson_r": 0.5778,
  "spearman_rho": 0.587
}
```
| qwen3-17b | completed | lora | 1874.1125 | 11148455936 | 50806178 |

### qwen3-17b metrics

```json
{
  "n": 120,
  "valid_n": 120,
  "invalid_rate": 0.0,
  "exact": 0.6417,
  "exact_ci": [
    0.5527,
    0.7218
  ],
  "within_one": 0.9333,
  "within_one_ci": [
    0.8739,
    0.9658
  ],
  "mae": 0.425,
  "absolute_error_histogram": {
    "0": 77,
    "1": 35,
    "2": 8
  },
  "p90_absolute_error": 1,
  "max_absolute_error": 2,
  "mean_signed_error": 0.0917,
  "pearson_r": 0.601,
  "spearman_rho": 0.6122
}
```
| qwen3-4b | completed | lora | 4334.7738 | 20703903744 | 82016444 |

### qwen3-4b metrics

```json
{
  "n": 120,
  "valid_n": 120,
  "invalid_rate": 0.0,
  "exact": 0.6917,
  "exact_ci": [
    0.6042,
    0.7673
  ],
  "within_one": 0.9333,
  "within_one_ci": [
    0.8739,
    0.9658
  ],
  "mae": 0.375,
  "absolute_error_histogram": {
    "0": 83,
    "1": 29,
    "2": 8
  },
  "p90_absolute_error": 1,
  "max_absolute_error": 2,
  "mean_signed_error": 0.0583,
  "pearson_r": 0.6472,
  "spearman_rho": 0.6397
}
```

## Interpretation

These candidates share one evaluation split. Selecting after comparison makes the selected result optimistic; v0.2 does not provide an independent confirmation set.
