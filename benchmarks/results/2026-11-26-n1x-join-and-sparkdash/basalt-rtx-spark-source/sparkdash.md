### Decode: Prose, thinking off, temp 0, 400 tokens, top_k 20

| Concurrency | TTFT | Aggregate tok/s | Per-stream tok/s |
|---:|---:|---:|---:|
| 1 | 73 ms | 93.3 | 93.3 |
| 2 | 862 ms | 109.8 | 56.9 |
| 3 | 491 ms | 116.9 | 42.9 |
| 4 | 856 ms | 112.5 | 30.0 |
| 5 | 2.78 s | 130.4 | 48.4 |

### Decode: Prose, thinking on, temp 0.7, 400 tokens, top_k 20

| Concurrency | TTFT | Aggregate tok/s | Per-stream tok/s |
|---:|---:|---:|---:|
| 1 | 59 ms | 87.3 | 87.3 |
| 2 | 543 ms | 107.8 | 56.2 |
| 3 | 413 ms | 134.1 | 47.5 |
| 4 | 507 ms | 134.8 | 36.1 |
| 5 | 2.59 s | 124.8 | 45.2 |

### Decode: Code, thinking off, temp 0, 400 tokens, top_k 20

| Concurrency | TTFT | Aggregate tok/s | Per-stream tok/s |
|---:|---:|---:|---:|
| 1 | 345 ms | 148.0 | 148.0 |
| 2 | 288 ms | 134.9 | 94.0 |
| 3 | 352 ms | 146.5 | 64.7 |
| 4 | 832 ms | 148.5 | 49.3 |
| 5 | 968 ms | 142.5 | 41.2 |

### Prefill (cold)

| Context | Actual tokens | TTFT | Prefill tok/s |
|---:|---:|---:|---:|
| 1k | 1064 | 581 ms | 1967.8 |
| 2k | 2087 | 1.01 s | 2153.5 |
| 4k | 4134 | 1.87 s | 2258.9 |
| 8k | 8231 | 3.55 s | 2349.5 |
| 16k | 16420 | 6.87 s | 2406.1 |
| 32k | 32806 | 13.68 s | 2408.3 |
| 64k | 65572 | 28.06 s | 2342.8 |
| 128k | 131111 | 60.30 s | 2178.2 |

