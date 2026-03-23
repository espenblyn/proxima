# proxima-ml

Distance and similarity metrics for Rust.

Generic over `f32` and `f64`, with batch operations out of the box.

## Quick start
```toml
[dependencies]
proxima = "0.1"
```
```rust
use proxima::{Distance, Similarity, Euclidean, Cosine};

let a = &[1.0, 2.0, 3.0];
let b = &[4.0, 5.0, 6.0];

let dist = Euclidean::distance(a, b);
let sim = Cosine::similarity(a, b);
```

## Metrics

| Metric | Type | 
|--------|------|
| `Euclidean` | Distance |
| `SqEuclidean` | Distance |
| `Manhattan` | Distance |
| `Cosine` | Similarity + Distance |
| `Dot` | Similarity |
| `Hamming` | Distance |

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).
