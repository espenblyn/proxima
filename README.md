# proxima-ml

Distance and similarity metrics for Rust.

Generic over `f32` and `f64`, with batch operations out of the box.

## Quick start
```toml
[dependencies]
proxima-ml = "0.2"
```
```rust
use proxima_ml::{Distance, Similarity, Euclidean, Cosine};

let a = &[1.0, 2.0, 3.0];
let b = &[4.0, 5.0, 6.0];

let dist = Euclidean::distance(a, b);
let sim = Cosine::similarity(a, b);
```

## ndarray support

Enable the `ndarray` feature to pass directly to any metric:
```toml
[dependencies]
proxima-ml = { version = "0.2", features = ["ndarray"] }
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
