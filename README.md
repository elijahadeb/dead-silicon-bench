# dead-silicon-bench

Dot product over 100M float32, four ways: float64 reference, scalar f32,
AVX2 single thread, AVX2 threaded. Measures achieved memory bandwidth
against the theoretical absolute limit.

## Run

```
make
```

Builds both binaries, generates the data if missing, and runs the benchmark.

Data is 100,000,000 float32 per file, Mersenne Twister seeded 0 and 2,
uniform over [0,1). It's not in the repo, `make` generates it.

The float64 reference should print `2.500202884446200e+07`. If yours differs,
the data is wrong.

## Results

i7-7700HQ, 2x8 GiB DDR4-2400 dual channel. Theoretical peak 38.4 GB/s.
Median of 9 runs.

| path           | GB/s | % of peak | rel err  |
|----------------|------|-----------|----------|
| scalar f32     | 6.5  | 17%       | 5.24e-12 |
| avx2, 1 thread | 16.5 | 43%       | 2.06e-09 |
| avx2, threaded | 22.9 | 60%       | 2.06e-09 |

Single runs do vary a lot, threaded avx2 ranged 19.4 to 27.6 GB/s on the
same binary. Take a median.

If you run it, open an issue with your CPU, RAM config and numbers. I've only
measured this on one machine.
