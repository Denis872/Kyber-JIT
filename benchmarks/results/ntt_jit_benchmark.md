# NTT JIT Optimization Results
--------------------------------------------------------------------------------
   Params    |  keygen  |  keygen/s  |  encap  |  encap/s  |  decap  |  decap/s
--------------------------------------------------------------------------------
 Kyber512    |   1.71ms |     584.62 |  1.54ms |    651.04 |  1.88ms |  531.82 |
 Kyber768    |   1.38ms |     723.72 |  1.97ms |    508.47 |  2.81ms |  356.16 |
 Kyber1024   |   2.18ms |     458.08 |  2.86ms |    349.32 |  3.94ms |  253.93 |

Compared to original implementation:
- Kyber512 decap: 3.44ms → 1.88ms (1.83x faster)
- Kyber768 decap: 5.13ms → 2.81ms (1.83x faster)
- Kyber1024 decap: 7.76ms → 3.94ms (1.97x faster)

