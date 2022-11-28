# interflop-backend-verrou

## Arguments
```bash
VFC_BACKENDS="libinterflop_verrou.so --help"
Usage: libinterflop_verrou.so [OPTION...] 

      --rounding-mode=ROUNDING MODE
                             select rounding mode among {nearest, upward,
                             downward, toward_zero, random, random_det,
                             random_comdet, average, average_det,
                             average_comdet, farthest,float,native,ftz}
      --seed=SEED            fix the random generator seed
      --static-backend       load the operators directly instead of switching
                             which makes computations faster
  -?, --help                 Give this help list
      --usage                Give a short usage message
```