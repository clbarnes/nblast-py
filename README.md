# nblast-py

Pure(ish)-python implementation of NBLAST (as published
[here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4961245/)
and implemented
[here](https://github.com/natverse/nat.nblast/)
by the Jefferis lab), for experimentation purposes.
Written for python 3.8.

Not intended for use in production; it's slow.
For a high-performance NBLAST library with python bindings, see [nblast-rs](https://github.com/clbarnes/nblast-rs), WIP.

## Test data

Generated (in R) with

```R
library(nat.nblast)
library(nat)
kcscores <- nblast_allbyall(kcs20)
write.csv(kcscores, file='data/kcscores.csv')
```

This depends on point clouds with tangents pre-calculated by the `nat` library, using a `k=5`.
These point clouds/ tangent vectors/ alpha values are included in `data/dotprops`.

The score matrix `data/smat_fcwb.csv` is also pre-calculated, extracted from the nat.nblast [github repository](https://github.com/natverse/nat.nblast/blob/b9be0e51590c39c151309cabd00cdf250a5d6ff1/data/smat.fcwb.rda).
