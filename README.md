# A Python Library For Empirical Calibration

## tl:dr

Dealing with biased data samples is a common task across many statistical
fields. In survey sampling, bias often occurs due to the unrepresentative
samples. In causal studies with observational data, the treated vs untreated
group assignment is often correlated with covariates, i.e., not random.
Empirical calibration is a generic weighting method that presents a unified
view on correcting or reducing the data biases for the tasks mentioned above.
We provide a Python library EC to compute the empirical calibration weights.
The problem is formulated as a convex optimization and solved efficiently in
the dual form. Compared to existing software, EC is both more efficient and
robust. EC also accommodates different optimization objectives, supports weight
clipping, and allows inexact calibration which improves the usability.
We demonstrate its usage across various experiments with both simulated and
real-world data.

## Paper

Please see [pdf](https://arxiv.org/abs/1906.11920)