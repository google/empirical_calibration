# A Python Library For Empirical Calibration

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

Wang, Xiaojing, Miao, Jingang, and Sun, Yunting. (2019).
A Python Library For Empirical Calibration.
[*arXiv preprint arXiv:1906.11920*](https://arxiv.org/abs/1906.11920).

## Installation

The easiest way is propably using pip:

```
pip install -q git+https://github.com/google/empirical_calibration
```

If you are using a machine without admin rights, you can do:

```
pip install -q git+https://github.com/google/empirical_calibration --user
```

If you are using [Google Colab](https://colab.research.google.com/), just add
"!" to the beginning:

```
!pip install -q git+https://github.com/google/empirical_calibration
```

Package works for python 3.6 or later.

## Usage
Package can be imported as

```python
import empirical_calibration as ec
```

The best way to learn how to use the package is probably by following one of the
notebooks, and the recommended way of opening them is Google Colab.

* Survey calibration
   - [Toy simulation](./notebooks/survey_calibration_simulated.ipynb)
   - [Kang Schafer simulation](./notebooks/kang_schafer_population_mean.ipynb)
   - [Example in R package CVXR](./notebooks/survey_calibration_cvxr.ipynb)
* Causal inference
   - [Lalonde](./notebooks/causal_inference_lalonde.ipynb)
   - [Kang-Schafer simulation](./notebooks/causal_inference_kang_schafer.ipynb)

## People
Package is created and maintained by Xiaojing Wang, Jingang Miao, and Yunting
Sun. Special thanks to Emil Martayan for helping add `baseline_weights` support.


