# Vision Foundations — Harris Corner Detection

A reproducible, mathematical walkthrough of the Harris detector and the local structure tensor, developed from an original synthetic-wedge directional-derivative experiment.

## Contents
- `notebooks/01_harris_structure_tensor.ipynb` — derivation, synthetic image, gradient tensor, eigenvalues, polar plots, Harris candidates, exercises
- `src/harris.py` — reusable numerical functions
- `tests/test_harris.py` — basic numerical checks
- `original_harris_corner.py` — original supplied experiment, preserved for provenance

## Run
```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_harris_structure_tensor.ipynb
python -m pytest -q
```

## Main improvements over original experiment
- Uses grayscale gradients instead of per-channel directional maxima.
- Uses the 2×2 second-moment matrix and actual Harris response.
- Computes all angular energies from three smoothed tensor components, avoiding a large 4D buffer and repeated Gaussian blurs.
- Clarifies `(x, y)` versus NumPy `[row, column]` indexing.
- Adds eigenvalue interpretation, corner visualization, exercises, and tests.

## Attribution
Original synthetic-wedge experiment and polar-plot concept supplied by repository author; notebook refactored and expanded collaboratively.

## Next steps
Add comparisons with OpenCV `cornerHarris`, noise robustness, parameter sensitivity, and feature matching.
