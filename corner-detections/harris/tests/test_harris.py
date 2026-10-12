import numpy as np
from src.harris import synthetic_wedge,grayscale,structure_tensor,harris_response,directional_energy,eigenvalues

def test_shapes_and_finite():
    g=grayscale(synthetic_wedge())
    a,b,c=structure_tensor(g)
    r=harris_response(a,b,c)
    assert r.shape==(200,200)
    assert np.isfinite(r).all()

def test_directional_energy_matches_eigenvalues():
    a,b,c=2.0,0.0,1.0
    angles=np.linspace(0,2*np.pi,720,endpoint=False)
    e=directional_energy(np.array([[a]]),np.array([[b]]),np.array([[c]]),angles)[0,0]
    lmax,lmin=eigenvalues(np.array([[a]]),np.array([[b]]),np.array([[c]]))
    assert np.isclose(e.max(),lmax[0,0])
    assert np.isclose(e.min(),lmin[0,0])
