# pure-logic test for the residualisation that carries the specificity claim.

import torch


def _gram_schmidt(vectors, tolerance=1e-6):
    basis = []
    for vector in vectors:
        residual = vector.clone()
        for existing in basis:
            residual = residual - (residual @ existing) * existing
        if float(residual.norm()) > tolerance * max(float(vector.norm()), 1e-12):
            basis.append(residual / residual.norm())
    return basis


def test_gram_schmidt_returns_an_orthonormal_basis():
    generator = torch.Generator().manual_seed(0)
    vectors = [torch.randn(30, generator=generator) for _ in range(4)]
    basis = _gram_schmidt(vectors)
    assert len(basis) == 4
    for i, a in enumerate(basis):
        assert abs(float(a.norm()) - 1.0) < 1e-5
        for b in basis[i + 1 :]:
            assert abs(float(a @ b)) < 1e-5


def test_gram_schmidt_drops_a_duplicate_direction():
    vector = torch.randn(20, generator=torch.Generator().manual_seed(1))
    basis = _gram_schmidt([vector, vector * 2.5])
    assert len(basis) == 1


def test_residualising_removes_everything_the_others_explain():
    generator = torch.Generator().manual_seed(2)
    others = [torch.randn(40, generator=generator) for _ in range(3)]
    basis = _gram_schmidt(others)
    # an own-vector lying entirely in the others' span must residualise to nothing
    own = 2.0 * basis[0] - 1.5 * basis[1]
    residual = own.clone()
    for unit in basis:
        residual = residual - (residual @ unit) * unit
    assert float(residual.norm()) / float(own.norm()) < 1e-5


def test_residualising_keeps_a_genuinely_independent_component():
    generator = torch.Generator().manual_seed(3)
    others = [torch.randn(40, generator=generator) for _ in range(3)]
    basis = _gram_schmidt(others)
    independent = torch.randn(40, generator=generator)
    for unit in basis:
        independent = independent - (independent @ unit) * unit
    own = basis[0] + independent
    residual = own.clone()
    for unit in basis:
        residual = residual - (residual @ unit) * unit
    assert 0.3 < float(residual.norm()) / float(own.norm()) < 1.0
