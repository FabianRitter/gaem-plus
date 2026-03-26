"""
Basic unit tests for GAEM+ core components.
Run with: python -m pytest gaem/tests/test_core.py -v
"""

import torch
import sys
import os

# Add parent to path so gaem package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def test_procrustes_identity():
    """Procrustes of identical features should give identity."""
    from gaem.alignment.procrustes import procrustes_orthogonal
    X = torch.randn(100, 64)
    O = procrustes_orthogonal(X, X)
    assert torch.allclose(O, torch.eye(64), atol=1e-5), "Procrustes of same features should be identity"


def test_procrustes_is_orthogonal():
    """Result of Procrustes should be an orthogonal matrix."""
    from gaem.alignment.procrustes import procrustes_orthogonal
    X = torch.randn(100, 32)
    Y = torch.randn(100, 32)
    O = procrustes_orthogonal(X, Y)
    product = O @ O.T
    assert torch.allclose(product, torch.eye(32), atol=1e-5), "O @ O^T should be identity"


def test_procrustes_reduces_error():
    """Aligning with Procrustes should reduce reconstruction error vs no alignment."""
    from gaem.alignment.procrustes import procrustes_orthogonal, compute_alignment_error
    X = torch.randn(100, 32)
    R = torch.linalg.qr(torch.randn(32, 32))[0]  # random rotation
    Y = X @ R + 0.1 * torch.randn(100, 32)  # rotated + noise

    error_before = compute_alignment_error(X, Y, torch.eye(32))
    O = procrustes_orthogonal(X, Y)
    error_after = compute_alignment_error(X, Y, O)
    assert error_after < error_before, f"Alignment should reduce error: {error_before:.4f} -> {error_after:.4f}"


def test_lors_decomposition():
    """LoRS should decompose into low-rank + sparse with bounded reconstruction error."""
    from gaem.decomposition.lors import lors_decompose, compute_lors_stats
    tv = torch.randn(64, 64)
    lr, sp = lors_decompose(tv, rank_ratio=0.1, sparsity=0.9)
    assert lr.shape == tv.shape
    assert sp.shape == tv.shape

    stats = compute_lors_stats(tv, rank_ratio=0.1, sparsity=0.9)
    assert stats["relative_error"] < 1.0, "Reconstruction error should be bounded"


def test_lors_state_dict():
    """LoRS decomposition should work on state dicts."""
    from gaem.decomposition.lors import lors_decompose_state_dict
    sd = {"weight1": torch.randn(128, 64), "weight2": torch.randn(64, 64), "bias": torch.randn(64)}
    lr, sp = lors_decompose_state_dict(sd, rank_ratio=0.2, sparsity=0.8)
    assert set(lr.keys()) == set(sd.keys())
    assert set(sp.keys()) == set(sd.keys())


def test_task_arithmetic():
    """Basic task arithmetic merge should work."""
    from gaem.merging.task_arithmetic import compute_task_vector, task_arithmetic_merge
    base = {"w": torch.zeros(4, 4)}
    ft1 = {"w": torch.ones(4, 4)}
    ft2 = {"w": -torch.ones(4, 4)}

    tv1 = compute_task_vector(ft1, base)
    tv2 = compute_task_vector(ft2, base)
    merged = task_arithmetic_merge(base, [tv1, tv2], [0.5, 0.5])
    assert torch.allclose(merged["w"], torch.zeros(4, 4)), "Equal opposing TVs should cancel"


def test_ties_merge():
    """TIES merge should produce non-zero output for non-zero inputs."""
    from gaem.merging.task_arithmetic import ties_merge
    tv1 = {"w": torch.randn(32, 32)}
    tv2 = {"w": torch.randn(32, 32)}
    merged = ties_merge([tv1, tv2], [0.5, 0.5], k=0.5)
    assert "w" in merged
    assert merged["w"].abs().sum() > 0


def test_gaem_plus_pipeline():
    """Full GAEM+ pipeline should run without errors."""
    from gaem.merging.gaem_plus import gaem_plus_merge
    d = 32
    base = {"w1": torch.randn(d, d), "w2": torch.randn(d, d), "b1": torch.randn(d)}
    ft1 = {"w1": base["w1"] + 0.1 * torch.randn(d, d),
            "w2": base["w2"] + 0.1 * torch.randn(d, d),
            "b1": base["b1"] + 0.1 * torch.randn(d)}
    ft2 = {"w1": base["w1"] + 0.1 * torch.randn(d, d),
            "w2": base["w2"] + 0.1 * torch.randn(d, d),
            "b1": base["b1"] + 0.1 * torch.randn(d)}

    features = [torch.randn(50, d), torch.randn(50, d)]
    merged = gaem_plus_merge(
        base, [ft1, ft2], [0.5, 0.5], features,
        alignment="orthogonal", decompose=True,
    )
    assert set(merged.keys()) == set(base.keys())


def test_interpolation_barrier():
    """Interpolation barrier between identical models should be zero."""
    from gaem.evaluation.barriers import interpolation_barrier
    sd = {"w": torch.randn(16, 16)}
    result = interpolation_barrier(sd, sd, eval_fn=lambda s: torch.norm(s["w"]).item())
    assert abs(result["barrier"]) < 1e-5, "Barrier between identical models should be ~0"


def test_domain_interference():
    """Interference metrics should return valid values."""
    from gaem.evaluation.interference import compute_domain_interference
    base = {"w": torch.zeros(16, 16)}
    tv1 = {"w": torch.randn(16, 16)}
    tv2 = {"w": torch.randn(16, 16)}
    results = compute_domain_interference(base, [tv1, tv2], ["speech", "music"])
    assert "cosine_speech_vs_music" in results
    assert -1 <= results["cosine_speech_vs_music"] <= 1


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
