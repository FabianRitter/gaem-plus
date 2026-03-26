"""
GAEM+ (Generalized Audio Encoder Merging)

A framework for merging speech, music, and audio encoders using:
- Orthogonal alignment (Procrustes solver, semi-permutation)
- Low-rank + sparse task vector decomposition (LoRS)
- Pre-training orthogonalization (OSRM) [future]

Usage:
    from gaem.alignment import procrustes_align, permutation_align
    from gaem.decomposition import lors_decompose
    from gaem.merging import task_arithmetic_merge, gaem_plus_merge
"""

__version__ = "0.1.0"
