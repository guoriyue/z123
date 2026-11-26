#!/usr/bin/env python3
"""
Coefficients z1, z2, z3 of the Z-polynomial of the rank-7 matroid

        M_q = T(G ⊕ G),

where G = G_{4,q} is the rank-4 matroid of Bérczi–Coron–Eur–Li (arXiv:2610.17135, Def. 4.12 and
Prop. 4.13 with n = 6, m = q^3, b = q^9), ⊕ is the direct sum and T is the truncation (rank 8 -> 7).

Z-polynomial (Proudfoot–Xu–Young, arXiv:1706.05575):
        Z_M(t) = Σ_{F flat of M} t^{rk F} · P_{M/F}(t),      P = Kazhdan–Lusztig polynomial.
Z_M is palindromic of degree rk M, so here
        Z_M(t) = 1 + z1 t + z2 t^2 + z3 t^3 + z3 t^4 + z2 t^5 + z1 t^6 + t^7.

Python standard library only.   Run:  python3 z123.py
"""
from math import comb
from itertools import product

q = 27          # a prime power: the order of the finite field F_q

# ----------------------------------------------------------------------------------------------
# 1. The matroid G and its flag f-vector
#
# Ground set:  E(G) = P(V) ⊔ {p_c},  where V = F_q^6 = F_q^3 ⊕ F_q^3,  P(V) is the set of
# 1-dimensional subspaces of V, and there is one auxiliary point p_c for each coset c of the
# rank-metric code  C = { x ↦ a·x + b·x^q : a, b ∈ F_{q^3} } ⊂ End_{F_q}(F_{q^3})  (|C| = q^6, every
# nonzero element has rank ≥ 2).  For a linear map A: F_q^3 → F_q^3 let X_A = {(u, Au)} ⊂ V be its
# graph, a 3-dimensional subspace; [A] denotes the coset A + C.
# ----------------------------------------------------------------------------------------------
def gaussian_binomial(n, k):
    """[n choose k]_q = number of k-dimensional subspaces of F_q^n."""
    num = den = 1
    for i in range(k):
        num *= q ** (n - i) - 1
        den *= q ** (i + 1) - 1
    return num // den

m = q ** 3                      # number of auxiliary points  = number of cosets of C
b = q ** 9                      # number of linear maps A     = number of coalesced hyperplanes
N1, N2, N3 = (gaussian_binomial(6, k) for k in (1, 2, 3))   # points, lines, planes of PG(5, q)
t = q * q + q + 1               # [3 choose 1]_q = number of points (= number of lines) of PG(2, q)

# The rank-3 flats (= hyperplanes) of G come in four types.  For each type:
#   (number of flats,  |F|,  number of rank-2 flats below F,  number of flags  point < line < F)
hyperplane_types = [
    # (H) 3-element flats {x, p_c, p_c'} with x ∈ P(V), and {p_c, p_c', p_c''}
    (N1 * comb(m, 2) + comb(m, 3), 3,     3,     6),
    # (B) coalesced hyperplanes  P(X_A) ∪ {p_[A]}, one for each linear map A
    (b,                            t + 1, 2 * t, (q + 3) * t),
    # (U) P(W) ∪ {p_c}  with dim W = 2 and W contained in no X_A with [A] = c
    (m * N2 - b * t,               q + 2, q + 2, 3 * (q + 1)),
    # (N) P(X)  with dim X = 3 and X not of the form X_A
    (N3 - b,                       t,     t,     (q + 1) * t),
]
W1 = N1 + m                         # rank-1 flats: the points
W2 = N2 + N1 * m + comb(m, 2)       # rank-2 flats: projective lines, pairs {x, p_c}, pairs {p_c, p_c'}

# Flag f-vector of G:  alpha_G[S] = #{ chains of flats F_1 < ... < F_k with {rk F_i} = S },  S ⊆ {1, 2, 3}
alpha_G = {
    (): 1,
    (1,): W1,
    (2,): W2,
    (3,): sum(n for n, _, _, _ in hyperplane_types),
    (1, 2): (q + 1) * N2 + 2 * (N1 * m + comb(m, 2)),
    (1, 3): sum(n * pts for n, pts, _, _ in hyperplane_types),
    (2, 3): sum(n * lns for n, _, lns, _ in hyperplane_types),
    (1, 2, 3): sum(n * flags for n, _, _, flags in hyperplane_types),
}

# ----------------------------------------------------------------------------------------------
# 2. Flag numbers of M = T(G ⊕ G)
#
# The flats of G ⊕ G are the sets F' ⊔ F'' with F', F'' flats of G, and rk(F' ⊔ F'') = rk F' + rk F''.
# The truncation keeps the flats of rank ≤ 6.  Hence a chain of flats of M with ranks s_1 < ... < s_k
# is a pair of weakly increasing chains (multichains) in L(G), with ranks i_j and s_j − i_j.
# A multichain in L(G) with ranks i_1 ≤ ... ≤ i_k is counted by alpha_G of the set of its distinct
# ranks in {1, 2, 3}, because L(G) has a unique flat of rank 0 and a unique flat of rank 4.
# ----------------------------------------------------------------------------------------------
def alpha(*ranks):
    """alpha_M(S): number of chains of flats of M whose rank set is S = {ranks},  S ⊆ {1, ..., 6}."""
    total = 0
    for left in product(range(5), repeat=len(ranks)):
        right = tuple(s - i for s, i in zip(ranks, left))
        if any(j < 0 or j > 4 for j in right):
            continue
        if list(left) != sorted(left) or list(right) != sorted(right):
            continue
        proper = lambda seq: tuple(sorted(set(r for r in seq if 0 < r < 4)))
        total += alpha_G[proper(left)] * alpha_G[proper(right)]
    return total

# ----------------------------------------------------------------------------------------------
# 3. The coefficients
#
# By palindromicity, z_k = [t^{7-k}] Z_M(t) = Σ_F [t^{7-k-rk F}] P_{M/F}(t).  Only contractions of
# rank ≤ 5 contribute to z1, z2, z3, and for a matroid N with W_i rank-i flats and flag numbers a_S:
#     rk N ≤ 2:  P_N = 1
#     rk N = 3:  [t] P_N = W_2 − W_1
#     rk N = 4:  [t] P_N = W_3 − W_1
#     rk N = 5:  [t^2] P_N = W_3 − W_2 + a_{24} − a_{23} − a_{14} + a_{12}
# Summing over the flats F of the appropriate rank turns these into flag numbers of M.
# ----------------------------------------------------------------------------------------------
z1 = alpha(6)
z2 = alpha(5) + alpha(4, 6) - alpha(4, 5)
z3 = (alpha(4) + alpha(3, 6) - alpha(3, 4) + alpha(2, 5) - alpha(2, 4)
      + alpha(2, 4, 6) - alpha(2, 4, 5) - alpha(2, 3, 6) + alpha(2, 3, 4))

print("q =", q, "  |E(M)| =", 2 * W1, "  rk M = 7")
print("z1 =", z1)
print("z2 =", z2)
print("z3 =", z3)
print("z2^2 - z1*z3 =", z2 * z2 - z1 * z3)
print("log-concave at index 2" if z2 * z2 >= z1 * z3 else "NOT log-concave: z2^2 < z1*z3")
