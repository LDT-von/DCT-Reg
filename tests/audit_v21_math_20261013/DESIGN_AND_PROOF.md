# Exact-collision diversity and anchored transport: independent mathematical audit

This is a toy mathematics audit, not a DCT implementation or a patient experiment.

## Why a smooth symmetric pair penalty cannot fix exact collapse

Let a differentiable scalar loss be invariant to slot permutations and common
translation, and depend only on pair differences. At `s_1 = ... = s_K`, symmetry
requires every slot gradient to be equal. Translation invariance requires their
sum to be zero. Consequently every slot gradient is zero. Merely replacing one
squared-distance polynomial by another does not remove this obstruction.

## Candidate: true distance hinge with a stated nonsmooth selection

Use `mean_{i<j} max(0, margin - ||s_i-s_j||_2)`, with positive margin.
For nonzero differences its derivative is the ordinary Euclidean derivative.
At an exact zero difference choose the unit norm subgradient
`u_ij = (c_i-c_j)/||c_i-c_j||`, with `u_ji = -u_ij`.
Codes are fixed distinct slot identities: one-hot vectors for `K <= D`, otherwise
equally spaced circle vertices in the first two coordinates (`D >= 2`). Codes do
not change the scalar forward loss and are not an additional anchor penalty.

At exact collapse, all hinges are strictly active. The selected whole-loss
gradient is a limit of ordinary gradients along the consistent perturbation
`s_i(epsilon) = s + epsilon*c_i`, as epsilon decreases to zero. Thus it is a
Bouligand limiting gradient and, in particular, a legitimate Clarke generalized
gradient; it is not an invented smooth derivative or a straight-through estimate.

Writing `P=K*(K-1)/2`, the chosen slot gradient is
`g_i = -(1/P) * sum_{j != i} u_ij`. Their sum is zero, so there is no common-mode
translation. A finite distinct code set has an exposed point: choose a vector
v with a unique maximizing code c_i. Every term `v dot (c_i-c_j)` is positive.
Hence `v dot g_i < 0`, and at least one slot gradient is nonzero. Since their
sum is zero, they cannot all be equal. For the one-hot/circle codes used here,
every slot has nonzero gradient in the tested cases.

A sufficiently small direct update `s_i <- s_i - eta*g_i` leaves all distances
below the margin but makes at least one pair distinct. Therefore the forward
hinge loss decreases strictly. This proves escape for directly optimized slot
parameters, not through every possible neural-network parameterization; a
singular model Jacobian can still block the corresponding update.

### Why fixed codes are preferable to mixed prototype fallback

Prototype differences could also guide the exact-zero selection. For a fully
collapsed slot set and a complete active pair graph, prototype differences for
distinct prototype groups, plus antisymmetric code differences within identical
prototype groups, cannot make every net slot force zero: an exposed prototype
group has positive group force, while its internal forces cancel. The choices
can be realized by the perturbation `epsilon*p_i + epsilon^2*c_i`.

Nevertheless, fixed codes provide a shorter proof, deterministic behavior when
prototypes also collapse, and no dependence on slowly changing prototype values.
For arbitrary partial-collision/active-edge configurations neither mechanism
guarantees the absence of all stationary points. Only exact global-collapse
escape is being claimed.

### Floating-point and verification boundary

Half-precision features are cast before subtraction to float32; double inputs
retain float64. Exact zero is detected by the maximum absolute represented
component being zero. No near-zero threshold selects the code direction. The
norm uses max-component scaling so tiny nonzero float32 differences retain the
ordinary direction rather than being treated as a collision through squared
norm underflow. Quantization can make distinct real values identical in float16;
the code can only operate on the represented values.

Ordinary finite-difference gradcheck passes away from zero and the hinge boundary.
At zero the norm is nondifferentiable, so a symmetric finite-difference gradcheck
must not be presented as evidence for the selected derivative. The audit instead
checks the stated selection and strict forward loss decrease.

## Explicit fixed-budget topology witness for a slotwise nonlinear anchor

Use two one-dimensional slots in both modalities: `h=g=(-1,+1)`.
The diagonal and antidiagonal plans each have total mass 1 and row/column budgets
`(.5,.5)`, so local quality ratios are all 1. Let phi be ReLU solely for this
counterexample, and define the anchored slotwise readout in either direction as
`mean_j [phi(g_j + .5*q_j*(moved_h_j-g_j)) - phi(g_j)]`.
Average the two directions. Under the diagonal plan the readout is 0. Under the
antidiagonal plan the anchor arguments are both 0, so the readout is -0.5.
Native means and transport budgets are identical; topology alone changes it.
Zero quality gives exact zero anchored output even if phi contains a bias.

This is an existence witness, not a universal guarantee: a linear phi, some odd
nonlinearities on symmetric examples, or a constant/dead learned module can still
erase topology. The useful fix is to perform slotwise nonlinear processing
before averaging, retain the anchor subtraction, and validate the actual module.

## Audit result

`audit_symmetry_math.py` passes 12 exact-collapse direct-step cases across
float16/float32/float64, 5 tiny-difference/quantization cases, ordinary-domain
gradcheck, and the same-budget topology witness. Detailed values are saved in
`symmetry_math_results.json`. No model source was modified.
