 # Physics-Informed Neural Networks for Solving Direct and Inverse Problems in Nonlinear Corneal Curvature
# Method

The objective of this study is to determine the corneal curvature of the eye by analyzing the solution of a second-order nonlinear differential equation subject to mixed boundary conditions. To solve this equation, two neural network-based approaches were employed:
1. strong-form PINN
2. Variational PINN (VPINN). a method utilizing first-order derivatives, which demonstrates a shorter execution time.
In other word, variational physics-informed neural network applied to nonlinear corneal curvature. the variational loss uses only first derivatives, lowering training cost versus PINNs.
Also, multi-seed tests show faster VPINN training and stable convergence in all regimes
Furthermore, a sensitivity analysis regarding the parameters was conducted. Finally, the problem was formulated as an inverse problem, and the corresponding codes were provided.


