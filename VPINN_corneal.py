# vpinn_corneal_legendre_shooting_GL_final.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import time

alpha = 2
beta = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
layers = [1, 32, 32, 1]
activation = torch.tanh
lr = 1e-3
epochs = 2000
n_test_funcs = 8
quad_points = 100
weight_neumann = 1.0
print_every = 100


class MLP(nn.Module):
    def __init__(self, layers, act=activation):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers)-1):
            self.net.append(nn.Linear(layers[i], layers[i+1]))
        self.act = act
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.net):
            y = layer(y)
            if i < len(self.net)-1:
                y = self.act(y)
        return y


def u_trial(x, net):
    return (1.0 - x) * net(x)


def legendre_test_functions(x_np, n_funcs):
    xi = 2*x_np - 1.0
    V = np.zeros((n_funcs, len(x_np)))
    dV = np.zeros((n_funcs, len(x_np)))
    for j in range(n_funcs):
        coeffs = np.zeros(j+1); coeffs[j]=1
        Pj = L.legval(xi, coeffs)
        dPj_dxi = L.legval(xi, L.legder(coeffs))
        Pj_at_1 = L.legval([1.0], coeffs)[0]
        V[j,:] = Pj - Pj_at_1
        dV[j,:] = 2.0 * dPj_dxi
    return V, dV


def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5*(x+1.0), 0.5*w


net = MLP(layers).to(device).to(dtype)
x_quad_np, w_quad_np = gauss_legendre(quad_points)
V_np, dV_np = legendre_test_functions(x_quad_np, n_test_funcs)

x_quad = torch.tensor(x_quad_np.reshape(-1,1), dtype=dtype, requires_grad=True, device=device)
w_quad = torch.tensor(w_quad_np.reshape(-1,1), dtype=dtype, device=device)
V = torch.tensor(V_np, dtype=dtype, device=device)
dV = torch.tensor(dV_np, dtype=dtype, device=device)
x0 = torch.tensor([[0.0]], dtype=dtype, requires_grad=True, device=device)


def compute_u_and_up(x_tensor):
    u = u_trial(x_tensor, net)
    up = torch.autograd.grad(u, x_tensor, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    return u, up


def variational_loss():
    u_q, up_q = compute_u_and_up(x_quad)
    u_q = u_q.view(-1)
    up_q = up_q.view(-1)
    f_q = 1.0 / torch.sqrt(1.0 + up_q**2)

    u_b = u_q.unsqueeze(0)
    up_b = up_q.unsqueeze(0)
    f_b = f_q.unsqueeze(0)

    integrand = up_b * dV + alpha * u_b * V - beta * f_b * V
    w_row = w_quad.view(1,-1)
    I_domain = torch.sum(integrand * w_row, dim=1)

    u0, up0 = compute_u_and_up(x0)

    I_total = I_domain
    loss_var = torch.mean(I_total**2)
    loss_neu = (up0.view(-1)[0])**2

    loss = loss_var + weight_neumann * loss_neu
    return loss, loss_var.item(), (weight_neumann * loss_neu).item()


def compute_residual_array():
    x_test = torch.linspace(0, 1, 400, dtype=dtype, device=device).reshape(-1,1)
    x_test.requires_grad = True
    u, up = compute_u_and_up(x_test)
    upp = torch.autograd.grad(up, x_test, grad_outputs=torch.ones_like(up),
                            create_graph=True, retain_graph=True)[0]
    residual = upp - alpha*u + beta/torch.sqrt(1 + up**2)
    return x_test.detach().cpu().numpy().flatten(), residual.detach().cpu().numpy().flatten()

def check_residual():
    x_test_np, r_np = compute_residual_array()
    print(f"Residual stats - Mean: {np.mean(r_np):.3e}, Std: {np.std(r_np):.3e}, Max: {np.max(np.abs(r_np)):.3e}")
    return np.mean(r_np**2)


x_ref = np.linspace(0, 1, 2001)

def corneal_rhs(x, y):
    u, up = y
    return [up, alpha*u - beta/np.sqrt(1 + up**2)]

def shooting(u0_guess):
    sol = solve_ivp(
        corneal_rhs, [0, 1], [u0_guess, 0.0],
        t_eval=x_ref, method='RK45',
        rtol=1e-12, atol=1e-12
    )
    return sol.y[0], sol.y[1]

def shooting_target(g):
    u_sol, _ = shooting(g)
    return u_sol[-1]

print("Computing reference solution...")
res = root_scalar(shooting_target, bracket=[0.0, 2.0], method='bisect', xtol=1e-14)
u0_opt = res.root
u_ref, du_ref = shooting(u0_opt)
print(f"Reference solution found with u(0) = {u0_opt:.6f}")

# evaluation grid
x_eval = np.linspace(0,1,400).reshape(-1,1)
x_eval_t = torch.tensor(x_eval, dtype=dtype, device=device, requires_grad=True)
x_eval_flat = x_eval.flatten()

u_ref_eval = np.interp(x_eval_flat, x_ref, u_ref)
du_ref_eval = np.interp(x_eval_flat, x_ref, du_ref)


def compute_errors():
    u_eval_t, du_eval_t = compute_u_and_up(x_eval_t)
    u_vp = u_eval_t.detach().cpu().numpy().flatten()
    du_vp = du_eval_t.detach().cpu().numpy().flatten()

    diff_u  = u_vp - u_ref_eval
    diff_du = du_vp - du_ref_eval

    L2_u = np.sqrt(np.trapz(diff_u**2, x_eval_flat))
    Linf_u = np.max(np.abs(diff_u))
    H1 = np.sqrt(np.trapz(diff_du**2, x_eval_flat))
    return L2_u, Linf_u, H1


optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

loss_history = []
L2_history = []
Linf_history = []
H1_history = []

start_time = time.time()
for ep in range(1, epochs+1):

    optimizer.zero_grad()
    loss, lv, ln = variational_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())
    L2_u, Linf_u, H1 = compute_errors()
    L2_history.append(L2_u)
    Linf_history.append(Linf_u)
    H1_history.append(H1)

    if ep % print_every == 0 or ep == 1:
        print(f"Epoch {ep:6d}: loss={loss.item():.3e}, var={lv:.3e}, neu={ln:.3e}, "
              f"L2={L2_u:.3e}, Linf={Linf_u:.3e}, H1={H1:.3e}")

end_time = time.time()
print(f"\nAdam completed in {end_time-start_time:.2f} s")


print("\nLBFGS fine-tuning...")

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _, _ = variational_loss()
    loss.backward()
    return loss

optimizer_lbfgs = optim.LBFGS(
    net.parameters(),
    max_iter=500,
    tolerance_grad=1e-12,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe"
)

optimizer_lbfgs.step(closure)
print("LBFGS done.")



def plot_error_history(L2_hist, Linf_hist, H1_hist):
    """Plot error history similar to PINN code"""
    epochs = np.arange(len(L2_hist))

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, L2_hist, label="L2 Error")
    plt.semilogy(epochs, Linf_hist, label="L∞ Error")
    plt.semilogy(epochs, H1_hist, label="H1 Error")

    plt.xlabel("Epoch")
    plt.ylabel("Error (log scale)")
    plt.title("Error vs Epoch")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("vpinn_error_history.png", dpi=300)
    plt.show()

def plot_loss_history(loss_hist):
    """Plot loss history similar to PINN code"""
    epochs = np.arange(len(loss_hist))

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss vs Epoch")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("vpinn_loss_history.png", dpi=300)
    plt.show()

def plot_solution_comparison():
    """Plot VPINN solution vs reference solution"""
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    x_plot_t = torch.tensor(x_plot, dtype=dtype, device=device, requires_grad=True)
    u_plot_t, _ = compute_u_and_up(x_plot_t)
    u_plot = u_plot_t.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, u_plot, 'b-', label="VPINN Solution", linewidth=2)
    plt.plot(x_ref, u_ref, 'r--', label="Reference Solution", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("VPINN Solution vs Reference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vpinn_solution.png", dpi=300)
    plt.show()

def plot_error_distribution():
    """Plot absolute error distribution"""
    x_plot = np.linspace(0, 1, 500)
    # FIX: Set requires_grad=True for the tensor
    x_plot_t = torch.tensor(x_plot.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
    u_final, _ = compute_u_and_up(x_plot_t)
    u_final_np = u_final.detach().cpu().numpy().flatten()
    u_ref_interp = np.interp(x_plot, x_ref, u_ref)
    
    err = np.abs(u_final_np - u_ref_interp)
    
    print(f"Max Error: {np.max(err):.3e}")
    print(f"Mean Error: {np.mean(err):.3e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(x_plot, err)
    plt.xlabel("x")
    plt.ylabel("Absolute Error (log scale)")
    plt.title("Absolute Error Distribution")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("vpinn_error_distribution.png", dpi=300)
    plt.show()

def plot_residual():
    """Plot residual distribution"""
    x_res_np, r_np = compute_residual_array()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_res_np, r_np, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.title("Residual r(x) = u'' - αu + β/√(1+u'^2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vpinn_residual.png", dpi=300)
    plt.show()

def plot_derivative_comparison():
    """Plot derivative comparison"""
    x_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    # FIX: Set requires_grad=True for the tensor
    x_plot_t = torch.tensor(x_plot, dtype=dtype, device=device, requires_grad=True)
    _, du_plot_t = compute_u_and_up(x_plot_t)
    du_plot = du_plot_t.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, du_plot, 'b-', label="VPINN Derivative", linewidth=2)
    plt.plot(x_ref, du_ref, 'r--', label="Reference Derivative", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u'(x)")
    plt.title("Derivative Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vpinn_derivative.png", dpi=300)
    plt.show()



print("\n" + "="*50)
print("GENERATING PLOTS")
print("="*50)

# Plot 1: Loss history
plot_loss_history(loss_history)

# Plot 2: Error history
plot_error_history(L2_history, Linf_history, H1_history)

# Plot 3: Solution comparison
plot_solution_comparison()

# Plot 4: Derivative comparison
plot_derivative_comparison()

# Plot 5: Error distribution
plot_error_distribution()

# Plot 6: Residual
plot_residual()


L2_final, Linf_final, H1_final = compute_errors()
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Final Errors:")
print(f"L2 error: {L2_final:.3e}")
print(f"L∞ error: {Linf_final:.3e}")
print(f"H1 seminorm error: {H1_final:.3e}")
print("="*50)