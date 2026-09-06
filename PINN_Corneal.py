import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, root_scalar
from scipy.interpolate import interp1d




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

Neorons = 256




class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, Neorons),
            nn.SiLU(),
            nn.Linear(Neorons, Neorons),
            nn.SiLU(),
            nn.Linear(Neorons, Neorons),
            nn.SiLU(),
            nn.Linear(Neorons, 1)
        )

    def forward(self, x):
        return self.net(x)





def trial_solution(x, net):
    return (1 - x) * net(x)



def differential_equation(x, alpha, beta, net):
    u = trial_solution(x, net)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx - alpha * u + beta / torch.sqrt(1 + u_x**2)



def loss_function(x, alpha, beta, net):
    eq_term = differential_equation(x, alpha, beta, net)
    eq_loss = torch.mean(eq_term**2)

    x0 = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True).to(DEVICE)
    u0 = trial_solution(x0, net)
    u0_x = torch.autograd.grad(u0, x0, grad_outputs=torch.ones_like(x0), create_graph=True)[0]
    bc_loss = (u0_x - 0.0)**2

    return eq_loss + bc_loss, eq_loss.item(), bc_loss.item()



def solve_cornea_rk4(alpha, beta, guess, N=1000):
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)

    y1[0] = guess
    y2[0] = 0.0

    for i in range(N - 1):
        y1_i = y1[i]
        y2_i = y2[i]
        def f1(y1i, y2i): return y2i
        def f2(y1i, y2i): return alpha * y1i - beta / np.sqrt(1 + y2i**2)

        k1_y1 = f1(y1_i, y2_i)
        k1_y2 = f2(y1_i, y2_i)

        y1_k2 = y1_i + 0.5 * h * k1_y1
        y2_k2 = y2_i + 0.5 * h * k1_y2
        k2_y1 = f1(y1_k2, y2_k2)
        k2_y2 = f2(y1_k2, y2_k2)

        y1_k3 = y1_i + 0.5 * h * k2_y1
        y2_k3 = y2_i + 0.5 * h * k2_y2
        k3_y1 = f1(y1_k3, y2_k3)
        k3_y2 = f2(y1_k3, y2_k3)

        y1_k4 = y1_i + h * k3_y1
        y2_k4 = y2_i + h * k3_y2
        k4_y1 = f1(y1_k4, y2_k4)
        k4_y2 = f2(y1_k4, y2_k4)

        y1[i + 1] = y1_i + (h / 6.0) * (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1)
        y2[i + 1] = y2_i + (h / 6.0) * (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2)

    return x, y1


def shooting_cornea(alpha, beta, N=1000):
    def boundary_error(guess):
        _, y1 = solve_cornea_rk4(alpha, beta, guess, N)
        return y1[-1]

    sol = root_scalar(boundary_error, bracket=[-1, 1], method='brentq', xtol=1e-10)
    if not sol.converged:
        raise RuntimeError("Shooting method failed")

    return solve_cornea_rk4(alpha, beta, sol.root, N)



def compute_norms(x, net, u_ref_interp, du_ref_interp):
    x = x.clone().detach().requires_grad_(True)
    u_pred = trial_solution(x, net)

    u_pred_x = torch.autograd.grad(
        u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=False
    )[0]

    u_pred = u_pred.detach()
    u_pred_x = u_pred_x.detach()

    diff = (u_pred[:, 0] - u_ref_interp).cpu().numpy()
    L2 = np.sqrt(np.mean(diff**2))

    Linf = np.max(np.abs(diff))

    diff_x = (u_pred_x[:, 0] - du_ref_interp).cpu().numpy()
    H1 = np.sqrt(np.mean(diff_x**2))

    return L2, Linf, H1



def train_model(alpha, beta):
    net = NeuralNet().to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    x_train = torch.linspace(0, 1, 120, requires_grad=True).view(-1, 1).to(DEVICE)
    x_eval = torch.linspace(0, 1, 500).view(-1, 1)

    x_ref, u_ref = shooting_cornea(alpha, beta, N=1000)
    interp_func = interp1d(x_ref, u_ref, kind='cubic')

    u_ref_eval = interp_func(x_eval.numpy().flatten())
    du_ref_eval = np.gradient(u_ref_eval, x_eval.numpy().flatten())

    L2_hist, Linf_hist, H1_hist = [], [], []

    
    loss_hist = []

    total_epochs = 2000
    warmup_epochs = 100
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-7)

    def adjust_lr(epoch):
        if epoch < warmup_epochs:
            new_lr = (epoch + 1) / warmup_epochs * 1e-3
            optimizer.param_groups[0]['lr'] = new_lr
        else:
            scheduler_cosine.step()

    for epoch in range(total_epochs):
        optimizer.zero_grad()
        loss, eq_loss, bc_loss = loss_function(x_train, alpha, beta, net)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        adjust_lr(epoch)

        L2, Linf, H1 = compute_norms(x_eval, net, u_ref_eval, du_ref_eval)
        L2_hist.append(L2)
        Linf_hist.append(Linf)
        H1_hist.append(H1)

        
        loss_hist.append(loss.item())

        if epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:4d} | Loss={loss.item():.2e} | ODE_Loss={eq_loss:.2e} | "
                f"BC_Loss={bc_loss:.2e} | L2={L2:.2e} | Linf={Linf:.2e} | "
                f"H1={H1:.2e} | LR={lr:.1e}"
            )

        if loss.item() < 1e-12:
            print(f"Early stopping at epoch {epoch}")
            break

    optimizer_lbfgs = optim.LBFGS(net.parameters(), lr=1.0, max_iter=800)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _ = loss_function(x_train, alpha, beta, net)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    return net, L2_hist, Linf_hist, H1_hist, loss_hist



def plot_error_history(L2_hist, Linf_hist, H1_hist):
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
    plt.savefig("error_history.png", dpi=300)
    plt.show()



def plot_loss_history(loss_hist):
    epochs = np.arange(len(loss_hist))

    plt.figure(figsize=(8, 5))
    plt.semilogy(epochs, loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss vs Epoch")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("loss_history.png", dpi=300)
    plt.show()



def plot_results(net):
    x_test = torch.linspace(0, 1, 200).view(-1, 1).to(DEVICE)
    u_pred = trial_solution(x_test, net).detach().cpu().numpy()
    plt.plot(x_test.cpu(), u_pred)
    plt.title("PINN Solution")
    plt.show()


def plot_error(alpha, beta, net):
    x_test = np.linspace(0, 1, 500)
    x_ref, u_ref = shooting_cornea(alpha, beta, N=1000)
    u_ref_interp = interp1d(x_ref, u_ref, kind='cubic')(x_test)

    u_nn = trial_solution(torch.tensor(x_test).view(-1, 1).to(DEVICE), net)
    u_nn = u_nn.detach().cpu().numpy().flatten()

    err = np.abs(u_nn - u_ref_interp)
    print("Max Error:", np.max(err))
    print("Mean Error:", np.mean(err))

    plt.semilogy(x_test, err)
    plt.title("Absolute Error")
    plt.show()




if __name__ == "__main__":
    ALPHA = 2
    BETA = 2

    net, L2_hist, Linf_hist, H1_hist, loss_hist = train_model(ALPHA, BETA)

    plot_results(net)
    plot_error(ALPHA, BETA, net)
    plot_error_history(L2_hist, Linf_hist, H1_hist)

    
    plot_loss_history(loss_hist)
