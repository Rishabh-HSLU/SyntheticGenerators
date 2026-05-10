import torch
from generator import Generator
from discriminator import Discriminator
from gradient_penalty import gradient_penalty
from alignment import AlignmentModule


def train_sfag(
    dataloader,
    val_real:        torch.Tensor,
    latent_dim:      int   = 128,
    T:               int   = 252,
    n_assets:        int   = 10,
    hidden_dim:      int   = 256,
    n_critic:        int   = 5,
    lambda_gp:       float = 10.0,
    lr:              float = 2e-4,
    betas:           tuple = (0.5, 0.9),
    max_gen_iters:   int   = 50_000,
    warmup_iters:    int   = 10_000,
    lambda1:         float = 1.0,
    lambda2:         float = 1.0,
    lambda3:         float = 1.0,
    lambda4:         float = 1.0,
    patience:        int   = 500,
    clip_grad:       float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    G = Generator(latent_dim, T, n_assets, hidden_dim).to(device)
    D = Discriminator(T, n_assets, hidden_dim).to(device)
    alignment = AlignmentModule(lambda1=lambda1, lambda2=lambda2,
                                lambda3=lambda3, lambda4=lambda4).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    val_real   = val_real.to(device)
    data_iter  = iter(dataloader)
    best_gap   = float("inf")
    no_improve = 0
    loss_D     = torch.tensor(0.0, device=device)
    loss_G     = torch.tensor(0.0, device=device)
    history    = {"iter": [], "loss_D": [], "loss_G": [], "sfag_gap": []}

    for gen_iter in range(1, max_gen_iters + 1):
        try:
            real = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            real = next(data_iter).to(device)

        B = real.size(0)

        # ── critic updates (WGAN-GP: n_critic steps) ──
        for _ in range(n_critic):
            z      = torch.randn(B, latent_dim, device=device)
            fake   = G(z).detach()
            gp     = gradient_penalty(D, real, fake, device)
            loss_D = D(fake).mean() - D(real).mean() + lambda_gp * gp
            opt_D.zero_grad()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), clip_grad)
            opt_D.step()

        # ── generator update ──
        lambda_anneal = min(1.0, gen_iter / warmup_iters)
        z      = torch.randn(B, latent_dim, device=device)
        fake   = G(z)
        loss_G = -D(fake).mean() + lambda_anneal * alignment(real, fake)
        opt_G.zero_grad()
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), clip_grad)
        opt_G.step()

        # ── convergence check ──
        if gen_iter % 100 == 0:
            G.eval()
            with torch.no_grad():
                fv       = G(torch.randn(val_real.size(0), latent_dim, device=device))
                sfag_gap = alignment(val_real, fv).item()
            G.train()

            history["iter"].append(gen_iter)
            history["loss_D"].append(loss_D.item())
            history["loss_G"].append(loss_G.item())
            history["sfag_gap"].append(sfag_gap)

            if sfag_gap < best_gap:
                best_gap   = sfag_gap
                no_improve = 0
                torch.save(G.state_dict(), "best_G.pt")
            else:
                no_improve += 1

            print(f"iter {gen_iter:>6} | loss_D: {loss_D.item():.4f} | "
                  f"loss_G: {loss_G.item():.4f} | sfag_gap: {sfag_gap:.4f} | "
                  f"λ_anneal: {lambda_anneal:.3f}")

            if no_improve >= patience:
                print(f"Early stopping at iter {gen_iter} — sfag_gap converged.")
                break

    return G, D, history
