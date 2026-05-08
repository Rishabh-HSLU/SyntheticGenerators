import torch
from generator import Generator
from discriminator import Discriminator
from gradient_penalty import gradient_penalty
from alignment import AlignmentModule

def train_sfag(
    dataloader,
    val_real: torch.Tensor,                 # held-out real batch for convergence check
    latent_dim: int = 128,
    T: int = 252,
    n_assets: int = 10,
    hidden_dim: int = 256,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr: float = 2e-4,
    betas: tuple = (0.5, 0.9),
    max_gen_iters: int = 50_000,
    warmup_iters: int = 10_000,             # 20% of 50,000
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    lambda4: float = 1.0,
    patience: int = 500,                    # early stopping patience
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    G = Generator(latent_dim, T, n_assets, hidden_dim).to(device)
    D = Discriminator(T, n_assets, hidden_dim).to(device)
    alignment = AlignmentModule(lambda1=lambda1, lambda2=lambda2,
                                lambda3=lambda3, lambda4=lambda4).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    val_real = val_real.to(device)

    gen_iter = 0
    best_sfag_gap = float("inf")
    no_improve = 0
    data_iter = iter(dataloader)

    while gen_iter < max_gen_iters:

        # ── Reload iterator if exhausted ──
        try:
            real = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            real = next(data_iter).to(device)

        B = real.size(0)

        # ── Critic update (n_critic steps) ──
        for _ in range(n_critic):
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z).detach()

            gp = gradient_penalty(D, real, fake, device)
            loss_D = D(fake).mean() - D(real).mean() + lambda_gp * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ── Annealing weight ──
        lambda_anneal = min(1.0, gen_iter / warmup_iters)

        # ── Generator update ──
        z = torch.randn(B, latent_dim, device=device)
        fake = G(z)

        loss_adv  = -D(fake).mean()
        loss_sfag = alignment(real, fake)
        loss_G    = loss_adv + lambda_anneal * loss_sfag

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        gen_iter += 1

        # ── Convergence check on held-out val batch ──
        if gen_iter % 100 == 0:
            G.eval()
            with torch.no_grad():
                z_val = torch.randn(val_real.size(0), latent_dim, device=device)
                fake_val = G(z_val)
            # Eval mode off for alignment (uses stats)
            G.train()
            sfag_gap = alignment(val_real, fake_val).item()

            if sfag_gap < best_sfag_gap:
                best_sfag_gap = sfag_gap
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

    return G, D