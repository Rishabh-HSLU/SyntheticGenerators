import os
import json
import torch
import matplotlib.pyplot as plt
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
    checkpoint_dir:  str   = "checkpoints/sfag_run",
    use_wandb:       bool  = False,
    wandb_plot_every: int  = 1000,   # log a real-vs-fake chart every N gen iters
    breakthrough_delta: float = 0.05,  # event fires when sfag_gap drops by >= this
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if use_wandb:
        import wandb            # local import — keeps train.py runnable without wandb installed

    os.makedirs(checkpoint_dir, exist_ok=True)

    G = Generator(latent_dim, T, n_assets, hidden_dim).to(device)
    D = Discriminator(T, n_assets, hidden_dim).to(device)
    alignment = AlignmentModule(lambda1=lambda1, lambda2=lambda2,
                                lambda3=lambda3, lambda4=lambda4).to(device)

    if use_wandb:
        wandb.watch(G, log="gradients", log_freq=100)
        wandb.watch(D, log="gradients", log_freq=100)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    val_real      = val_real.to(device)
    data_iter     = iter(dataloader)
    best_gap      = float("inf")
    prev_sfag_gap = float("inf")
    no_improve    = 0
    loss_D        = torch.tensor(0.0, device=device)
    loss_G        = torch.tensor(0.0, device=device)
    history       = {"iter": [], "loss_D": [], "loss_G": [], "sfag_gap": []}

    for gen_iter in range(1, max_gen_iters + 1):
        try:
            real = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            real = next(data_iter).to(device)

        B = real.size(0)

        # ── critic updates ──
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

            # Save history every check — survives kernel crashes
            with open(f'{checkpoint_dir}/history.json', 'w') as f:
                json.dump(history, f)

            if sfag_gap < best_gap:
                best_gap   = sfag_gap
                no_improve = 0
                torch.save(G.state_dict(), f'{checkpoint_dir}/best_G.pt')
            else:
                no_improve += 1

            print(f"iter {gen_iter:>6} | loss_D: {loss_D.item():.4f} | "
                  f"loss_G: {loss_G.item():.4f} | sfag_gap: {sfag_gap:.4f} | "
                  f"λ_anneal: {lambda_anneal:.3f}")

            # ── wandb scalar logging ──
            if use_wandb:
                wandb.log({
                    "Match/Loss_D":         loss_D.item(),
                    "Match/Loss_G":         loss_G.item(),
                    "SFAG/Total_Gap":       sfag_gap,
                    "SFAG/Best_Gap":        best_gap,
                    "System/Lambda_Anneal": lambda_anneal,
                }, step=gen_iter)

                # Breakthrough event: significant gap drop
                if gen_iter > 100 and (prev_sfag_gap - sfag_gap) > breakthrough_delta:
                    wandb.log({"Events/Breakthrough": 1.0,
                               "Events/Gap_Drop":     prev_sfag_gap - sfag_gap},
                              step=gen_iter)
                prev_sfag_gap = sfag_gap

            if no_improve >= patience:
                print(f"Early stopping at iter {gen_iter} — sfag_gap converged.")
                break

        # ── wandb image logging (real vs fake chart) ──
        if use_wandb and gen_iter % wandb_plot_every == 0:
            G.eval()
            with torch.no_grad():
                z_plot    = torch.randn(1, latent_dim, device=device)
                fake_plot = G(z_plot)[0, :, 0].cpu().numpy()
            G.train()
            real_plot = real[0, :, 0].cpu().numpy()

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            axes[0].plot(real_plot, color="steelblue", lw=0.6)
            axes[0].set_title(f"Real (iter {gen_iter})")
            axes[1].plot(fake_plot, color="tomato",    lw=0.6)
            axes[1].set_title(f"Synthetic (iter {gen_iter})")
            plt.tight_layout()
            wandb.log({"Replays/Market_Chart": wandb.Image(fig)}, step=gen_iter)
            plt.close(fig)

    return G, D, history