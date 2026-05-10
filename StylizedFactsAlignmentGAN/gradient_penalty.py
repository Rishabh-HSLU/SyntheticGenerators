import torch

def gradient_penalty(discriminator, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    B = real.size(0)

    # Random interpolation coefficient
    alpha = torch.rand(B, 1, 1, device=device)
    fake = fake.detach()
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # Critic score on interpolated
    d_interp = discriminator(interpolated)

    # Gradients w.r.t interpolated
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.view(B, -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()  # scalar
    return penalty