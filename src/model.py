"""
Tiny Recursive Model (TRM) Implementation
Based on "Less is More: Recursive Reasoning with Tiny Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for reasoning tasks.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of network layers
        n_recursions: Latent recursion steps
        T_cycles: Deep recursion cycles
        seq_len: Sequence length
        use_attention: Use self-attention vs MLP
    """

    def __init__(
        self,
        vocab_size=10,
        hidden_size=256,
        num_layers=2,
        n_recursions=6,
        T_cycles=3,
        seq_len=81,
        use_attention=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_recursions = n_recursions
        self.T_cycles = T_cycles
        self.seq_len = seq_len
        self.use_attention = use_attention

        self.input_embedding = nn.Embedding(vocab_size, hidden_size)
        self.y_init = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.01)
        self.z_init = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.01)

        if use_attention:
            self.net = self._build_transformer_network()
        else:
            self.net = self._build_mlp_network()

        self.output_head = nn.Linear(hidden_size, vocab_size)
        self.q_head = nn.Linear(hidden_size, 1)

    def _build_mlp_network(self):
        layers = [
            nn.Linear(self.hidden_size * 3, self.hidden_size * 4),
            nn.ReLU(),
        ]
        
        for i in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
                nn.ReLU(),
            ])
        
        layers.append(nn.Linear(self.hidden_size * 4, self.hidden_size))
        return nn.Sequential(*layers)

    def _build_transformer_network(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size * 3,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        return nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers),
            nn.Linear(self.hidden_size * 3, self.hidden_size),
        )

    def latent_recursion(self, x, y, z):
        for _ in range(self.n_recursions):
            combined = torch.cat([x, y, z], dim=-1)
            z = self.net(combined)
            if hasattr(self, "use_residual") and self.use_residual:
                z = z + self.z_init.expand(x.size(0), -1, -1)

        combined = torch.cat([x, y, z], dim=-1)
        y = self.net(combined)
        return y, z

    def deep_recursion(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T_cycles - 1):
                y, z = self.latent_recursion(x, y, z)

        y, z = self.latent_recursion(x, y, z)
        y_pred = self.output_head(y)
        q_pred = torch.sigmoid(self.q_head(y.mean(dim=1)))
        return (y.detach(), z.detach()), y_pred, q_pred

    def forward(self, x_input, y_true=None, n_supervision=16):
        B = x_input.size(0)
        device = x_input.device

        x = self.input_embedding(x_input)
        y = self.y_init.expand(B, -1, -1).to(device)
        z = self.z_init.expand(B, -1, -1).to(device)

        total_loss = 0.0
        steps_taken = 0

        for step in range(n_supervision):
            (y, z), y_pred, q_pred = self.deep_recursion(x, y, z)

            if y_true is not None:
                ce_loss = F.cross_entropy(
                    y_pred.reshape(-1, self.vocab_size),
                    y_true.reshape(-1),
                    reduction="mean",
                )

                correct = (
                    (y_pred.argmax(dim=-1) == y_true).float().mean(dim=1, keepdim=True)
                )
                halt_loss = F.binary_cross_entropy(q_pred, correct, reduction="mean")

                step_loss = ce_loss + 0.5 * halt_loss
                total_loss += step_loss
                steps_taken += 1

                if q_pred.mean() > 0.5:
                    break

        if y_true is not None:
            return y_pred, {
                "total_loss": total_loss / max(steps_taken, 1),
                "ce_loss": ce_loss,
                "halt_loss": halt_loss,
                "steps": steps_taken,
                "final_q": q_pred.mean().item(),
            }
        else:
            return y_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "n_recursions": self.n_recursions,
            "T_cycles": self.T_cycles,
            "seq_len": self.seq_len,
            "use_attention": self.use_attention,
            "num_parameters": self.count_parameters(),
        }


if __name__ == "__main__":
    # Test model
    print("Testing Tiny Recursive Model...")

    model = TinyRecursiveModel(
        vocab_size=10,
        hidden_size=256,
        num_layers=2,
        n_recursions=6,
        T_cycles=3,
        use_attention=False,
    )

    print(f"\nModel Configuration:")
    for k, v in model.get_config().items():
        print(f"  {k}: {v}")

    print(f"\nParameters: {model.count_parameters() / 1e6:.2f}M")

    # Test forward pass
    batch_size = 4
    x = torch.randint(0, 10, (batch_size, 81))
    y = torch.randint(1, 10, (batch_size, 81))

    print(f"\nInput shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Training mode
    pred, losses = model(x, y, n_supervision=2)
    print(f"\nOutput shape: {pred.shape}")
    print(f"Losses: {losses}")

    # Inference mode
    pred = model(x)
    print(f"\nInference output shape: {pred.shape}")

    print("\n Model test passed!")
