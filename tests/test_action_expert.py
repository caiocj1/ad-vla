import copy

import torch

from ad_vla.models.layers.action_expert import ActionExpert


def _expert_inputs():
    return {
        "action_tokens": torch.randn(2, 4, 32, requires_grad=True),
        "state_tokens": torch.randn(2, 1, 32, requires_grad=True),
        "language_states": torch.randn(2, 3, 24, requires_grad=True),
        "visual_states": torch.randn(2, 5, 24, requires_grad=True),
        "language_mask": torch.tensor([[True, True, True], [False, True, True]]),
        "visual_mask": torch.tensor(
            [[True, True, True, True, True], [False, False, True, True, True]]
        ),
    }


def test_training_checkpoint_matches_direct_forward_and_gradients():
    torch.manual_seed(11)
    checkpointed = ActionExpert(
        dim=32,
        depth=2,
        num_heads=4,
        language_dim=24,
        visual_dim=24,
        ffn_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
        use_cape=False,
    )
    direct = copy.deepcopy(checkpointed)
    checkpointed.train()
    direct.eval()  # Evaluation takes the same direct path used before this change.

    checkpointed_inputs = _expert_inputs()
    direct_inputs = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in checkpointed_inputs.items()
    }

    checkpointed_output = checkpointed(**checkpointed_inputs)
    direct_output = direct(**direct_inputs)
    assert torch.equal(checkpointed_output, direct_output)

    checkpointed_output.square().mean().backward()
    direct_output.square().mean().backward()
    for name in (
        "action_tokens",
        "state_tokens",
        "language_states",
        "visual_states",
    ):
        assert torch.allclose(
            checkpointed_inputs[name].grad,
            direct_inputs[name].grad,
            rtol=1e-5,
            atol=1e-6,
        )
    for checkpointed_parameter, direct_parameter in zip(
        checkpointed.parameters(), direct.parameters()
    ):
        assert torch.allclose(
            checkpointed_parameter.grad,
            direct_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )
