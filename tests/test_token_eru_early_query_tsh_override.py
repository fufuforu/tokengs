import torch

from tokengs.models.shared_unit_instance_head import SharedUnitInstanceHead


def test_native_seed_and_override_keep_one_query_bank():
    torch.manual_seed(0)
    head = SharedUnitInstanceHead(num_groups=100, unit_dim=256)
    units = torch.randn(1, 1024, 8, 256)
    seed = head.get_object_query_seed(1)
    assert seed.shape == (1, 100, 256)
    parameter_ids = [id(p) for p in head.parameters()]
    assert parameter_ids.count(id(head.group_tokens)) == 1
    old = head(units)
    override = seed + 0.1
    new = head(units, query_state_override=override)
    assert old["unit_logits"].shape in {(1, 1024, 8, 101), (1, 8192, 101)}
    assert new["unit_logits"].shape == old["unit_logits"].shape
    assert new["unit_logits"].shape == old["unit_logits"].shape
    assert not torch.equal(old["unit_logits"], new["unit_logits"])
    assert new["unit_logits"].shape[-1] == 101


def test_override_does_not_add_state_dict_parameter():
    head = SharedUnitInstanceHead(num_groups=100, unit_dim=256)
    assert not any("query_state_override" in key for key in head.state_dict())
