from tokengs.models.token_eru.early_query_codecoder import EarlyObjectQueryAdapter


def test_adapter_has_no_duplicate_query_parameter_or_external_state():
    adapter = EarlyObjectQueryAdapter()
    names = [name for name, _ in adapter.named_parameters()]
    assert len(names) == len(set(names))
    assert not any("group_tokens" in name or "query_seed" in name for name in names)
