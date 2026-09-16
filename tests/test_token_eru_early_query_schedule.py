from types import SimpleNamespace

from tokengs.models.semantic_tokengs_v6 import SemanticTokenGSv6


def test_early_query_gate_schedule():
    opt = SimpleNamespace()
    assert SemanticTokenGSv6.token_eru_early_query_gate(0, opt) == 0.0
    assert SemanticTokenGSv6.token_eru_early_query_gate(1, opt) == 0.04
    assert SemanticTokenGSv6.token_eru_early_query_gate(25, opt) == 1.0
    assert SemanticTokenGSv6.token_eru_early_query_gate(100, opt) == 1.0

