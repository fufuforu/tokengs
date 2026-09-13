import torch

from tokengs.models.token_eru.metric_clustering import (
    historical_metric_cluster,
    historical_metric_cluster_oracle_audit,
)


def test_historical_cluster_shape_and_eps():
    embeddings = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]
    )
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]]
    )
    objectness = torch.ones(1, 4)
    output = historical_metric_cluster(
        embeddings, positions, objectness, {}, {}, eps=0.5
    )
    assert output.gaussian_cluster_ids.shape == (1, 32)
    assert output.cluster_count[0] >= 1
    assert output.rendered_masks.numel() == 0


def test_empty_cluster_input_is_safe():
    output = historical_metric_cluster(
        torch.empty(1, 0, 4),
        torch.empty(1, 0, 3),
        torch.empty(1, 0),
        {},
        {},
        eps=0.5,
    )
    assert output.gaussian_cluster_ids.shape == (1, 0)
    assert output.cluster_count == [0]


def test_objectness_filters_cluster_and_keeps_void_id():
    embeddings = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]
    )
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]]
    )
    objectness = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    output = historical_metric_cluster(
        embeddings, positions, objectness, {}, {}, eps=0.5
    )
    assert output.cluster_count == [1]
    assert output.cluster_confidence[0].shape == (1,)
    # The two rejected units are assigned to the extra void cluster.
    assert torch.equal(
        output.gaussian_cluster_ids[0].reshape(4, 8)[2:],
        torch.ones(2, 8, dtype=torch.long),
    )


def test_oracle_audit_is_explicitly_gt_derived_and_separate():
    embeddings = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]]
    )
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]]
    )
    # Channel 0 is the historical GT-derived background column.
    p_u_gt = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    output = historical_metric_cluster_oracle_audit(
        embeddings,
        positions,
        p_u_gt,
        {},
        {},
        background_index=0,
    )
    assert output.cluster_count == [1]
    assert output.gaussian_cluster_ids.shape == (1, 32)
