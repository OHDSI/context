import math

import pytest
import rustworkx as rx
import torch
from hypll.optim import RiemannianAdam, RiemannianSGD

from context.dataset import GraphEmbeddingDataset
from context.eval import evaluate_model, fast_average_precision, hyperbolic_distance
from context.loss import candidate_loss, poincare_embeddings_loss
from context.model import create_embedding, pairwise_distance


@pytest.mark.parametrize("geometry", ["hyperbolic", "euclidean"])
@pytest.mark.parametrize("adam", [False, True])
def test_metric_gradients_and_fixed_root_after_updates(geometry, adam):
    c = 2.5
    model = create_embedding(4, 2, geometry, c, root_concept_index=2)
    with torch.no_grad():
        model.coordinates().copy_(torch.tensor([[.2, .0], [-.3, .0], [0., 0.], [.2, .2]]))
    opt = ((RiemannianAdam if adam else RiemannianSGD) if geometry == "hyperbolic"
           else (torch.optim.Adam if adam else torch.optim.SGD))(model.parameters(), lr=.015)
    sources = torch.tensor([[2], [0]])
    targets = torch.tensor([[0, 1, 3], [3, 1, 2]])
    distances = model(sources, targets)
    expected = .2 if geometry == "euclidean" else 2 / math.sqrt(c) * math.atanh(.2 * math.sqrt(c))
    assert distances[0, 0].item() == pytest.approx(expected)
    pairs = torch.stack((sources.expand_as(targets), targets), dim=-1)
    torch.testing.assert_close(model(pairs), distances)
    points = model.coordinates()
    reference = pairwise_distance(points, points, model.curvature)
    torch.testing.assert_close(distances, reference[sources, targets])
    before = candidate_loss(distances).item()
    for _ in range(5):
        opt.zero_grad()
        candidate_loss(model(sources, targets)).backward()
        assert torch.isfinite(points.grad).all()
        assert torch.equal(points.grad[2], torch.zeros(2))
        opt.step()
    assert candidate_loss(model(sources, targets)).item() < before
    assert torch.count_nonzero(points[2]) == 0
    if geometry == "hyperbolic":
        assert model.curvature.item() == pytest.approx(c)
        assert (c * points.square().sum(1) < 1).all()
    model.reset_parameters()
    assert torch.count_nonzero(points[2]) == 0


def test_boundary_self_distance_and_unbounded_euclidean_vectors():
    c = 4.0
    x = torch.tensor([[.49999, 0.], [-.49999, 0.]], dtype=torch.float64, requires_grad=True)
    d = hyperbolic_distance(x, x, c)
    torch.testing.assert_close(d.diag(), torch.zeros_like(d.diag()))
    assert d[0, 1].item() == pytest.approx(2 * math.atanh(.49999 * 2))
    d.sum().backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="outside"):
        hyperbolic_distance(torch.tensor([[.50, 0.]]), x.detach().float(), c)
    model = create_embedding(2, 2, "euclidean")
    with torch.no_grad():
        model.coordinates().copy_(torch.tensor([[0., 0.], [3., 4.]]))
    assert model(torch.tensor([[0, 1]])).item() == 5


def test_loss_is_per_positive_and_stable_at_large_offsets():
    d = torch.tensor([[0., 1.], [102., 101.]], dtype=torch.float64, requires_grad=True)
    expected = (math.log1p(math.exp(-1)) + math.log1p(math.exp(1))) / 2
    assert candidate_loss(d).item() == pytest.approx(expected)
    torch.testing.assert_close(candidate_loss(d), poincare_embeddings_loss(d, torch.tensor([[1, 0], [1, 0]])))
    assert candidate_loss(torch.full((3, 2), 1.e20)).item() == pytest.approx(math.log(2))
    candidate_loss(d).backward()
    assert torch.isfinite(d.grad).all() and torch.all(d.grad[:, 0] > 0)


def graph_dataset(nodes, edges):
    g = rx.PyDiGraph()
    g.add_nodes_from(list(range(nodes)))
    g.add_edges_from_no_data(edges)
    return GraphEmbeddingDataset(g, directed=True)


def test_ties_and_multiple_positives_receive_query_macro_ap():
    ds = graph_dataset(5, [(0, 1), (0, 2), (3, 1)])
    model = create_embedding(5, 1, "euclidean")
    with torch.no_grad():
        model.weight[:, 0] = torch.tensor([0., 1., -1., 3., 1.])
    assert evaluate_model(model, ds, batch_size=1) == pytest.approx((1.5, 7/12))
    assert evaluate_model(model, ds, batch_size=32) == pytest.approx((1.5, 7/12))
    assert fast_average_precision(torch.tensor([1, 0, 1, 0]), torch.ones(4)).item() == .5


def test_model_metric_changes_neighbor_order():
    ds = graph_dataset(3, [(0, 1)])
    for geometry, expected in [("euclidean", (1, 1)), ("hyperbolic", (2, .5))]:
        model = create_embedding(3, 1, geometry)
        with torch.no_grad():
            model.coordinates()[:, 0] = torch.tensor([.5, .65, .34])
        assert evaluate_model(model, ds) == pytest.approx(expected)


@pytest.mark.parametrize("geometry,c", [("manhattan", 1), ("hyperbolic", 0), ("hyperbolic", float("nan"))])
def test_invalid_geometry(geometry, c):
    with pytest.raises(ValueError):
        create_embedding(3, 1, geometry, c)


@pytest.mark.parametrize("geometry", ["euclidean", "hyperbolic"])
def test_mps_gradients_match_cpu_for_repeated_root_and_nonroot_indices(geometry):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cpu = create_embedding(5, 2, geometry, 2.5, root_concept_index=2)
    gpu = create_embedding(5, 2, geometry, 2.5, root_concept_index=2)
    gpu.load_state_dict(cpu.state_dict())
    gpu = gpu.to("mps")
    sources = torch.tensor([[0], [2], [2], [3]])
    targets = torch.tensor([[2, 1, 4], [1, 0, 3], [3, 1, 4], [1, 2, 4]])
    for model, device in ((cpu, "cpu"), (gpu, "mps")):
        candidate_loss(model(sources.to(device), targets.to(device))).backward()
        assert torch.count_nonzero(model.coordinates().grad[2]) == 0
    torch.testing.assert_close(gpu.coordinates().grad.cpu(), cpu.coordinates().grad, rtol=2e-4, atol=1e-5)
