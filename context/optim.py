import torch
from hypll.manifolds.poincare_ball.math.diffgeom import euc_to_tangent, expmap
from hypll.optim import RiemannianAdam, RiemannianSGD


class RowWiseSGD(torch.optim.Optimizer):
    def __init__(self, model, lr):
        self.curvature = model.curvature
        super().__init__([model.coordinates()], {"lr": lr})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weights, = group["params"]
            gradient = weights.grad.coalesce()
            rows = gradient.indices()[0]
            values = gradient.values()
            if self.curvature is None:
                weights.index_add_(0, rows, values, alpha=-group["lr"])
            else:
                points = weights.index_select(0, rows)
                tangent = euc_to_tangent(points, values, self.curvature)
                weights.index_copy_(0, rows, expmap(points, -group["lr"] * tangent, self.curvature))


def make_optimizer(model, args, rate):
    device = model.coordinates().device.type
    sparse_capable = device in ("cpu", "cuda") and args.optimizer == "sgd"
    if args.sgd_updates == "sparse" and not sparse_capable:
        raise ValueError("Sparse row updates require SGD on CPU or CUDA")
    sampled_rows = args.batch_size * (args.negative_samples + 2)
    sparse = args.sgd_updates == "sparse" or (args.sgd_updates == "auto" and sparse_capable
             and model.geometry == "hyperbolic" and model.num_embeddings >= 4 * sampled_rows)
    model.sparse_gradients = sparse
    if sparse:
        return RowWiseSGD(model, rate)
    if model.geometry == "hyperbolic":
        optimizer = RiemannianAdam if args.optimizer == "adam" else RiemannianSGD
    else:
        optimizer = torch.optim.Adam if args.optimizer == "adam" else torch.optim.SGD
    return optimizer(model.parameters(), lr=rate)
