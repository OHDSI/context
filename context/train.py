from argparse import Namespace
import csv
import pathlib

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianSGD, RiemannianAdam
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm
import polars as pl
import numpy as np

from context.dataset import GraphEmbeddingDataset
from context.eval import evaluate_model
from context.graph import Graph
from context.loss import poincare_embeddings_loss
from context.model import PoincareEmbedding
from context.stopper import EarlyStopper

def train(args: Namespace):
    g = Graph.load(args.graph_file)
    root_node_index = next(
        node_index for node_index in g.subgraph[args.graph_id].node_indexes()
        if g.subgraph[args.graph_id][node_index] == args.root_node_label
    )

    dataset = GraphEmbeddingDataset(graph=g.subgraph[args.graph_id], num_negative_samples = args.negative_samples)
    dataloader = DataLoader(dataset, sampler=BatchSampler(sampler=RandomSampler(dataset), batch_size=args.batch_size,
                                                          drop_last=False))

    output_directory = pathlib.Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    poincare_ball = PoincareBall(c=Curvature(args.curvature))

    model = PoincareEmbedding(
        num_embeddings=dataset.num_nodes,
        embedding_dim=args.embedding_dim,
        manifold=poincare_ball,
        root_concept_index=root_node_index
    )
    if args.compile:
        model = torch.compile(model, fullgraph=True)
    model = model.to(args.device)
    if args.optimizer == 'adam':
        optimizer_object = RiemannianAdam
    else:
        optimizer_object = RiemannianSGD

    if args.burn_in:
        optimizer = optimizer_object(
            params=model.parameters(),
            lr=args.learning_rate / args.burn_in_lr_divisor,
        )
        for epoch in range(args.burn_in_epochs):
            average_loss = torch.tensor(0.0, device=args.device)
            for idx, (edges, edge_label_targets) in tqdm(enumerate(dataloader)):
                edges = edges.to(args.device)
                edge_label_targets = edge_label_targets.to(args.device)

                optimizer.zero_grad()

                dists = model(edges)
                loss = poincare_embeddings_loss(distances=dists, targets=edge_label_targets)
                loss.backward()
                optimizer.step()

                average_loss += loss.detach()

            average_loss /= len(dataloader)
            tqdm.write(f"Burn-in epoch {epoch} loss: {average_loss}")

    # Now we use the actual learning rate
    optimizer = optimizer_object(
        params=model.parameters(),
        lr=args.learning_rate# ,
        # weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=args.lr_reduce_factor,
                                                           patience=args.lr_reduce_patience)
    early_stopper = EarlyStopper(patience=args.early_stop_patience)
    loss_per_epoch = torch.empty(args.epochs, device=args.device)
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        average_loss = torch.tensor(0.0, device=args.device)
        for idx, (edges, edge_label_targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            edges = edges.to(args.device)
            edge_label_targets = edge_label_targets.to(args.device)

            optimizer.zero_grad()

            dists = model(edges)
            loss = poincare_embeddings_loss(distances=dists, targets=edge_label_targets)
            loss.backward()
            optimizer.step()

            average_loss += loss.detach()
        average_loss /= len(dataloader)
        scheduler.step(average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch
            save_model(model, args, output_directory, epoch, average_loss, loss_per_epoch)

        if early_stopper.update(average_loss):
            tqdm.write(f"Early stopping at epoch {epoch}")
            break
        tqdm.write(f"Epoch {epoch} loss: {average_loss} lr: {scheduler.get_last_lr()}")
        loss_per_epoch[epoch] = average_loss

    model.load_state_dict(
        torch.load(output_directory.joinpath("models", f"epoch_{best_epoch}-loss_{best_loss:3f}-{args.model_file}"),
                   weights_only=False)['state_dict'])

    weights_np = model.weight.data.cpu().numpy()
    vec_file_path = output_directory.joinpath("vec.tsv")
    with open(vec_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in weights_np:
            writer.writerow(row)

    csv_path = args.concept_map
    concept_df = pl.read_csv(csv_path, separator="\t", quote_char=None)
    concept_dict = dict(zip(
        concept_df["concept_id"].cast(pl.Int64).to_list(),
        concept_df["concept_name"].to_list()
    ))

    labels = [concept_dict[node_id] for node_id in g.subgraph[args.graph_id].nodes()]
    df_labels = pl.DataFrame({"label": labels})
    labels_file_path = output_directory.joinpath("labels.tsv")
    df_labels.write_csv(str(labels_file_path), separator="\t", include_header=False)

    mean_rank, map_score = evaluate_model(model, dataset)
    # save rank and map as txt file
    np.savetxt(output_directory.joinpath(f"mean-rank:{mean_rank}_map:{map_score}.txt"), [mean_rank, map_score])

# def save_for_plp(embeddings, dataset, args):
#     concept_ids = torch.as_tensor(list(nx.get_node_attributes(dataset.graph, 'concept_id').values()), dtype=torch.long)
#     torch.save({'concept_ids': concept_ids, 'embeddings': embeddings}, 'poincare_embeddings_snomed_plp.pt')

def save_model(model, args, output_directory, epoch, average_loss, loss_per_epoch):
    models_dir = output_directory.joinpath("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    # only keep top args.save_top models
    saved_models = len(list(models_dir.glob(f"epoch_*-{args.model_file}")))
    while saved_models >= args.save_top:
        # find the worst loss value from filename of saved models
        candidate_models = list(models_dir.glob(f"epoch_*-{args.model_file}"))
        losses = [float(model.name.split('-')[1].split('_')[1]) for model in candidate_models]
        worst_loss = max(losses)
        worst_model = [model for model in candidate_models if float(model.name.split('-')[1].split('_')[1]) == worst_loss][0]
        # remove the worst model
        if worst_model.exists():
            worst_model.unlink()
        saved_models -= 1
    torch.save({'state_dict': model.state_dict(),
                'args': args.__dict__,
                'losses': loss_per_epoch,
                'epoch': epoch,
                }, models_dir.joinpath(f"epoch_{epoch}-loss_{average_loss:3f}-{args.model_file}"))
