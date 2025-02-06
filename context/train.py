from argparse import Namespace
import csv
from pathlib import Path
import pathlib

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianSGD, RiemannianAdam
import polars as pl
import simple_parsing
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm

from context.args import Args
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

    dataset = GraphEmbeddingDataset(graph=g.subgraph[args.graph_id],
                                    # device=args.device,
                                    num_negative_samples = args.negative_samples)
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
            print(f"Burn-in epoch {epoch} loss: {average_loss}")

    # Now we use the actual learning rate
    optimizer = optimizer_object(
        params=model.parameters(),
        lr=args.learning_rate# ,
        # weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000)
    early_stopper = EarlyStopper(patience=1000)
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
            print(f"Early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch} loss: {average_loss} lr: {scheduler.get_last_lr()}")
        loss_per_epoch[epoch] = average_loss

    model.load_state_dict(torch.load(output_directory.joinpath(f"epoch:{best_epoch}-loss:{best_loss:3f}-{args.model_file}"))['state_dict'])

    weights_np = model.weight.data.cpu().numpy()

    tsv_file_path = "/Users/hjohn/Desktop/output/vec.tsv"
    with open(tsv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in weights_np:
            writer.writerow(row)

    concept_dict = {}
    csv_path = "/Users/hjohn/data/vocabulary/snomed/CONCEPT.csv"
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            concept_id = int(row['concept_id'])
            concept_name = row['concept_name']
            concept_dict[concept_id] = concept_name

    labels_file_path = "/Users/hjohn/Desktop/output/labels.tsv"
    with open(labels_file_path, 'w', newline='') as labels_file:
        writer = csv.writer(labels_file, delimiter='\t')
        for index, node_id in enumerate(g.subgraph["intermediate"].nodes()):
            concept_name = concept_dict[node_id]
            writer.writerow([concept_name])

    # mean_rank, map_score = evaluate_model(model, dataset)
    # save rank and map as txt file
    # savetxt(output_directory.joinpath(f"mean-rank:{mean_rank}_map:{map_score}.txt"), [mean_rank, map_score])

# def save_for_plp(embeddings, dataset, args):
#     concept_ids = torch.as_tensor(list(nx.get_node_attributes(dataset.graph, 'concept_id').values()), dtype=torch.long)
#     torch.save({'concept_ids': concept_ids, 'embeddings': embeddings}, 'poincare_embeddings_snomed_plp.pt')

def save_model(model, args, output_directory, epoch, average_loss, loss_per_epoch):
    # only keep top args.save_top models
    saved_models = len(list(output_directory.glob(f"epoch:*-{args.model_file}")))
    while saved_models >= args.save_top:
        # find worst loss value from filename of saved models
        candidate_models = list(output_directory.glob(f"epoch:*-{args.model_file}"))
        losses = [float(model.name.split('-')[1].split('-')[0].split(':')[1]) for model in candidate_models]
        worst_loss = max(losses)
        worst_model = [model for model in candidate_models if float(model.name.split('-')[1].split('-')[0].split(':')[1]) == worst_loss][0]
        # remove the worst model
        if worst_model.exists():
            worst_model.unlink()
        saved_models -= 1
    torch.save({'state_dict': model.state_dict(),
            'args': args.__dict__, # convert dataclass to dict
            'losses': loss_per_epoch,
            'epoch': epoch,
            }, output_directory.joinpath(f"epoch:{epoch}-loss:{average_loss:3f}-{args.model_file}"))
