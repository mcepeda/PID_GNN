import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_sum
from src.dataset.functions_data import (
    get_ratios,
    find_mask_no_energy,
    find_cluster_id,
    get_particle_features,
    get_hit_features,
    calculate_distance_to_boundary,
    concatenate_Particles_GT,
)


def create_inputs_from_table(output, hits_only, prediction=False, hit_chis=False):
    """Used by graph creation to get nodes and edge features

    Args:
        output (_type_): input from the root reading
        hits_only (_type_): reading only hits or also tracks
        prediction (bool, optional): if running in eval mode. Defaults to False.

    Returns:
        _type_: all information to construct a graph
    """
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))

    (
        pos_xyz_hits,
        p_hits,
        e_hits,
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        unique_list_particles,
        cluster_id,
        hit_type_feature,
        pandora_pfo_link,
        daughters,
        hit_link_modified,
        connection_list,
        chi_squared_tracks,
        labels_true,
    ) = get_hit_features(
        output, number_hits, prediction, number_part, hit_chis=hit_chis
    )

    # features particles
    y_data_graph = get_particle_features(
        unique_list_particles, output, prediction, connection_list
    )

    assert len(y_data_graph) == len(unique_list_particles)

    result = [
        y_data_graph,  # y_data_graph[~mask_particles],
        p_hits,
        e_hits,
        cluster_id,
        hit_particle_link,
        pos_xyz_hits,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        pandora_pfo_link,
        hit_type_feature,
        hit_link_modified,
        labels_true,
    ]
    if hit_chis:
        result.append(
            chi_squared_tracks,
        )
    else:
        result.append(None)
    hit_type = hit_type_feature
    # if hits only remove tracks, otherwise leave tracks

    # if we want the tracks keep only 1 track hit per charged particle.
    hit_mask = hit_type == 0
    hit_mask = ~hit_mask
    for i in range(1, len(result)):
        if result[i] is not None:
            result[i] = result[i][hit_mask]
    hit_type_one_hot = torch.nn.functional.one_hot(
        hit_type_feature[hit_mask] - 1, num_classes=3
    )

    result.append(hit_type_one_hot)
    result.append(connection_list)
    return result


def create_graph(
    output,
    config=None,
    n_noise=0,
):
    hits_only = config.graph_config.get(
        "only_hits", False
    )  # Whether to only include hits in the graph
    # standardize_coords = config.graph_config.get("standardize_coords", False)
    extended_coords = config.graph_config.get("extended_coords", False)
    prediction = config.graph_config.get("prediction", False)
    hit_chis = config.graph_config.get("hit_chis_track", False)
    (
        y_data_graph,
        p_hits,
        e_hits,
        cluster_id,
        hit_particle_link,
        pos_xyz_hits,
        pandora_cluster,
        pandora_cluster_energy,
        pandora_pfo_energy,
        pandora_pfo_link,
        hit_type,
        hit_link_modified,
        labels_true,
        chi_squared_tracks,
        hit_type_one_hot,
        connections_list,
    ) = create_inputs_from_table(
        output, hits_only=hits_only, prediction=prediction, hit_chis=hit_chis
    )
    graph_coordinates = pos_xyz_hits  # / 3330  # divide by detector size
    if pos_xyz_hits.shape[0] > 0:
        graph_empty = False
        g = dgl.graph(([], []))
        g.add_nodes(graph_coordinates.shape[0])
        if hits_only == False:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 8
        else:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 9

        g.ndata["h"] = hit_features_graph
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        g.ndata["label_true"] = 1.0 * labels_true.view(-1, 1)
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hit_type
        g.ndata[
            "e_hits"
        ] = e_hits  # if no tracks this is e and if there are tracks this fills the tracks e values with p
        if hit_chis:
            g.ndata["chi_squared_tracks"] = chi_squared_tracks
        g.ndata["particle_number"] = cluster_id
        g.ndata["hit_link_modified"] = hit_link_modified
        g.ndata["particle_number_nomap"] = hit_particle_link
        if prediction:
            g.ndata["pandora_cluster"] = pandora_cluster
            g.ndata["pandora_pfo"] = pandora_pfo_link
            g.ndata["pandora_cluster_energy"] = pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = pandora_pfo_energy
        y_data_graph.calculate_corrected_E(g, connections_list)

    return [g, y_data_graph], graph_empty


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    # list_y = add_batch_number(list_graphs)
    # ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys
