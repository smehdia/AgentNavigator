import math

import torch
import torch.nn.functional as F


def get_graph_distance(
    distance_graph_dict,
    node_id1: str,
    node_id2: str,
) -> float | None:
    distance = distance_graph_dict.get((node_id1, node_id2))
    if distance is None:
        return None
    if isinstance(distance, float) and math.isinf(distance):
        return None
    return float(distance)


def get_symmetric_graph_distance(
    distance_graph_dict,
    node_id1: str,
    node_id2: str,
) -> float | None:
    forward = get_graph_distance(
        distance_graph_dict,
        node_id1,
        node_id2,
    )

    backward = get_graph_distance(
        distance_graph_dict,
        node_id2,
        node_id1,
    )

    available = [
        distance
        for distance in [forward, backward]
        if distance is not None
    ]

    if not available:
        return None

    return min(available)


def direct_text_alignment_loss(
    predicted_embedding: torch.Tensor,  # [B, D]
    target_text_embedding: torch.Tensor,  # [B, D]
    available_mask: torch.Tensor | None = None,  # [B]
) -> torch.Tensor:
    """Cosine alignment of a predicted embedding to a frozen text embedding."""
    predicted = F.normalize(predicted_embedding, dim=-1)
    target = F.normalize(target_text_embedding.detach(), dim=-1)
    per_sample_loss = 1.0 - (predicted * target).sum(dim=-1)

    if available_mask is None:
        return per_sample_loss.mean()

    mask = available_mask.to(per_sample_loss.dtype)
    return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)


def soft_set_alignment_loss(
    predicted_embedding: torch.Tensor,       # [B, D]
    target_set_embeddings: torch.Tensor,     # [B, K, D]
    set_mask: torch.Tensor,                  # [B, K], bool
    available_mask: torch.Tensor | None = None,  # [B]
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Soft best-match alignment of a prediction to a set of frozen embeddings.
    """
    predicted = F.normalize(predicted_embedding, dim=-1)
    targets = F.normalize(target_set_embeddings.detach(), dim=-1)

    cosine = torch.einsum("bd,bkd->bk", predicted, targets)
    logits = cosine / temperature
    logits = logits.masked_fill(~set_mask.bool(), float("-inf"))

    weights = torch.softmax(logits, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    soft_similarity = (weights * cosine).sum(dim=-1)

    per_sample_loss = 1.0 - soft_similarity
    sample_available = set_mask.any(dim=-1)
    if available_mask is not None:
        sample_available = sample_available & available_mask.bool()

    mask = sample_available.to(per_sample_loss.dtype)
    return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)


def batch_siamese_text_similarity_loss(
    predicted_embeddings: torch.Tensor,      # [B, D]
    target_text_embeddings: torch.Tensor,    # [B, D]
    available_mask: torch.Tensor | None = None,  # [B]
) -> torch.Tensor:
    """
    Match all pairwise predicted similarities to frozen-text similarities.
    """
    predicted = F.normalize(predicted_embeddings, dim=-1)
    predicted_similarity = predicted @ predicted.T

    with torch.no_grad():
        target = F.normalize(target_text_embeddings, dim=-1)
        target_similarity = target @ target.T

    batch_size = predicted.shape[0]
    pair_mask = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=predicted.device,
    )

    if available_mask is not None:
        available = available_mask.bool()
        pair_mask = pair_mask & available[:, None] & available[None, :]

    if not pair_mask.any():
        return predicted.sum() * 0.0

    return F.smooth_l1_loss(
        predicted_similarity[pair_mask],
        target_similarity[pair_mask],
    )

def interactive_element_loss(
    current_element_logits: torch.Tensor,  # [B, N]
    interactive_bboxes: list[list[tuple[float, float, float, float]]],
    image_hw: tuple[int, int],             # (height, width)
    grid_hw: tuple[int, int],              # (grid_h, grid_w)
    min_overlap: float = 0.25,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    """
    Supervises per-token element logits using interactive-element bounding boxes.

    Args:
        current_element_logits:
            Raw element logits with shape [B, N].

        interactive_bboxes:
            One list of pixel-coordinate boxes per batch sample:
            [
                [(x1, y1, x2, y2), ...],  # sample 0
                [(x1, y1, x2, y2), ...],  # sample 1
            ]

        image_hw:
            Original screenshot size as (height, width).
            Assumes all screenshots in the batch have the same size.

        grid_hw:
            Visual-token grid as (grid_h, grid_w).

        min_overlap:
            A token is positive when at least this fraction of its area
            overlaps an interactive-element box.

    Returns:
        Scalar weighted BCE loss.
    """
    batch_size, num_tokens = current_element_logits.shape
    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw

    if num_tokens != grid_h * grid_w:
        raise ValueError(
            f"Logits contain {num_tokens} tokens, but grid_hw={grid_hw} "
            f"contains {grid_h * grid_w} cells."
        )

    if len(interactive_bboxes) != batch_size:
        raise ValueError(
            f"Expected {batch_size} bbox lists, "
            f"but received {len(interactive_bboxes)}."
        )

    device = current_element_logits.device
    dtype = current_element_logits.dtype

    # Token-cell boundaries in pixel coordinates.
    x_edges = torch.linspace(
        0,
        image_w,
        grid_w + 1,
        device=device,
        dtype=torch.float32,
    )
    y_edges = torch.linspace(
        0,
        image_h,
        grid_h + 1,
        device=device,
        dtype=torch.float32,
    )

    token_y1, token_x1 = torch.meshgrid(
        y_edges[:-1],
        x_edges[:-1],
        indexing="ij",
    )
    token_y2, token_x2 = torch.meshgrid(
        y_edges[1:],
        x_edges[1:],
        indexing="ij",
    )

    token_boxes = torch.stack(
        [
            token_x1.flatten(),
            token_y1.flatten(),
            token_x2.flatten(),
            token_y2.flatten(),
        ],
        dim=-1,
    )  # [N, 4]

    token_areas = (
        (token_boxes[:, 2] - token_boxes[:, 0])
        * (token_boxes[:, 3] - token_boxes[:, 1])
    ).clamp_min(1e-6)

    targets = torch.zeros(
        batch_size,
        num_tokens,
        device=device,
        dtype=torch.float32,
    )

    for batch_index, boxes in enumerate(interactive_bboxes):
        if not boxes:
            continue

        boxes_tensor = torch.as_tensor(
            boxes,
            device=device,
            dtype=torch.float32,
        )

        if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] != 4:
            raise ValueError(
                "Each sample must contain boxes in "
                "(x1, y1, x2, y2) format."
            )

        # Correct accidentally reversed coordinates.
        x1 = torch.minimum(boxes_tensor[:, 0], boxes_tensor[:, 2])
        y1 = torch.minimum(boxes_tensor[:, 1], boxes_tensor[:, 3])
        x2 = torch.maximum(boxes_tensor[:, 0], boxes_tensor[:, 2])
        y2 = torch.maximum(boxes_tensor[:, 1], boxes_tensor[:, 3])

        boxes_tensor = torch.stack([x1, y1, x2, y2], dim=-1)

        # Clip boxes to screenshot boundaries.
        boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(
            0, image_w
        )
        boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(
            0, image_h
        )

        # [M, 1, 4] against [1, N, 4].
        elements = boxes_tensor[:, None, :]
        tokens = token_boxes[None, :, :]

        intersection_x1 = torch.maximum(
            elements[..., 0],
            tokens[..., 0],
        )
        intersection_y1 = torch.maximum(
            elements[..., 1],
            tokens[..., 1],
        )
        intersection_x2 = torch.minimum(
            elements[..., 2],
            tokens[..., 2],
        )
        intersection_y2 = torch.minimum(
            elements[..., 3],
            tokens[..., 3],
        )

        intersection_area = (
            (intersection_x2 - intersection_x1).clamp_min(0)
            * (intersection_y2 - intersection_y1).clamp_min(0)
        )  # [M, N]

        token_overlap = intersection_area / token_areas[None, :]

        maximum_overlap = token_overlap.amax(dim=0)

        targets[batch_index] = (
            maximum_overlap >= min_overlap
        ).float()

    targets = targets.to(dtype)

    positive_count = targets.sum()
    negative_count = targets.numel() - positive_count

    pos_weight = (
        negative_count / positive_count.clamp_min(1.0)
    ).clamp(max=max_pos_weight)

    return F.binary_cross_entropy_with_logits(
        current_element_logits,
        targets,
        pos_weight=pos_weight,
    )


def transition_embedding_loss(
    predicted_current_embedding: torch.Tensor,  # [B, D]
    current_screen_embedding: torch.Tensor,      # [B, D]
    history_available: torch.Tensor | None = None,  # [B]
    smooth_l1_weight: float = 0.1,
) -> torch.Tensor:
    """
    Makes the previous-screen + action prediction match the actual
    current-screen embedding.

    First-step samples can be excluded using history_available=False.
    """
    if predicted_current_embedding.shape != current_screen_embedding.shape:
        raise ValueError(
            "Predicted and target embeddings must have the same shape."
        )

    predicted = F.normalize(
        predicted_current_embedding,
        dim=-1,
    )

    # Detach prevents the target current-screen branch from moving merely
    # to make this auxiliary prediction task easier.
    target = F.normalize(
        current_screen_embedding.detach(),
        dim=-1,
    )

    cosine_loss = 1.0 - (predicted * target).sum(dim=-1)

    smooth_l1_loss = F.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
    ).mean(dim=-1)

    per_sample_loss = (
        cosine_loss
        + smooth_l1_weight * smooth_l1_loss
    )

    if history_available is not None:
        mask = history_available.to(per_sample_loss.dtype)

        return (
            per_sample_loss * mask
        ).sum() / mask.sum().clamp_min(1.0)

    return per_sample_loss.mean()

def semantic_text_alignment_loss(
    predicted_tab_embedding: torch.Tensor,
    predicted_subtab_embedding: torch.Tensor,
    predicted_description_embedding: torch.Tensor,
    target_tab_embedding: torch.Tensor,
    target_subtab_embedding: torch.Tensor,
    target_description_embedding: torch.Tensor,
    tab_available: torch.Tensor | None = None,
    subtab_available: torch.Tensor | None = None,
    description_available: torch.Tensor | None = None,
    tab_weight: float = 1.0,
    subtab_weight: float = 1.0,
    description_weight: float = 1.0,
) -> dict[str, torch.Tensor]:

    def masked_cosine_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
        available: torch.Tensor | None,
    ) -> torch.Tensor:
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target.detach(), dim=-1)

        per_sample_loss = 1.0 - (predicted * target).sum(dim=-1)

        if available is None:
            return per_sample_loss.mean()

        mask = available.to(per_sample_loss.dtype)

        return (
            per_sample_loss * mask
        ).sum() / mask.sum().clamp_min(1.0)

    loss_tab = masked_cosine_loss(
        predicted_tab_embedding,
        target_tab_embedding,
        tab_available,
    )

    loss_subtab = masked_cosine_loss(
        predicted_subtab_embedding,
        target_subtab_embedding,
        subtab_available,
    )

    loss_description = masked_cosine_loss(
        predicted_description_embedding,
        target_description_embedding,
        description_available,
    )

    total_loss = (
        tab_weight * loss_tab
        + subtab_weight * loss_subtab
        + description_weight * loss_description
    )

    return {
        "loss": total_loss,
        "loss_tab": loss_tab,
        "loss_subtab": loss_subtab,
        "loss_description": loss_description,
    }

def siamese_text_similarity_loss(
    predicted_embedding1: torch.Tensor,   # [B, D_pred]
    predicted_embedding2: torch.Tensor,   # [B, D_pred]
    target_text_embedding1: torch.Tensor, # [B, D_text]
    target_text_embedding2: torch.Tensor, # [B, D_text]
    available_mask: torch.Tensor | None = None,  # [B], bool
) -> torch.Tensor:
    """
    Matches similarity between two predicted embeddings to similarity
    between their corresponding frozen text embeddings.
    """
    if predicted_embedding1.shape != predicted_embedding2.shape:
        raise ValueError(
            "Predicted embedding pairs must have the same shape."
        )

    if target_text_embedding1.shape != target_text_embedding2.shape:
        raise ValueError(
            "Target text embedding pairs must have the same shape."
        )

    predicted_similarity = F.cosine_similarity(
        predicted_embedding1,
        predicted_embedding2,
        dim=-1,
    )  # [B]

    with torch.no_grad():
        target_similarity = F.cosine_similarity(
            target_text_embedding1,
            target_text_embedding2,
            dim=-1,
        )  # [B]

    per_pair_loss = F.smooth_l1_loss(
        predicted_similarity,
        target_similarity,
        reduction="none",
    )

    if available_mask is None:
        return per_pair_loss.mean()

    mask = available_mask.to(
        device=per_pair_loss.device,
        dtype=per_pair_loss.dtype,
    )

    return (
        per_pair_loss * mask
    ).sum() / mask.sum().clamp_min(1.0)



def node_prototype_loss(
    embeddings: torch.Tensor,       # [B, D]
    node_ids: torch.Tensor,         # [B]
    node_prototypes: torch.Tensor,  # [num_nodes, D]
    temperature: float = 0.07,
) -> torch.Tensor:

    embeddings = F.normalize(embeddings, dim=-1)
    prototypes = F.normalize(node_prototypes, dim=-1)

    logits = embeddings @ prototypes.T
    logits = logits / temperature

    return F.cross_entropy(logits, node_ids)


def soft_geodesic_siamese_loss(
    embedding1: torch.Tensor,      # [B, D]
    embedding2: torch.Tensor,      # [B, D]
    graph_distances: torch.Tensor, # [B]
    graph_temperature: float = 1.0,
    max_distance: float | None = None,
    available_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Matches embedding cosine similarity to a soft target derived from
    shortest graph distance.
    """
    predicted_similarity = F.cosine_similarity(
        embedding1,
        embedding2,
        dim=-1,
    )

    distances = graph_distances.float()

    if max_distance is not None:
        distances = distances.clamp(max=max_distance)

    with torch.no_grad():
        transformed_distance = torch.log1p(distances)

        target_similarity = torch.exp(
            -transformed_distance / graph_temperature
        )

    per_pair_loss = F.smooth_l1_loss(
        predicted_similarity,
        target_similarity,
        reduction="none",
    )

    if available_mask is None:
        return per_pair_loss.mean()

    mask = available_mask.to(per_pair_loss.dtype)

    return (
        per_pair_loss * mask
    ).sum() / mask.sum().clamp_min(1.0)

def batch_soft_geodesic_loss(
    embeddings: torch.Tensor,          # [B, D]
    graph_distance_matrix: torch.Tensor, # [B, B]
    graph_temperature: float = 1.0,
    valid_pair_mask: torch.Tensor | None = None,
) -> torch.Tensor:

    embeddings = F.normalize(embeddings, dim=-1)

    predicted_similarity = embeddings @ embeddings.T

    with torch.no_grad():
        target_similarity = torch.exp(
            -torch.log1p(graph_distance_matrix.float())
            / graph_temperature
        )

    batch_size = embeddings.shape[0]

    pair_mask = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=embeddings.device,
    )

    if valid_pair_mask is not None:
        pair_mask = pair_mask & valid_pair_mask.bool()

    if not pair_mask.any():
        return embeddings.sum() * 0.0

    return F.smooth_l1_loss(
        predicted_similarity[pair_mask],
        target_similarity[pair_mask],
    )

def pairwise_node_localization_loss(
    embedding1: torch.Tensor,  # [B, D]
    embedding2: torch.Tensor,  # [B, D]
    same_node: torch.Tensor,   # [B], bool or 0/1
    negative_margin: float = 0.2,
) -> torch.Tensor:
    """
    Same node:
        push cosine similarity toward 1.

    Different node:
        penalize cosine similarity above negative_margin.
    """
    similarity = F.cosine_similarity(
        embedding1,
        embedding2,
        dim=-1,
    )

    same_node = same_node.to(
        device=similarity.device,
        dtype=similarity.dtype,
    )

    positive_loss = 1.0 - similarity

    negative_loss = F.relu(
        similarity - negative_margin
    )

    per_pair_loss = (
        same_node * positive_loss
        + (1.0 - same_node) * negative_loss
    )

    return per_pair_loss.mean()

def on_graph_detection_loss(
    on_graph_logits: torch.Tensor,  # [B]
    on_graph_targets: torch.Tensor, # [B], 1=on graph, 0=off graph
    positive_weight: float | None = None,
) -> torch.Tensor:
    """
    Binary on-graph/off-graph detection loss.

    Positive class:
        on graph

    Negative class:
        off graph
    """
    if on_graph_logits.ndim != 1:
        raise ValueError(
            "on_graph_logits must have shape [B]."
        )

    if on_graph_targets.shape != on_graph_logits.shape:
        raise ValueError(
            "on_graph_targets must have the same shape as logits."
        )

    targets = on_graph_targets.to(
        device=on_graph_logits.device,
        dtype=on_graph_logits.dtype,
    )

    if positive_weight is None:
        return F.binary_cross_entropy_with_logits(
            on_graph_logits,
            targets,
        )

    pos_weight = torch.tensor(
        positive_weight,
        device=on_graph_logits.device,
        dtype=on_graph_logits.dtype,
    )

    return F.binary_cross_entropy_with_logits(
        on_graph_logits,
        targets,
        pos_weight=pos_weight,
    )