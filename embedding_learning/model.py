from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCROLL, SWIPE, TAP, NONE = 1, 2, 3, 4


@dataclass
class VisualTokens:
    features: torch.Tensor      # [N, token_dim]
    positions: torch.Tensor     # [N, 2], normalized x/y centers
    image_hw: tuple[int, int]


@dataclass
class EmbedderOutput:
    embedding: torch.Tensor
    on_graph_logit: torch.Tensor
    predicted_current_screen: torch.Tensor

    predicted_tab_embedding: torch.Tensor
    predicted_subtab_embedding: torch.Tensor
    predicted_description_embedding: torch.Tensor

    predicted_waypoint_embedding: torch.Tensor
    predicted_transition_hint_embedding: torch.Tensor
    predicted_intent_embedding: torch.Tensor

    current_screen: torch.Tensor
    previous_screen: torch.Tensor
    action_vector: torch.Tensor

    current_element_logits: torch.Tensor
    previous_element_logits: torch.Tensor

    current_global_attention: torch.Tensor
    current_element_attention: torch.Tensor
    previous_global_attention: torch.Tensor
    previous_element_attention: torch.Tensor
    action_attention: torch.Tensor


class TransitionPredictor(nn.Module):
    """
    Predicts the current screen vector from:
        previous_screen_vector + previous_action_vector
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        previous_screen_vector: torch.Tensor,  # [B, D]
        action_vector: torch.Tensor,           # [B, D]
    ) -> torch.Tensor:
        predicted_current = self.network(
            torch.cat(
                [
                    previous_screen_vector,
                    action_vector,
                ],
                dim=-1,
            )
        )

        return F.normalize(predicted_current, dim=-1)

def make_token_positions(grid_h, grid_w, *, device, dtype):
    ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / grid_h
    xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / grid_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class FrozenMAIUIVisualExtractor:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def __call__(self, screenshot: np.ndarray) -> VisualTokens:
        rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        inputs = self.processor.image_processor(
            images=rgb, return_tensors="pt"
        ).to(self.model.device)

        out = self.model.get_image_features(
            inputs.pixel_values, inputs.image_grid_thw
        )
        features = out.pooler_output[0]

        merge = self.model.config.vision_config.spatial_merge_size
        grid_t, grid_h, grid_w = inputs.image_grid_thw[0].tolist()
        merged_t, merged_h, merged_w = grid_t, grid_h // merge, grid_w // merge
        if merged_t != 1:
            raise ValueError(f"Expected screenshot grid_t=1, got {merged_t}.")
        if features.shape[0] != merged_h * merged_w:
            raise ValueError("Visual-token count does not match merged grid.")

        positions = make_token_positions(
            merged_h, merged_w, device=features.device, dtype=features.dtype
        )
        image_h, image_w = screenshot.shape[:2]
        return VisualTokens(
            features=features.detach().cpu(),
            positions=positions.detach().cpu(),
            image_hw=(image_h, image_w),
        )


class AttentionPool(nn.Module):
    def __init__(self, token_dim: int, pooled_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(pooled_dim) / math.sqrt(pooled_dim))
        self.key = nn.Linear(token_dim, pooled_dim, bias=False)
        self.value = nn.Linear(token_dim, pooled_dim, bias=False)

    def forward(self, tokens, bias=None, valid_mask=None):
        keys, values = self.key(tokens), self.value(tokens)
        logits = torch.einsum("bnd,d->bn", keys, self.query) / math.sqrt(keys.shape[-1])
        if bias is not None:
            logits = logits + bias
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(values.dtype)
        pooled = torch.einsum("bn,bnd->bd", weights, values)
        return pooled, weights


class ElementFocusedAttention(nn.Module):
    """Predicts element scores and uses them only in this branch."""
    def __init__(self, token_dim=2048, pooled_dim=256, position_dim=32,
                 hidden_dim=256, bias_strength=1.0):
        super().__init__()
        self.bias_strength = bias_strength
        self.position_encoder = nn.Sequential(
            nn.Linear(2, position_dim), nn.GELU(), nn.Linear(position_dim, position_dim)
        )
        self.element_score = nn.Sequential(
            nn.LayerNorm(token_dim + position_dim),
            nn.Linear(token_dim + position_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionPool(token_dim, pooled_dim)

    def forward(self, tokens, positions, valid_mask=None):
        pos = self.position_encoder(positions)
        logits = self.element_score(torch.cat([tokens, pos], dim=-1)).squeeze(-1)
        vector, attention = self.pool(
            tokens, bias=self.bias_strength * logits, valid_mask=valid_mask
        )
        return {"vector": vector, "attention": attention, "logits": logits}

class ScreenEncoder(nn.Module):
    def __init__(self, token_dim=2048, pooled_dim=256):
        super().__init__()

        self.global_pool = AttentionPool(
            token_dim,
            pooled_dim,
        )

        self.element_pool = ElementFocusedAttention(
            token_dim,
            pooled_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * pooled_dim, pooled_dim),
            nn.GELU(),
            nn.LayerNorm(pooled_dim),
        )

    def forward(self, tokens, positions, valid_mask=None):
        global_vec, global_attn = self.global_pool(
            tokens,
            valid_mask=valid_mask,
        )

        element = self.element_pool(
            tokens,
            positions,
            valid_mask=valid_mask,
        )

        screen = self.fusion(
            torch.cat(
                [global_vec, element["vector"]],
                dim=-1,
            )
        )

        return {
            "screen": screen,
            "element_logits": element["logits"],
            "global_attention": global_attn,
            "element_attention": element["attention"],
        }

class ActionCoordinateAttentionPool(nn.Module):
    """One fixed-size action vector from previous tokens, type, and tap coordinate."""
    def __init__(self, token_dim=2048, pooled_dim=256, type_dim=32,
                 coord_dim=32, sigma=0.06, bias_strength=1.0):
        super().__init__()
        self.sigma = sigma
        self.bias_strength = bias_strength
        self.pool = AttentionPool(token_dim, pooled_dim)
        self.type_embedding = nn.Embedding(5, type_dim, padding_idx=0)
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, coord_dim), nn.GELU(), nn.Linear(coord_dim, coord_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(pooled_dim + type_dim + coord_dim, pooled_dim),
            nn.GELU(), nn.LayerNorm(pooled_dim)
        )

    def forward(self, tokens, positions, action_type, action_xy, valid_mask=None):
        if torch.any((action_type < 1) | (action_type > 4)):
            raise ValueError("action_type must be in {1,2,3,4}.")

        is_tap = action_type.eq(TAP)
        d2 = (positions - action_xy[:, None, :]).square().sum(dim=-1)
        coord_bias = -d2 / (2.0 * self.sigma ** 2)
        coord_bias = coord_bias * is_tap[:, None].to(coord_bias.dtype)

        visual, attention = self.pool(
            tokens, bias=self.bias_strength * coord_bias, valid_mask=valid_mask
        )
        type_feature = self.type_embedding(action_type)
        coord_feature = self.coord_encoder(action_xy)
        coord_feature = coord_feature * is_tap[:, None].to(coord_feature.dtype)
        action = self.fusion(torch.cat([visual, type_feature, coord_feature], dim=-1))
        return {"action": action, "attention": attention}


class HistoryAttention(nn.Module):
    """Fuses [previous screen, previous action, current screen]."""
    def __init__(self, dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.role_embeddings = nn.Parameter(torch.randn(3, dim) / math.sqrt(dim))
        self.null_previous_screen = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.null_previous_action = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=4 * dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.gate = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, previous_screen, previous_action, current_screen, history_available):
        batch = current_screen.shape[0]
        has_hist = history_available[:, None]
        null_screen = self.null_previous_screen[None].expand(batch, -1)
        null_action = self.null_previous_action[None].expand(batch, -1)

        previous_screen = torch.where(has_hist, previous_screen, null_screen)
        previous_action = torch.where(has_hist, previous_action, null_action)

        sequence = torch.stack([previous_screen, previous_action, current_screen], dim=1)
        encoded = self.transformer(sequence + self.role_embeddings[None])
        context = encoded[:, 2]

        gate = self.gate(torch.cat([
            current_screen,
            context,
            history_available[:, None].to(current_screen.dtype),
        ], dim=-1))
        gate = gate * history_available[:, None].to(gate.dtype)
        return self.norm(current_screen + gate * context)

class UIGraphEmbedder(nn.Module):
    """Trainable network over frozen MAI-UI visual tokens."""

    EMBEDDING_MODULE_NAMES = (
        "token_norm",
        "screen_encoder",
        "action_encoder",
        "history_encoder",
        "embedding_head",
    )

    def __init__(
        self,
        token_dim: int = 2048,
        pooled_dim: int = 256,
        embedding_dim: int = 256,
        text_embedding_dim: int = 768,
    ):
        super().__init__()

        self.token_norm = nn.LayerNorm(token_dim)

        self.screen_encoder = ScreenEncoder(
            token_dim=token_dim,
            pooled_dim=pooled_dim,
        )

        self.action_encoder = ActionCoordinateAttentionPool(
            token_dim=token_dim,
            pooled_dim=pooled_dim,
        )

        self.history_encoder = HistoryAttention(
            dim=pooled_dim,
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.transition_predictor = TransitionPredictor(
            dim=pooled_dim,
            hidden_dim=pooled_dim * 2,
        )

        self.tab_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.subtab_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.description_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.waypoint_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.transition_hint_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.intent_projection = nn.Linear(
            embedding_dim,
            text_embedding_dim,
        )

        self.on_graph_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1),
        )

        self.token_dim = token_dim
        self.pooled_dim = pooled_dim
        self.embedding_dim = embedding_dim
        self.text_embedding_dim = text_embedding_dim

    def embedding_parameter_count(self) -> int:
        return sum(
            count_trainable_parameters(getattr(self, name))
            for name in self.EMBEDDING_MODULE_NAMES
        )

    def embedding_state_dict(self) -> dict:
        state = {}
        for name in self.EMBEDDING_MODULE_NAMES:
            module = getattr(self, name)
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value
        return state

    def load_embedding_state_dict(self, state_dict, strict: bool = True):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [
            key for key in missing
            if key.split(".", 1)[0] in self.EMBEDDING_MODULE_NAMES
        ]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error loading embedding state_dict. "
                f"Missing: {missing}. Unexpected: {unexpected}."
            )
        return missing, unexpected

    def forward_embedding(
        self,
        current_tokens,
        current_positions,
        previous_tokens: Optional[torch.Tensor],
        previous_positions: Optional[torch.Tensor],
        action_type,
        action_xy,
        history_available,
        current_valid_mask=None,
        previous_valid_mask=None,
    ) -> torch.Tensor:
        """Inference path: screenshot tokens + optional prev/action → embedding."""
        current_tokens = self.token_norm(current_tokens)
        current = self.screen_encoder(
            current_tokens,
            current_positions,
            current_valid_mask,
        )

        batch_size, _, token_dim = current_tokens.shape

        if previous_tokens is None or previous_positions is None:
            if torch.any(history_available):
                raise ValueError(
                    "Previous tokens are required for samples with history."
                )
            previous_tokens = torch.zeros(
                batch_size,
                1,
                token_dim,
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )
            previous_positions = torch.zeros(
                batch_size,
                1,
                2,
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )
            previous_valid_mask = torch.ones(
                batch_size,
                1,
                device=current_tokens.device,
                dtype=torch.bool,
            )
        else:
            previous_tokens = self.token_norm(previous_tokens)

        previous = self.screen_encoder(
            previous_tokens,
            previous_positions,
            previous_valid_mask,
        )
        action = self.action_encoder(
            previous_tokens,
            previous_positions,
            action_type,
            action_xy,
            previous_valid_mask,
        )
        history_state = self.history_encoder(
            previous_screen=previous["screen"],
            previous_action=action["action"],
            current_screen=current["screen"],
            history_available=history_available,
        )
        return F.normalize(self.embedding_head(history_state), dim=-1)

    def forward(
        self,
        current_tokens,
        current_positions,
        previous_tokens: Optional[torch.Tensor],
        previous_positions: Optional[torch.Tensor],
        action_type,
        action_xy,
        history_available,
        current_valid_mask=None,
        previous_valid_mask=None,
    ):
        current_tokens = self.token_norm(current_tokens)

        current = self.screen_encoder(
            current_tokens,
            current_positions,
            current_valid_mask,
        )

        batch_size, _, token_dim = current_tokens.shape

        if previous_tokens is None or previous_positions is None:
            if torch.any(history_available):
                raise ValueError(
                    "Previous tokens are required for samples with history."
                )

            previous_tokens = torch.zeros(
                batch_size,
                1,
                token_dim,
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )

            previous_positions = torch.zeros(
                batch_size,
                1,
                2,
                device=current_tokens.device,
                dtype=current_tokens.dtype,
            )

            previous_valid_mask = torch.ones(
                batch_size,
                1,
                device=current_tokens.device,
                dtype=torch.bool,
            )
        else:
            previous_tokens = self.token_norm(previous_tokens)

        previous = self.screen_encoder(
            previous_tokens,
            previous_positions,
            previous_valid_mask,
        )

        action = self.action_encoder(
            previous_tokens,
            previous_positions,
            action_type,
            action_xy,
            previous_valid_mask,
        )

        history_state = self.history_encoder(
            previous_screen=previous["screen"],
            previous_action=action["action"],
            current_screen=current["screen"],
            history_available=history_available,
        )

        embedding = F.normalize(
            self.embedding_head(history_state),
            dim=-1,
        )

        on_graph_logit = self.on_graph_head(
        embedding).squeeze(-1)  # [B]


        predicted_current_screen = self.transition_predictor(
            previous_screen_vector=previous["screen"],
            action_vector=action["action"],
        )

        predicted_tab_embedding = F.normalize(
            self.tab_projection(embedding),
            dim=-1,
        )

        predicted_subtab_embedding = F.normalize(
            self.subtab_projection(embedding),
            dim=-1,
        )

        predicted_description_embedding = F.normalize(
            self.description_projection(embedding),
            dim=-1,
        )

        predicted_waypoint_embedding = F.normalize(
            self.waypoint_projection(embedding),
            dim=-1,
        )

        predicted_transition_hint_embedding = F.normalize(
            self.transition_hint_projection(embedding),
            dim=-1,
        )

        predicted_intent_embedding = F.normalize(
            self.intent_projection(embedding),
            dim=-1,
        )

        return EmbedderOutput(
            embedding=embedding,
            on_graph_logit=on_graph_logit,
            predicted_current_screen=predicted_current_screen,
            predicted_tab_embedding=predicted_tab_embedding,
            predicted_subtab_embedding=predicted_subtab_embedding,
            predicted_description_embedding=predicted_description_embedding,
            predicted_waypoint_embedding=predicted_waypoint_embedding,
            predicted_transition_hint_embedding=
                predicted_transition_hint_embedding,
            predicted_intent_embedding=predicted_intent_embedding,
            current_screen=current["screen"],
            previous_screen=previous["screen"],
            action_vector=action["action"],
            current_element_logits=current["element_logits"],
            previous_element_logits=previous["element_logits"],
            current_global_attention=current["global_attention"],
            current_element_attention=current["element_attention"],
            previous_global_attention=previous["global_attention"],
            previous_element_attention=previous["element_attention"],
            action_attention=action["attention"],
        )

class ScreenshotUIGraphEmbedder(nn.Module):
    """Batch-size-1 convenience wrapper accepting raw OpenCV screenshots."""
    def __init__(self, visual_extractor, graph_embedder):
        super().__init__()
        self.visual_extractor = visual_extractor
        self.graph_embedder = graph_embedder

    def _batch(self, visual, device, dtype):
        return (
            visual.features.to(device=device, dtype=dtype)[None],
            visual.positions.to(device=device, dtype=dtype)[None],
        )

    def forward(
        self,
        current_screenshot: np.ndarray,
        previous_screenshot: Optional[np.ndarray] = None,
        action_type: int = NONE,
        action_xy_pixels: Optional[tuple[float, float]] = None,
    ):
        current_visual = self.visual_extractor(current_screenshot)
        parameter = next(self.graph_embedder.parameters())
        device, dtype = parameter.device, parameter.dtype
        current_tokens, current_positions = self._batch(current_visual, device, dtype)

        has_history = previous_screenshot is not None
        history_available = torch.tensor([has_history], device=device, dtype=torch.bool)
        action_type_tensor = torch.tensor(
            [action_type if has_history else NONE], device=device, dtype=torch.long
        )

        if not has_history:
            previous_tokens = previous_positions = None
            action_xy = torch.zeros(1, 2, device=device, dtype=dtype)
        else:
            previous_visual = self.visual_extractor(previous_screenshot)
            previous_tokens, previous_positions = self._batch(
                previous_visual, device, dtype
            )
            if action_type == TAP:
                if action_xy_pixels is None:
                    raise ValueError("Tap requires action_xy_pixels=(x,y).")
                x, y = action_xy_pixels
                image_h, image_w = previous_visual.image_hw
                action_xy = torch.tensor([[
                    x / max(image_w - 1, 1),
                    y / max(image_h - 1, 1),
                ]], device=device, dtype=dtype).clamp(0, 1)
            else:
                action_xy = torch.zeros(1, 2, device=device, dtype=dtype)

        return self.graph_embedder(
            current_tokens=current_tokens,
            current_positions=current_positions,
            previous_tokens=previous_tokens,
            previous_positions=previous_positions,
            action_type=action_type_tensor,
            action_xy=action_xy,
            history_available=history_available,
        )


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)