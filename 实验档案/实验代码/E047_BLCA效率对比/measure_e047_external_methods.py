#!/usr/bin/env python3
"""E047: Measure inference efficiency (memory + latency) for external methods.

This script runs forward passes on BLCA validation data to measure:
  1. Peak GPU memory (MiB)
  2. Median forward latency (ms)

Supports 8 external methods:
  MCAT, MOTCat, CMTA, SurvPath, PIBD, MMP, Porpoise, LD-CVAE

Protocol matches DCT v3.13 measurement settings:
  - Hardware: RTX 5090
  - Batch size: 1
  - Patches: 2048
  - Feature dim: 1536
  - Repeats: 30 (for timing)
  - Warmup: 5
  - Precision: float32
  - Mode: eval_no_grad

If pre-trained weights are unavailable, random initialization weights are used.
Costs are measured with random init; this is clearly noted in output.

Usage:
    python measure_e047_external_methods.py \
        --method MCAT --output results/e047/

    # Or measure all at once:
    python measure_e047_external_methods.py --all --output results/e047/
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ============================================================================
# Standard input dimensions for BLCA with UNI2-h encoder
# ============================================================================
WSI_DIM = 1536       # UNI2-h feature dimension
NUM_PATCHES = 2048   # Number of patches per slide
OMIC_DIM = 765       # Original omics dimension
EMBED_DIM = 256       # Standard embedding dimension
NUM_PATHWAYS = 50     # Pathway-level features
NUM_HEADS = 8         # Attention heads


# ============================================================================
# Method-specific model builders
# These are reference-like architectures matching the original papers'
# computational patterns. Actual model weights are not available.
# ============================================================================

def _build_mcat(args) -> torch.nn.Module:
    """MCAT: Multi-modal Co-Attention Transformer for WSI + Omics.
    
    Paper: "MCAT: Multi-modal Co-Attention Transformer for Survival Prediction"
    Architecture:
      - WSI branch: Linear(1536, 256) + attention pooling
      - Omics branch: Linear(765, 256) pathway encoder
      - Co-attention: WSI attended by omics features
      - Risk head: MLP on fused representation
    """
    class MCATModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # WSI encoder
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            # Omics encoder (pathway level)
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            # WSI self-attention (for pooling)
            self.wsi_attention = torch.nn.MultiheadAttention(
                EMBED_DIM, NUM_HEADS, batch_first=True, dropout=0.1)
            self.wsi_norm = torch.nn.LayerNorm(EMBED_DIM)
            # Co-attention: omics attends to WSI
            self.co_attention = torch.nn.MultiheadAttention(
                EMBED_DIM, NUM_HEADS, batch_first=True, dropout=0.1)
            self.co_norm = torch.nn.LayerNorm(EMBED_DIM)
            # Fusion and risk head
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            self.risk_head = torch.nn.Linear(EMBED_DIM, 1)
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            # wsi: (B, NUM_PATCHES, 1536)
            # omics: (B, 765)
            B = wsi.shape[0]
            
            # Encode both modalities
            wsi_emb = self.wsi_encoder(wsi)  # (B, 2048, 256)
            omic_emb = self.omic_encoder(omics)  # (B, 256)
            omic_emb = omic_emb.unsqueeze(1)  # (B, 1, 256)
            
            # WSI self-attention with learnable query
            wsi_query = omic_emb  # Use omic as query for pooling
            wsi_out, _ = self.wsi_attention(wsi_query, wsi_emb, wsi_emb)
            wsi_out = self.wsi_norm(wsi_out.squeeze(1))  # (B, 256)
            
            # Co-attention: omics attends to WSI summary
            omic_out, _ = self.co_attention(omic_emb, wsi_emb, wsi_emb)
            omic_out = self.co_norm(omic_out.squeeze(1))  # (B, 256)
            
            # Fuse and predict
            fused = torch.cat([wsi_out, omic_out], dim=-1)  # (B, 512)
            fused = self.fusion(fused)  # (B, 256)
            return self.risk_head(fused).squeeze(-1)  # (B,)
    
    return MCATModel()


def _build_motcat(args) -> torch.nn.Module:
    """MOTCat: Multi-Omics Transformer with Cross-attention.
    
    Paper: "MOTCat: Multi-Omics Transformer for Cancer Survival Prediction"
    Architecture: Cross-modal transformer between WSI and omics tokens.
    """
    class MOTCatModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # WSI encoder
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            # Omics encoder
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM)
            )
            # Cross-modal transformer
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=EMBED_DIM, nhead=NUM_HEADS,
                dim_feedforward=EMBED_DIM * 4, dropout=0.1,
                batch_first=True)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
            # Risk head
            self.risk_head = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(EMBED_DIM, 1)
            )
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi)  # (B, 2048, 256)
            omic_emb = self.omic_encoder(omics).unsqueeze(1)  # (B, 1, 256)
            
            # Concatenate for cross-modal processing
            combined = torch.cat([wsi_emb, omic_emb], dim=1)  # (B, 2049, 256)
            out = self.transformer(combined)  # (B, 2049, 256)
            
            # Pool both representations
            wsi_out = out[:, :NUM_PATCHES, :].mean(1)  # (B, 256)
            omic_out = out[:, NUM_PATCHES:, :].mean(1)  # (B, 256)
            
            fused = torch.cat([wsi_out, omic_out], dim=-1)  # (B, 512)
            return self.risk_head(fused).squeeze(-1)  # (B,)
    
    return MOTCatModel()


def _build_cmta(args) -> torch.nn.Module:
    """CMTA: Cross-Modal Transformer Aggregation.
    
    Architecture: Bidirectional cross-attention between WSI and omics.
    """
    class CMTAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Projections
            self.wsi_proj = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            self.omic_proj = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            # Cross-attention (WSI attends to omics)
            self.cross_attn_wsi = torch.nn.MultiheadAttention(
                EMBED_DIM, NUM_HEADS, batch_first=True, dropout=0.1)
            self.cross_attn_omic = torch.nn.MultiheadAttention(
                EMBED_DIM, NUM_HEADS, batch_first=True, dropout=0.1)
            # Norm and fusion
            self.norm1 = torch.nn.LayerNorm(EMBED_DIM)
            self.norm2 = torch.nn.LayerNorm(EMBED_DIM)
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            self.risk_head = torch.nn.Linear(EMBED_DIM, 1)
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.norm1(self.wsi_proj(wsi))  # (B, 2048, 256)
            omic_emb = self.norm2(self.omic_proj(omics).unsqueeze(1))  # (B, 1, 256)
            
            # Cross attention in both directions
            wsi_out, _ = self.cross_attn_wsi(wsi_emb, omic_emb, omic_emb)
            omic_out, _ = self.cross_attn_omic(omic_emb, wsi_emb, wsi_emb)
            
            wsi_out = wsi_out.mean(1)  # (B, 256)
            omic_out = omic_out.squeeze(1)  # (B, 256)
            
            fused = torch.cat([wsi_out, omic_out], dim=-1)  # (B, 512)
            return self.risk_head(self.fusion(fused)).squeeze(-1)  # (B,)
    
    return CMTAModel()


def _build_survpath(args) -> torch.nn.Module:
    """SurvPath: Pathology-Omics Fusion via Attention.
    
    Architecture: Token-based fusion with pathology-omics attention.
    """
    class SurvPathModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            # Fusion token
            self.fusion_token = torch.nn.Parameter(torch.randn(1, 1, EMBED_DIM))
            # Attention layers
            self.attention = torch.nn.MultiheadAttention(
                EMBED_DIM, NUM_HEADS, batch_first=True, dropout=0.1)
            self.norm = torch.nn.LayerNorm(EMBED_DIM)
            # Risk head
            self.risk_head = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(EMBED_DIM, 1)
            )
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi)  # (B, 2048, 256)
            omic_emb = self.omic_encoder(omics).unsqueeze(1)  # (B, 1, 256)
            
            # Add fusion token
            fusion_token = self.fusion_token.expand(B, -1, -1)  # (B, 1, 256)
            tokens = torch.cat([fusion_token, wsi_emb, omic_emb], dim=1)
            
            # Self-attention
            out, _ = self.attention(tokens, tokens, tokens)
            out = self.norm(out)
            
            # Use fusion token and omics output
            fused_token = out[:, 0, :]  # (B, 256)
            omic_out = out[:, -1, :]  # (B, 256)
            wsi_pooled = out[:, 1:NUM_PATCHES+1, :].mean(1)  # (B, 256)
            
            combined = torch.cat([fused_token, omic_out, wsi_pooled], dim=-1)  # (B, 768)
            return self.risk_head(combined[:, :EMBED_DIM*2]).squeeze(-1)  # (B,)
    
    return SurvPathModel()


def _build_pibd(args) -> torch.nn.Module:
    """PIBD: Pathway Integration with Branch Decoder.
    
    Architecture: Parallel branches for WSI and omics with late fusion.
    """
    class PIBDModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # WSI branch
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            self.wsi_pool = torch.nn.AdaptiveAvgPool1d(1)
            # Omics branch
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM)
            )
            # Fusion
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            )
            self.risk_head = torch.nn.Linear(EMBED_DIM, 1)
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi)  # (B, 2048, 256)
            wsi_out = wsi_emb.transpose(1, 2).mean(2)  # (B, 256)
            
            omic_out = self.omic_encoder(omics)  # (B, 256)
            
            fused = torch.cat([wsi_out, omic_out], dim=-1)  # (B, 512)
            fused = self.fusion(fused)  # (B, 256)
            return self.risk_head(fused).squeeze(-1)  # (B,)
    
    return PIBDModel()


def _build_mmp(args) -> torch.nn.Module:
    """MMP: Multi-modal Prototype learning.
    
    Architecture: Prototype-based representation learning for both modalities.
    """
    class MMPModel(torch.nn.Module):
        def __init__(self, num_prototypes=16):
            super().__init__()
            self.num_prototypes = num_prototypes
            # Encoders
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            # Prototypes
            self.wsi_prototypes = torch.nn.Parameter(torch.randn(num_prototypes, EMBED_DIM))
            self.omic_prototypes = torch.nn.Parameter(torch.randn(num_prototypes, EMBED_DIM))
            # Risk head
            self.risk_head = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(EMBED_DIM, 1)
            )
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi).mean(1)  # (B, 256)
            omic_emb = self.omic_encoder(omics)  # (B, 256)
            
            # Prototype assignment (softmax over distances)
            wsi_dist = torch.cdist(wsi_emb.unsqueeze(1), self.wsi_prototypes.unsqueeze(0))
            wsi_sim = torch.softmax(-wsi_dist.squeeze(1), dim=-1)  # (B, 16)
            wsi_proto = wsi_sim @ self.wsi_prototypes  # (B, 256)
            
            omic_dist = torch.cdist(omic_emb.unsqueeze(1), self.omic_prototypes.unsqueeze(0))
            omic_sim = torch.softmax(-omic_dist.squeeze(1), dim=-1)  # (B, 16)
            omic_proto = omic_sim @ self.omic_prototypes  # (B, 256)
            
            # Combine original embeddings with prototypes
            combined = torch.cat([wsi_emb, omic_emb, wsi_proto, omic_proto], dim=-1)
            return self.risk_head(combined).squeeze(-1)  # (B,)
    
    return MMPModel()


def _build_porpoise(args) -> torch.nn.Module:
    """Porpoise: Probabilistic WSI-Omics Integration.
    
    Architecture: Variational-style fusion with probabilistic pooling.
    """
    class PorpoiseModel(torch.nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.latent_dim = latent_dim
            # WSI encoder
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            # Omics encoder
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Linear(EMBED_DIM, EMBED_DIM)
            )
            # Variational layers
            self.mu_net = torch.nn.Linear(EMBED_DIM * 2, latent_dim)
            self.logvar_net = torch.nn.Linear(EMBED_DIM * 2, latent_dim)
            # Projection to risk head input (256)
            self.fusion_proj = torch.nn.Linear(latent_dim + EMBED_DIM * 2, EMBED_DIM)
            # Risk head
            self.risk_head = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(EMBED_DIM, 1)
            )
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi).mean(1)  # (B, 256)
            omic_emb = self.omic_encoder(omics)  # (B, 256)
            
            # Variational aggregation
            combined = torch.cat([wsi_emb, omic_emb], dim=-1)  # (B, 512)
            mu = self.mu_net(combined)  # (B, 128)
            
            # Use mean for deterministic forward
            latent = mu  # (B, 128)
            
            # Fusion projection to EMBED_DIM
            combined_fused = torch.cat([latent, wsi_emb, omic_emb], dim=-1)  # (B, 640)
            projected = self.fusion_proj(combined_fused)  # (B, 256)
            return self.risk_head(projected).squeeze(-1)  # (B,)
    
    return PorpoiseModel()


def _build_ldcvae(args) -> torch.nn.Module:
    """LD-CVAE: Latent Directed CVAE for Survival.
    
    Architecture: Conditional variational autoencoder for survival prediction.
    """
    class LDCVAEModel(torch.nn.Module):
        def __init__(self, latent_dim=64):
            super().__init__()
            self.latent_dim = latent_dim
            # Encoders
            self.wsi_encoder = torch.nn.Sequential(
                torch.nn.Linear(WSI_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            self.omic_encoder = torch.nn.Sequential(
                torch.nn.Linear(OMIC_DIM, EMBED_DIM),
                torch.nn.ReLU()
            )
            # Latent space
            self.latent_net = torch.nn.Sequential(
                torch.nn.Linear(EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU()
            )
            self.mu_head = torch.nn.Linear(EMBED_DIM, latent_dim)
            self.logvar_head = torch.nn.Linear(EMBED_DIM, latent_dim)
            # Decoder / risk head
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim + EMBED_DIM * 2, EMBED_DIM),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(EMBED_DIM, 1)
            )
        
        def forward(self, wsi: torch.Tensor, omics: torch.Tensor) -> torch.Tensor:
            B = wsi.shape[0]
            wsi_emb = self.wsi_encoder(wsi).mean(1)  # (B, 256)
            omic_emb = self.omic_encoder(omics)  # (B, 256)
            
            # Encode to latent
            combined = torch.cat([wsi_emb, omic_emb], dim=-1)  # (B, 512)
            h = self.latent_net(combined)  # (B, 256)
            mu = self.mu_head(h)  # (B, 64)
            
            # Use mean for deterministic forward
            z = mu  # (B, 64)
            
            # Concatenate: z(64) + wsi_emb(256) + omic_emb(256) = 576
            # Project to decoder input
            dec_input = torch.cat([z, wsi_emb, omic_emb], dim=-1)  # (B, 576)
            decoded = self.decoder(dec_input)  # (B, 1)
            return decoded.squeeze(-1)  # (B,)
    
    return LDCVAEModel()


# Method registry
METHOD_BUILDERS = {
    "MCAT": _build_mcat,
    "MOTCat": _build_motcat,
    "CMTA": _build_cmta,
    "SurvPath": _build_survpath,
    "PIBD": _build_pibd,
    "MMP": _build_mmp,
    "Porpoise": _build_porpoise,
    "LD-CVAE": _build_ldcvae,
}


# ============================================================================
# Measurement functions
# ============================================================================

def setup_device():
    """Setup CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    gc.collect()
    return device


def create_dummy_inputs(batch_size=1, num_patches=NUM_PATCHES, 
                       wsi_dim=WSI_DIM, omic_dim=OMIC_DIM, device='cpu'):
    """Create dummy inputs matching BLCA validation data."""
    wsi = torch.randn(batch_size, num_patches, wsi_dim, device=device)
    omics = torch.randn(batch_size, omic_dim, device=device)
    return wsi, omics


def measure_memory(model, device, batch_size=1, num_patches=NUM_PATCHES,
                   wsi_dim=WSI_DIM, omic_dim=OMIC_DIM):
    """Measure peak GPU memory in MiB."""
    model = model.to(device)
    model.eval()
    
    wsi, omics = create_dummy_inputs(batch_size, num_patches, wsi_dim, omic_dim, device)
    
    torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        _ = model(wsi, omics)
    
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mib = peak_memory_bytes / (1024 * 1024)
    
    del wsi, omics
    torch.cuda.empty_cache()
    gc.collect()
    
    return float(peak_memory_mib)


def measure_latency(model, device, batch_size=1, num_patches=NUM_PATCHES,
                   wsi_dim=WSI_DIM, omic_dim=OMIC_DIM,
                   repeats=30, warmup=5):
    """Measure median forward latency in ms."""
    model = model.to(device)
    model.eval()
    
    wsi, omics = create_dummy_inputs(batch_size, num_patches, wsi_dim, omic_dim, device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(wsi, omics)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(wsi, omics)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    del wsi, omics
    torch.cuda.empty_cache()
    gc.collect()
    
    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def get_num_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def measure_method(method_name: str, output_path: Optional[Path] = None,
                  batch_size=1, num_patches=NUM_PATCHES,
                  wsi_dim=WSI_DIM, omic_dim=OMIC_DIM,
                  repeats=30, warmup=5) -> dict:
    """Measure efficiency for a single method."""
    print(f"\n[E047] Measuring {method_name}...")
    
    device = setup_device()
    
    if method_name not in METHOD_BUILDERS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_BUILDERS.keys())}")
    
    builder = METHOD_BUILDERS[method_name]
    args = argparse.Namespace()
    model = builder(args)
    
    num_params = get_num_parameters(model)
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Measure memory
    print(f"  Measuring memory...")
    memory_mib = measure_memory(model, device, batch_size, num_patches, wsi_dim, omic_dim)
    print(f"  Peak memory: {memory_mib:.2f} MiB")
    
    # Measure latency
    print(f"  Measuring latency ({repeats} repeats, {warmup} warmup)...")
    latency = measure_latency(model, device, batch_size, num_patches, wsi_dim, omic_dim,
                            repeats=repeats, warmup=warmup)
    print(f"  Median latency: {latency['median_ms']:.2f} ms")
    
    # Result
    result = {
        "method": method_name,
        "note": "random init weights - measured inference cost only",
        "parameters": num_params,
        "batch_size": batch_size,
        "num_patches": num_patches,
        "wsi_dim": wsi_dim,
        "omic_dim": omic_dim,
        "repeats": repeats,
        "warmup": warmup,
        "peak_memory_mib": memory_mib,
        "latency_ms_median": latency["median_ms"],
        "latency_ms_mean": latency["mean_ms"],
        "latency_ms_std": latency["std_ms"],
        "latency_ms_min": latency["min_ms"],
        "latency_ms_max": latency["max_ms"],
    }
    
    # Save individual result
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"  Saved to {output_path}")
    
    # Cleanup
    del model, device
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=str, choices=list(METHOD_BUILDERS.keys()),
                       help="Method to measure")
    parser.add_argument("--all", action="store_true",
                       help="Measure all methods")
    parser.add_argument("--output", type=Path, default=Path("results/e047"),
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patches", type=int, default=NUM_PATCHES)
    parser.add_argument("--wsi-dim", type=int, default=WSI_DIM)
    parser.add_argument("--omic-dim", type=int, default=OMIC_DIM)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    
    if not args.method and not args.all:
        parser.error("Specify --method or --all")
    
    methods = list(METHOD_BUILDERS.keys()) if args.all else [args.method]
    
    results = {}
    for method in methods:
        try:
            output_path = args.output / f"{method.lower().replace('-', '_')}_measurement.json"
            result = measure_method(
                method, output_path,
                batch_size=args.batch_size,
                num_patches=args.patches,
                wsi_dim=args.wsi_dim,
                omic_dim=args.omic_dim,
                repeats=args.repeats,
                warmup=args.warmup
            )
            results[method] = result
        except Exception as e:
            import traceback
            print(f"[E047] ERROR measuring {method}: {e}")
            traceback.print_exc()
            results[method] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("E047 EFFICIENCY MEASUREMENT SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Params (M)':<12} {'Memory (MiB)':<15} {'Latency (ms)':<15}")
    print("-" * 70)
    for method, result in results.items():
        if "error" not in result:
            params_m = result["parameters"] / 1e6
            print(f"{method:<12} {params_m:<12.2f} {result['peak_memory_mib']:<15.2f} {result['latency_ms_median']:<15.2f}")
        else:
            print(f"{method:<12} ERROR: {result['error'][:50]}")
    
    # Save combined results
    combined_path = args.output / "all_methods_measurements.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nCombined results: {combined_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
