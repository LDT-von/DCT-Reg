"""Explicit recipe for the independent DCT v2 survival model."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class DCTV2Config:
    omic_sizes: tuple[int, ...]
    wsi_input_dim: int = 1536
    dim: int = 256
    num_classes: int = 4
    wsi_slots: int = 8
    omics_slots: int = 8
    stages: int = 4
    heads: int = 4
    event_layers: int = 2
    slot_iters: int = 3
    slot_heads: int = 8
    slot_dim_head: int = 64
    slot_hidden_dim: int = 128
    slot_init_mode: str = "deterministic"
    slot_eval_seed: int = 1729
    dropout: float = .15
    geometries: tuple[str, ...] = ("cosine", "euclidean", "positive_dot")
    coordinate_temperature: float = .30
    stage_cost_weight: float = .20
    evidence_cost_weight: float = 0.
    evidence_mass_floor: float = .05
    marginal_strength: float = 1.
    cost_mode: str = "bounded"
    cost_scale: float = 1.
    euclidean_scale: float | None = None
    transport_mode: str = "uot"
    plan_mode: str = "learned"
    tau_wsi: float = .5
    tau_omics: float = .5
    eps_start: float = .10
    eps: float = .05
    eps_anneal_epochs: int = 12
    uot_iters: int = 200
    uot_tolerance: float = 1e-7
    sinkhorn_iters: int = 50
    projection_iters: int = 1000
    projection_tolerance: float = 1e-6
    min_plan_mass: float = 1e-8
    mass_aware: bool = True
    local_gating: bool = True
    residual_init: float = .1
    nll_alpha: float = 0.
    ipcw_weight: float = .10
    slot_weight: float = .05
    diversity_weight: float = .10
    relation_weight: float = .05
    native_weight: float = .05
    relation_warmup_epochs: int = 3
    relation_ramp_epochs: int = 5
    relation_tolerance: float = 0.
    ipcw_memory_size: int = 64
    ipcw_max_weight: float = 10.
    ipcw_rank_margin: float = .02
    ipcw_rank_temperature: float = .5
    variance_min: float = .001
    variance_max: float = .05
    target_pair_dist: float = 1.
    pair_weight: float = .5

    def __post_init__(self):
        sizes = tuple(self.omic_sizes)
        if not sizes or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in sizes):
            raise ValueError("omic_sizes must contain positive pathway input widths")
        object.__setattr__(self, "omic_sizes", sizes)
        positive_ints = ("wsi_input_dim", "dim", "num_classes", "wsi_slots", "omics_slots", "stages",
                        "heads", "event_layers", "slot_iters", "slot_heads", "slot_dim_head",
                        "slot_hidden_dim", "eps_anneal_epochs", "uot_iters", "sinkhorn_iters",
                        "projection_iters", "relation_ramp_epochs")
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("relation_warmup_epochs", "ipcw_memory_size", "slot_eval_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.dim < 2 or self.dim % self.heads:
            raise ValueError("dim must be >=2 and divisible by heads")
        positive = ("coordinate_temperature", "evidence_mass_floor", "cost_scale", "tau_wsi",
                    "tau_omics", "eps_start", "eps", "uot_tolerance", "projection_tolerance",
                    "ipcw_max_weight", "ipcw_rank_temperature")
        nonnegative = ("stage_cost_weight", "evidence_cost_weight", "min_plan_mass", "ipcw_weight",
                       "slot_weight", "diversity_weight", "relation_weight", "native_weight",
                       "relation_tolerance", "ipcw_rank_margin", "variance_min", "variance_max",
                       "target_pair_dist", "pair_weight")
        for name in positive + nonnegative + ("dropout", "marginal_strength", "residual_init", "nll_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + " must be finite numeric")
            if name in positive and value <= 0 or name in nonnegative and value < 0:
                raise ValueError(name + " has an invalid sign")
        if not 0 <= self.dropout < 1 or not 0 <= self.nll_alpha <= 1 or not 0 <= self.marginal_strength <= 1:
            raise ValueError("invalid dropout, NLL alpha or marginal strength")
        if self.variance_min > self.variance_max:
            raise ValueError("variance_min must not exceed variance_max")
        for name in ("mass_aware", "local_gating"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(name + " must be boolean")
        if self.transport_mode not in {"uot", "balanced"} or self.plan_mode not in {"learned", "independent"}:
            raise ValueError("invalid transport or plan mode")
        if self.cost_mode != "bounded":
            raise ValueError("independent DCT v2 uses bounded absolute costs")
        if self.slot_init_mode not in {"deterministic", "learned", "gaussian"}:
            raise ValueError("invalid slot initialization")
        geometries = tuple(self.geometries)
        if not geometries or len(set(geometries)) != len(geometries) or not set(geometries) <= {
                "cosine", "euclidean", "positive_dot"}:
            raise ValueError("invalid transport geometries")
        object.__setattr__(self, "geometries", geometries)
        scale = math.sqrt(self.dim) if self.euclidean_scale is None else self.euclidean_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("euclidean_scale must be finite positive")
        object.__setattr__(self, "euclidean_scale", float(scale))

    @property
    def projection_tol(self):
        return self.projection_tolerance

    def relation_effective_weight(self, epoch):
        return self.relation_weight * min(1., max(0., (epoch-self.relation_warmup_epochs+1)/self.relation_ramp_epochs))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("DCT v2 configuration must be an object")
        return cls(**values)

    @classmethod
    def from_args(cls, args):
        if getattr(args, "rna_format", "Pathways") != "Pathways" or getattr(args, "bag_loss", "nll_surv") != "nll_surv":
            raise ValueError("DCT v2 requires Pathways and nll_surv")
        ordinary = {"wsi_input_dim": "encoding_dim", "dim": "wsi_projection_dim", "num_classes": "n_classes",
                    "wsi_slots": "slot_num_wsi", "omics_slots": "slot_num_omics", "stages": "dct_num_stages",
                    "heads": "otehv2_heads", "event_layers": "otehv2_layers", "slot_iters": "slot_iters",
                    "slot_init_mode": "dct_slot_init_mode", "slot_eval_seed": "dct_slot_eval_seed",
                    "dropout": "otehv2_dropout", "coordinate_temperature": "dct_coordinate_temperature",
                    "stage_cost_weight": "spt_prog_cost", "eps_start": "rg_eps_start", "eps": "otehv2_eps",
                    "eps_anneal_epochs": "rg_eps_anneal", "nll_alpha": "alpha_surv",
                    "evidence_cost_weight": "dct_evidence_cost_weight", "evidence_mass_floor": "dct_evidence_mass_floor",
                    "marginal_strength": "dct_evidence_marginal_strength", "sinkhorn_iters": "otehv2_iter"}
        own = {"ipcw_weight": "lambda_ipcw_rank", "slot_weight": "lambda_slot_nll", "diversity_weight": "lambda_diversity",
               "relation_weight": "lambda_relation", "native_weight": "lambda_native"}
        values = {"omic_sizes": tuple(args.omic_sizes)}
        for field in fields(cls):
            name = field.name
            key = ordinary.get(name, "dct_v2_" + own.get(name, name))
            value = getattr(args, key, None)
            if value is not None:
                values[name] = value
        return cls(**values)
