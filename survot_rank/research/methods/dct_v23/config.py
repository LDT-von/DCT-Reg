"""Version-owned controls for genuine masked gene reconstruction."""
from dataclasses import dataclass
import math

from survot_rank.research.methods.dct_v22.config import DCTV22Config


@dataclass(frozen=True)
class DCTV23Config(DCTV22Config):
    ipcw_weight: float = 0.
    slot_weight: float = 0.
    relation_weight: float = 0.
    native_weight: float = 0.
    local_readout: str = "none"
    prediction_path: str = "combined"
    reconstruction_mode: str = "masked_plain"
    mask_ratio: float = .30
    reconstruction_weight: float = .05
    calibration_weight: float = .02
    recon_hidden_dim: int = 64
    calibration_warmup_epochs: int = 3
    gain_scale: float = .10

    def __post_init__(self):
        super().__post_init__()
        if self.prediction_path not in {"combined", "native_only", "transport_only"}:
            raise ValueError("invalid v2.3 prediction_path")
        if self.prediction_path != "combined" and (
                self.reconstruction_mode != "off" or self.relation_weight != 0 or self.native_weight != 0):
            raise ValueError("path attribution requires reconstruction off and relation/native weights zero")
        if self.reconstruction_mode not in {"off", "masked_plain", "gain_calibrated"}:
            raise ValueError("invalid v2.3 reconstruction_mode")
        for name in ("mask_ratio", "reconstruction_weight", "calibration_weight", "gain_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(name + " must be finite numeric")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0,1)")
        if min(self.reconstruction_weight, self.calibration_weight) < 0 or self.gain_scale <= 0:
            raise ValueError("invalid v2.3 objective weight or gain_scale")
        if isinstance(self.recon_hidden_dim, bool) or not isinstance(self.recon_hidden_dim, int) or self.recon_hidden_dim < 1:
            raise ValueError("recon_hidden_dim must be a positive integer")
        if isinstance(self.calibration_warmup_epochs, bool) or not isinstance(self.calibration_warmup_epochs, int) or self.calibration_warmup_epochs < 0:
            raise ValueError("calibration_warmup_epochs must be a nonnegative integer")
        if self.reconstruction_mode == "gain_calibrated" and (self.transport_mode != "uot" or self.plan_mode != "learned" or self.stage_cost_weight <= 0):
            raise ValueError("gain calibration requires learned UOT with active absolute stage costs")

    @classmethod
    def from_args(cls, args):
        values = DCTV22Config.from_args(args).to_dict()
        for name, key in (("ipcw_weight", "dct_v2_lambda_ipcw_rank"), ("slot_weight", "dct_v2_lambda_slot_nll"),
                          ("relation_weight", "dct_v2_lambda_relation"), ("native_weight", "dct_v2_lambda_native")):
            if getattr(args, key, None) is None:
                values[name] = 0.
        if getattr(args, "dct_v21_local_readout", None) is None:
            values["local_readout"] = "none"
        keys = {"reconstruction_weight": "lambda_reconstruction", "calibration_weight": "lambda_calibration"}
        for name in ("prediction_path", "reconstruction_mode", "mask_ratio", "reconstruction_weight", "calibration_weight",
                     "recon_hidden_dim", "calibration_warmup_epochs", "gain_scale"):
            value = getattr(args, "dct_v23_" + keys.get(name, name), None)
            if value is not None:
                values[name] = value
        return cls(**values)
