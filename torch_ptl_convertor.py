import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

torchscript_model = "l_version_1_300.pt"
export_model_name = "yolo8livensslite.ptl"

model = torch.jit.load(torchscript_model)
optimized_model = optimize_for_mobile(model)
optimized_model._save_for_lite_interpreter(export_model_name)

print(f"mobile optimized model exported to {export_model_name}")
