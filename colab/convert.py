import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

def convert_for_mobile(model, path, example=torch.rand(1, 3, 224, 224)):
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(path)