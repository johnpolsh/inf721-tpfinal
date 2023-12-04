# Useful scripts for converting the trained model to Pytorch Lite
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from torch.jit.mobile import _backport_for_mobile, _get_model_bytecode_version

def convert_for_mobile(model, name, ext='.ptl', example=torch.rand(1, 3, 224, 224)):
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(f"tmp_{name}{ext}")

    print(f"Model original version: ", _get_model_bytecode_version(f"tmp_{name}{ext}"))
    _backport_for_mobile(f"tmp_{name}{ext}", f"{name}_v5{ext}", 5)
    print(f"Model new version: ", _get_model_bytecode_version(f"{name}_v5{ext}"))