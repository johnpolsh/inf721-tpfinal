import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from torch.jit.mobile import _backport_for_mobile, _get_model_bytecode_version

def convert_for_mobile(model, path, example=torch.rand(1, 3, 224, 224)):
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(path)

    print(f"\nModel original version: {_get_model_bytecode_version(path)}")
    _backport_for_mobile(f_input=path, f_output=path, to_version=5)
    print(f"New model version: {_get_model_bytecode_version(path)}")