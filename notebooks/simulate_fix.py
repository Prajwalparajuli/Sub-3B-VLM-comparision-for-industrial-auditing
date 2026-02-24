
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Simulate bitsandbytes Params4bit
class Params4bit(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, **kwargs):
        if "_is_hf_initialized" in kwargs:
            raise TypeError(f"Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'")
        return super().__new__(cls, data, requires_grad=requires_grad)

# Simulate accelerate's set_module_tensor_to_device bug
def set_module_tensor_to_device_SIMULATED(module, tensor_name, device, **kwargs):
    param = module._parameters[tensor_name]
    param_cls = type(param)
    # This is what accelerate 1.12.0 does:
    kwargs_dict = param.__dict__.copy()
    print(f"DEBUG: accelerate is about to call {param_cls.__name__} with kwargs: {kwargs_dict}")
    new_param = param_cls(torch.randn(1), **kwargs_dict)
    module._parameters[tensor_name] = new_param

# The Fix: Monkeypatch
def patch_accelerate():
    import accelerate.utils.modeling
    original_set_module_tensor_to_device = accelerate.utils.modeling.set_module_tensor_to_device
    
    def patched_set_module_tensor_to_device(*args, **kwargs):
        # We need to reach into the module to clean the parameter's __dict__ before accelerate uses it
        module = args[0]
        tensor_name = args[1]
        if tensor_name in module._parameters:
            param = module._parameters[tensor_name]
            if hasattr(param, "_is_hf_initialized"):
                # Temporarily remove it from __dict__ if it exists
                # Actually, accelerate does: kwargs = module._parameters[tensor_name].__dict__
                # So we can just pop it from the dict if it's there.
                if "_is_hf_initialized" in param.__dict__:
                    print(f"DEBUG: Found _is_hf_initialized in {tensor_name}, cleaning...")
                    param.__dict__.pop("_is_hf_initialized")
        return original_set_module_tensor_to_device(*args, **kwargs)
    
    accelerate.utils.modeling.set_module_tensor_to_device = patched_set_module_tensor_to_device

# Test the simulation
class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Params4bit(torch.randn(1))
        self.weight._is_hf_initialized = True

m = MockModule()
print("--- Testing WITHOUT patch (Should fail) ---")
try:
    set_module_tensor_to_device_SIMULATED(m, "weight", "cuda")
except TypeError as e:
    print(f"Caught expected error: {e}")

print("\n--- Testing WITH patch (Should pass) ---")
# Mocking the actual accelerate module for the test
sys.modules['accelerate'] = MagicMock()
sys.modules['accelerate.utils'] = MagicMock()
sys.modules['accelerate.utils.modeling'] = MagicMock()
import accelerate.utils.modeling
accelerate.utils.modeling.set_module_tensor_to_device = set_module_tensor_to_device_SIMULATED

patch_accelerate()
try:
    accelerate.utils.modeling.set_module_tensor_to_device(m, "weight", "cuda")
    print("Success! Patch worked.")
except Exception as e:
    print(f"Patch failed: {e}")
