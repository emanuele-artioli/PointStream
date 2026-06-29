import torch
import numpy as np

def apply_compat_patches():
    """Apply global monkey patches required for compatibility with certain libraries."""
    _patch_torch_dlpack()
    _patch_ultralytics_numpy()

def _patch_torch_dlpack():
    """Patch torch.from_numpy to handle DLPack objects if numpy array conversion fails."""
    if hasattr(torch, "_original_from_numpy"):
        return
        
    _original_from_numpy = torch.from_numpy
    def _patched_from_numpy(x):
        try:
            return _original_from_numpy(x)
        except TypeError:
            import torch.utils.dlpack
            return torch.utils.dlpack.from_dlpack(x.__dlpack__())
            
    torch.from_numpy = _patched_from_numpy
    torch._original_from_numpy = _original_from_numpy

def _patch_ultralytics_numpy():
    """Patch ultralytics BaseTensor and Boxes to support .numpy() on lists."""
    try:
        from ultralytics.engine.results import BaseTensor
        
        if hasattr(BaseTensor, "_old_numpy"):
            return
            
        _old_numpy = BaseTensor.numpy
        def _new_numpy(self):
            if type(self.data).__name__ == "ndarray" or isinstance(self.data, np.ndarray):
                return self
            arr = np.array(self.data.tolist(), dtype=np.float32)
            return self.__class__(arr, self.orig_shape)
            
        BaseTensor.numpy = _new_numpy
        BaseTensor._old_numpy = _old_numpy
        
        from ultralytics.engine.results import Boxes
        _old_boxes_numpy = Boxes.numpy
        def _new_boxes_numpy(self):
            if type(self.data).__name__ == "ndarray" or isinstance(self.data, np.ndarray):
                return self
            arr = np.array(self.data.tolist(), dtype=np.float32)
            return self.__class__(arr, self.orig_shape)
            
        Boxes.numpy = _new_boxes_numpy
        Boxes._old_boxes_numpy = _old_boxes_numpy
    except ImportError:
        pass
