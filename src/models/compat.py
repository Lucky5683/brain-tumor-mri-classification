"""Compatibility shim for Keras 3.x deserialization.

Resolves compatibility issues when deserializing models containing initializer
configurations that include legacy or newer arguments (such as 'input_axes' and 'output_axes')
not accepted by Keras 3.14+ initializers.
"""

from __future__ import annotations
import inspect


def apply_keras_compatibility_shim() -> None:
    """Patch Keras initializers so that unexpected arguments like input_axes do not break loading."""
    try:
        import keras.src.initializers.random_initializers as ri

        for name in dir(ri):
            cls = getattr(ri, name)
            if isinstance(cls, type) and hasattr(cls, "__init__"):
                orig_init = cls.__init__
                # Only wrap if not already wrapped
                if not getattr(orig_init, "_is_shim_wrapped", False):
                    sig = inspect.signature(orig_init)

                    def make_patched(original_fn, original_sig):
                        def patched(self, *args, **kwargs):
                            if "input_axes" not in original_sig.parameters:
                                kwargs.pop("input_axes", None)
                            if "output_axes" not in original_sig.parameters:
                                kwargs.pop("output_axes", None)
                            return original_fn(self, *args, **kwargs)

                        patched._is_shim_wrapped = True
                        return patched

                    cls.__init__ = make_patched(orig_init, sig)
    except Exception:
        pass


# Apply automatically upon module import
apply_keras_compatibility_shim()
