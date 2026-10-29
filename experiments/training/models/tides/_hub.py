"""Hugging Face Hub support for the TIDES models.

TIDESClassifier and TIDESForecastingModel inherit PyTorchModelHubMixin, which
records their constructor arguments and adds

    model.save_pretrained(path)                 # config.json + model.safetensors
    model.push_to_hub("user/repo")
    Model.from_pretrained("user/repo" or path)

Without huggingface_hub installed the models work as before; only these three
methods raise.
"""

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # pragma: no cover - exercised only without huggingface_hub
    class PyTorchModelHubMixin:
        _MSG = ("Hugging Face Hub support needs `pip install huggingface_hub safetensors`")

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError(cls._MSG)

        def save_pretrained(self, *args, **kwargs):
            raise ImportError(self._MSG)

        def push_to_hub(self, *args, **kwargs):
            raise ImportError(self._MSG)


def hub_kwargs(*extra_tags: str) -> dict:
    """Model-card metadata shared by the TIDES models."""
    return dict(library_name="tides", license="mit",
                repo_url="https://github.com/TaylanSoydan/TIDES",
                tags=["time-series", "state-space-model", "tides", *extra_tags])
