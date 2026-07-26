"""Regression tests for optional Langfuse OpenAI instrumentation."""

import builtins

from llmware.models import _load_openai_client


def test_openai_client_falls_back_when_langfuse_import_fails(monkeypatch):
    real_import = builtins.__import__

    def fail_langfuse_import(name, *args, **kwargs):
        if name == "langfuse.openai":
            raise ModuleNotFoundError("incompatible langfuse/openai packages")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_langfuse_import)

    client_class = _load_openai_client()

    assert client_class.__module__.startswith("openai")
