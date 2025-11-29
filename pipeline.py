"""
Compatibility shim: expose `RSPipeline` under the `pipeline` module
so scripts that import `pipeline.RSPipeline` continue to work.
"""
from rs_pipeline import RSPipeline

__all__ = ["RSPipeline"]
