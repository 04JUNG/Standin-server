"""Rough body observation and catalog selection; independent of pose ranking."""
from .catalog import load_catalog
from .service import BodyMatchingService

__all__ = ["BodyMatchingService", "load_catalog"]
