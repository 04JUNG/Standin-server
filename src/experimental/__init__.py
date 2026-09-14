"""Opt-in experiments that must not become implicit production dependencies.

Modules in this package are loaded lazily behind explicit feature flags.  They
must fail open to the established production behavior and must not write to the
production pose or semantic data directories.
"""
