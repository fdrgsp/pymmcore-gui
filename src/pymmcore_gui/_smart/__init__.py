"""Engine behind the Smart Microscopy tab (event-driven acquisition).

Everything here except `_controller` is free of Qt; `_worker` must also stay
cheap to import, since it is the first module a spawned analysis process loads.
"""
