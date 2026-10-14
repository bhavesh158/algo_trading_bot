"""Candidate strategies for validation (Phase 3 build). ISOLATED: new files only.

Each subclasses the real BaseStrategy and reads only truncated replay data.
Session time is taken from bar timestamps (df.index[-1]), never wall-clock,
so these are exact in replay AND safe live.
"""
