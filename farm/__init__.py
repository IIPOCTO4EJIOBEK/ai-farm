"""Ферма ИИ: диспетчер, специалисты по доменам, контролёр.

Точка входа — :func:`run`. Всё остальное — детали: пул слотов живёт в
``models.py``, способы обращения к моделям — в ``backends.py``, промпты и
критерии приёмки — в ``roles.py``.
"""

from __future__ import annotations

from .backends import BackendError, Reply
from .ledger import Ledger, Record, Totals, balance, load, summarize
from .models import Slot, build_pool, free_slots, pick
from .orchestrator import Farm, FarmResult, Step, run
from .roles import DOMAINS, GENERAL, Domain, domain_by_key, route

__all__ = [
    "BackendError",
    "Reply",
    "Ledger",
    "Record",
    "Totals",
    "Slot",
    "Step",
    "Farm",
    "FarmResult",
    "Domain",
    "DOMAINS",
    "GENERAL",
    "balance",
    "build_pool",
    "free_slots",
    "load",
    "pick",
    "summarize",
    "domain_by_key",
    "route",
    "run",
]
