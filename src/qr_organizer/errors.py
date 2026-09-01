"""Exception types.

The house style here is explicit failure: every one of these is raised at the
point an assumption breaks, logged loudly, and surfaced on the status page.
Nothing in this app is allowed to swallow a failure and carry on with a
plausible-looking empty result.
"""

from __future__ import annotations


class QROrganizerError(Exception):
    """Base class for every error this app raises deliberately."""


class ConfigError(QROrganizerError):
    """Config is missing, unparseable, or internally inconsistent."""


class FatalConfigError(ConfigError):
    """Config problem severe enough that there is nothing meaningful to run."""


class VisionError(QROrganizerError):
    """An AI identification pass failed or returned something unusable."""


class VisionResponseError(VisionError):
    """The model replied, but the reply did not match the expected schema."""


class EmbeddingError(QROrganizerError):
    """The embedding backend is unavailable or failed on a specific image."""


class StorageError(QROrganizerError):
    """A database or media-file operation failed."""


class NotFoundError(QROrganizerError):
    """A record referenced by code or id does not exist."""


class ConflictError(QROrganizerError):
    """The requested operation contradicts current state."""


class LocationContextExpired(QROrganizerError):
    """A bin scan arrived with no live location context to attribute it to."""
