"""Provider-neutral failures for cocktail taste generation."""


class TasteGenerationError(RuntimeError):
    """Base class for taste-generation failures."""


class TasteGenerationFatalError(TasteGenerationError):
    """Credentials, permissions, model, or request configuration is invalid."""


class TasteGenerationQuotaError(TasteGenerationError):
    """The provider exhausted its current request or credit quota."""


class TasteGenerationItemError(TasteGenerationError):
    """One cocktail could not produce a valid taste profile."""
