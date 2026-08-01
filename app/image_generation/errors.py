"""Provider-neutral failures for the cocktail image-generation pipeline."""


class GenerationRequestError(RuntimeError):
    """A provider request failed after bounded retries."""


class GenerationFatalError(GenerationRequestError):
    """Credentials, permissions, model, or request configuration is invalid."""


class GenerationQuotaError(GenerationRequestError):
    """A provider exhausted its current request or credit quota."""


class GenerationItemError(GenerationRequestError):
    """One cocktail could not produce a valid provider response."""
