from pydantic import BaseModel, field_validator


# Valid pipeline step names — keep in sync with ALL_STEPS in orchestrator.py
VALID_STEPS = {"summarize", "audio_full", "audio_chapters", "cover", "alt_text", "mindmap"}


class SumReq(BaseModel):
    book_id:  str
    length:   str = "5min"       # 3min | 5min | 10min | 15min
    style:    str = "narrative"  # narrative | bullets | academic
    language: str = "en"         # en | ar


class Chapter(BaseModel):
    index: int
    title: str
    text:  str


class PipelineOptions(BaseModel):
    length: str = "10min"       # 3min | 5min | 10min | 15min
    style:  str = "narrative"   # narrative | bullets | academic


class PipelineReq(BaseModel):
    book_id:     str
    title:       str | None = None      # optional when looking up catalog book
    author:      str | None = None
    language:    str = "en"             # en | ar
    year:        int | None = None
    pages:       int | None = None
    grade_level: str | None = None
    genres:      list[str] = []
    chapters:    list[Chapter] = []     # empty = look up book_id from chunks table
    summary:     str | None = None      # pre-computed summary — skips Pass 1 & 2 if provided
    steps:       list[str] = []         # empty = run ALL steps; otherwise only the listed ones
    options:     PipelineOptions = PipelineOptions()
    source:      str = "custom_json"    # catalog | custom_json | pdf_upload

    @field_validator("steps")
    @classmethod
    def _check_steps(cls, v: list[str]) -> list[str]:
        """Reject unknown step names with a clear 422 error instead of silently ignoring them."""
        if not v:
            return v
        unknown = [s for s in v if s not in VALID_STEPS]
        if unknown:
            raise ValueError(
                f"Unknown pipeline step(s): {unknown}. "
                f"Valid steps are: {sorted(VALID_STEPS)}. "
                f"Send an empty list (or omit the field) to run all steps."
            )
        return v
