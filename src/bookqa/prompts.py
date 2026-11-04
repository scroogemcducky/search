"""Prompt templates.

Each prompt is a text file with a system part and a user part, separated by a
line containing only ``<<<USER>>>``. Placeholders use ``$name`` syntax
(``string.Template``), which leaves the curly braces of JSON examples alone.

The package ships generic defaults in ``bookqa/prompt_templates/``. A project overrides
any of them by putting a file with the same name in its ``prompts/`` folder,
which is where subject-specific wording belongs.
"""

from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path
from string import Template

SEPARATOR = "<<<USER>>>"


@dataclasses.dataclass(frozen=True)
class Prompt:
    name: str
    system: Template
    user: Template

    def render(self, **values: object) -> tuple[str, str]:
        text = {k: str(v) for k, v in values.items()}
        try:
            return self.system.substitute(text), self.user.substitute(text)
        except KeyError as e:
            raise KeyError(f"prompt '{self.name}' needs a value for ${e.args[0]}") from None


def load_prompt(name: str, override_dir: Path | None = None) -> Prompt:
    if override_dir is not None and (override_dir / f"{name}.md").exists():
        raw = (override_dir / f"{name}.md").read_text(encoding="utf-8")
    else:
        raw = (
            resources.files("bookqa.prompt_templates")
            .joinpath(f"{name}.md")
            .read_text(encoding="utf-8")
        )
    if SEPARATOR not in raw:
        raise ValueError(f"prompt '{name}' has no {SEPARATOR} line")
    system, user = raw.split(SEPARATOR, 1)
    return Prompt(name, Template(system.strip()), Template(user.strip()))
