"""A rule-based stand-in for a chat model, used by tests and ``--offline``.

It recognises each default prompt by a phrase in it and answers in the
expected format using simple text rules. Its answers are crude but valid, so
the whole pipeline (and the CLI) can run end to end with no model installed.
It's for checking the plumbing, not for reading the output.

Custom prompts in a project's ``prompts/`` folder may not contain the phrases
it looks for, in which case it raises an error showing the start of the prompt.
"""

import json
import re

from bookqa.llm import ScriptedLLM


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def offline_extract(user: str) -> str:
    claims = []
    for number, body in re.findall(
        r"\[Passage (\d+)\][^\n]*\n(.*?)(?=\n\n\[Passage|\Z)", user, re.S
    ):
        sentence = _first_sentence(body)
        evidence = "study" if re.search(r"\bstudy|trial|survey\b", body) else "opinion"
        claims.append(
            {
                "passage": int(number),
                "claim": sentence,
                "speaker": "",
                "stance": "as stated",
                "evidence_type": evidence,
                # Curly quotes and a trailing space, to exercise quote matching.
                "quote": sentence.replace("'", chr(0x2019)) + " ",
            }
        )
    return json.dumps({"claims": claims})


UNCITED = "This sentence makes a point without any citation at all."


def offline_group(user: str) -> str:
    by_book: dict[str, list[str]] = {}
    for label, book in re.findall(r"^\[(C\d+)\] \(([^;]+);", user, re.M):
        by_book.setdefault(book, []).append(label)
    positions = [{"position": f"The view in {b}", "claims": c} for b, c in by_book.items()]
    return json.dumps({"positions": positions})


def offline_answer(user: str) -> str:
    lines = [
        f"The books discuss this claim from {book} in some detail [{label}]."
        for label, book in re.findall(r"^  \[(C\d+)\] \(([^;]+);", user, re.M)
    ]
    lines += [
        f"One passage covers the question in more depth here [{label}]."
        for label in re.findall(r"^\[(P\d+)\]", user, re.M)
    ]
    return " ".join([*lines, UNCITED])


def respond(system: str, user: str) -> str:
    if "List the substantive claims" in user:
        return offline_extract(user)
    if "Group the claims" in user:
        return offline_group(user)
    if "Write an answer in Markdown" in user:
        return offline_answer(user)
    if "A checker found these problems" in user:
        draft = user.split("Draft answer:")[1].split("A checker found")[0]
        return draft.replace(UNCITED, "").strip()
    if "Verified answer:" in user:
        answer = user.split("Verified answer:")[1].split("\n\n")[1]
        return "# A Title\n\n" + answer
    raise ValueError(f"the offline model doesn't recognise this prompt:\n{user[:300]}")


def offline_llm(cache=None) -> ScriptedLLM:
    return ScriptedLLM(respond, cache)
