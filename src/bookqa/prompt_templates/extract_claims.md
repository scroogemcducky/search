You extract claims from book passages for a research database. You are precise and literal. You never add information that isn't in the passage.
<<<USER>>>
Book: "$title" by $author
Chapter: $chapter

Below are numbered passages from this chapter. List the substantive claims they make: statements of fact, findings, recommendations or arguments that a reader could agree or disagree with. Skip scene-setting, transitions and rhetorical questions.

For each claim give:
- "passage": the number of the passage it comes from
- "claim": the claim restated as one self-contained sentence that makes sense without the passage
- "speaker": who makes the claim. Use the book's author unless the passage attributes it to someone else
- "stance": the speaker's position in five words or fewer
- "evidence_type": exactly one of: $evidence_types
- "quote": the shortest exact sentence or phrase from the passage that supports the claim, copied character for character

How to choose the evidence type:
$evidence_guide

Answer with JSON only, in this shape:
{"claims": [{"passage": 1, "claim": "...", "speaker": "...", "stance": "...", "evidence_type": "...", "quote": "..."}]}

If the passages make no substantive claims, answer {"claims": []}.

$passages
