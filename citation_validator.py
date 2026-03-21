import re

def validate_citations(response_text: str, retrieved_docs: dict) -> tuple[bool, str]:
    if "I cannot answer this" in response_text:
        return True, response_text
    

    cited_ids = re.findall(r"\[(?:Source ID:\s*)?([^\]]+)\]", response_text)


    rag_citations = []
    for citation in cited_ids:
        clean_citation = citation.strip()
        if re.match(r"^[\d\s,]+$", clean_citation):
            continue
        rag_citations.append(clean_citation)


    if len(rag_citations) == 0:
        return False, "Safety Guardrail Triggered: The model provided an answer but failed to cite its sources."
    

    for clean_citation in rag_citations:
        if clean_citation not in retrieved_docs:
            return False, f"Safety Guardrail Triggered: The model hallucinated a fake source -> [{clean_citation}]"

    return True, response_text