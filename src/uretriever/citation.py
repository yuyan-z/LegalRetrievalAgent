import re


LAW_ABBREVS = (
    "ZGB|OR|StGB|BV|SchKG|ZPO|StPO|BGG|VwVG|IPRG|KG|DSG|MSchG|URG|PatG|"
    "DesG|UWG|PrSG|FINMAG|BankG|VAG|KAG|GwG|BEHG|FinfraG|FIDLEG|FINIG|"
    "ATSG|AHV|IVG|IV|EO|ALV|KVG|UVG|BVG|ArG|GlG|USG|RPG|WaG|JSG|TSchG|StBOG|"
    "LwG|PBG|EBG|SVG|LFG|SebG|SpG|BoeB|EMRK|SR|AS|BBl|ParlG|RVOG|RVOV|"
    "MG|BPG|BPV|VBGÖ|VDSG|MWSTG|DBG|StHG|VStG|StG|ZG|CO|CP|CC|CPC|CPP|"
    "LEtr|LAsi|LN|LDIP|LCart|LDA|LPM|LBI|LDes|LCD|LFINMA|LB|LSA|LPCC|"
    "LBA|LBVM|LIMF|LSFin|LEFin|LAVS|LAI|LAPG|LACI|LAMal|LAA|LPP|LTr|"
    "LEg|LPD|LPE|LAT|LFo|LChP|LPN|LAgr|LTV|LCdF|LNA|LPTh|LTAF|LTF"
)

SR_PATTERN = (
    r"SR\s+"
    r"(\d+(?:\.\d+)*)"   # Code with optional sublevel codes
)

BGE_PATTERN = (
    r"BGE\s+"
    r"(\d{1,3})\s+"                        # Volume
    r"([IVX]+[a-z]?)\s+"                   # Section
    r"(\d+)"                               # Page
    r"(?:\s+(?:E\.|cons\.?|Erw\.?)\s*"     # Optional Consideration prefix
    r"((?:[IVX]+\.)?"                      # Optional Roman prefix
    r"(?:\d+[a-z]?)"                       # Base: digit(s) + optional letter
    r"(?:\.\d+[a-z]?)*"                    # Decimal extensions
    r"(?:(?:/[a-z]+)|(?:-\d+[a-z]?))?))?"  # Suffix: slash+letters OR range
)

PARAGRAPH_PATTERN = (
    r"(?:Abs\.?|Absatz|al\.?|cpv\.?)\s+"  # Prefix
    r"(\d+(?:bis|[a-z])?)"                # Paragraph number
    r"(?![a-zA-Z])"                       # Not followed by another letter
)

ARTICLE_PATTERN = (
    r"(?:Art\.?|Artikel)\s+"
    r"(\d+[a-z]?)"
    r"(?![a-zA-Z])"
    r"(?:\s+" + PARAGRAPH_PATTERN + r")?"
    r"(?:\s+(?:" + LAW_ABBREVS + r"))?"
)

JUDGMENT_PATTERN = (
    r"(\d+[A-Z]_\d+/\d{4})"  # [Court Case Number][Case Category Letter]_[Serial Number]/[Year]
    r"(?:\s+(?:E\.|Erw\.?|cons\.?)\s*"
    r"((?:\d+[a-z]?)(?:\.\d+[a-z]?)*))?"
)


def extract_citations(text: str) -> list[str]:
    """Extract citations from a text.

    Args:
        text: The text to extract citations from.

    Returns:
        citations: A list of string.
    """
    pattern = re.compile(f"{SR_PATTERN}|{BGE_PATTERN}|{ARTICLE_PATTERN}|{JUDGMENT_PATTERN}")
    return list(dict.fromkeys(m.group(0) for m in pattern.finditer(text)))


def parse_citations(citation_str: str, separator: str = ";") -> list[str]:
    """Parse citation string into list.

    Args:
        citation_str: The string containing citations, with a separator.

    Returns:
        citations: A list of string.
    """
    if not citation_str or citation_str.strip() == "":
        return []
    return [c.strip() for c in citation_str.split(separator) if c.strip()]


if __name__ == "__main__":
    import pandas as pd

    from uretriever.configs.path_config import DATA_DIR

    file_path = DATA_DIR / "omnilex" / "raw" / "val.csv"
    df = pd.read_csv(file_path)

    for text in df["gold_citations"]:
        citations = extract_citations(text)
        gold_citations = parse_citations(text)
        diff = list(set(gold_citations) - set(citations))
        print(diff)
        print()
