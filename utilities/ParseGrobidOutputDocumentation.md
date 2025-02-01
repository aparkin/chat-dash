# TEI XML Data Structure

This document outlines the structure of the data extracted from TEI XML files using the `parse_tei` function in the `ParseTEI.py` script. The extracted data is enriched with named entity recognition (NER) results using BERT models, which identify entities related to Genes, Bioprocesses, Chemicals, and Organisms.

## Data Structure

```json
{
    "titles_with_hierarchy": [
        {
            "title": "string",
            "path": "string"
        },
        ...
    ],
    "authors": [
        {
            "name": "string",
            "email": "string"
        },
        ...
    ],
    "affiliations": [
        {
            "org_names": ["string", ...],
            "address": "string"
        },
        ...
    ],
    "abstract": "string",
    "references": [
        {
            "id": "string",
            "title": "string",
            "authors": ["string", ...],
            "journal": "string",
            "volume": "string",
            "pages": "string",
            "publication_date": "string",
            "raw_reference": "string"
        },
        ...
    ],
    "body_sections": [
        {
            "title": "string",
            "text": "string",
            "citations": ["string", ...]
        },
        ...
    ],
    "funding_info": [
        {
            "funder": "string",
            "grant_number": "string"
        },
        ...
    ],
    "acknowledgements": ["string", ...],
    "tables": ["string", ...],
    "figures": ["string", ...],
    "publication_info": {
        "publisher": "string",
        "availability": "string",
        "published_date": "string",
        "md5": "string",
        "doi": "string",
        "submission_note": "string"
    },
    "ner_results_<model_name>": [
        {
            "term": "string",
            "best_score": "float"
        },
        ...
    ]}
```

## Explanation

- **Titles with Hierarchy**: Captures titles and their hierarchical paths within the document.
- **Authors**: Contains author names and emails extracted from the document.
- **Affiliations**: Lists organization names and addresses associated with the authors.
- **Abstract**: The document's abstract, providing a summary of the content.
- **References**: Detailed bibliographic information for each reference cited in the document.
- **Body Sections**: Sections of the document with titles, text, and citations, representing the main content.
- **Funding Info**: Information about funders and associated grant numbers, indicating financial support.
- **Acknowledgements**: Texts of acknowledgements, recognizing contributions and support.
- **Tables and Figures**: Texts of tables and figures included in the document.
- **Publication Info**: Metadata about the publication, including publisher, availability, and identifiers.
- **NER Results**: Named entity recognition results for each model, identifying key terms and their confidence scores.

This structure provides a comprehensive view of the extracted data, enriched with semantic information from NER processing, making it valuable for further analysis and integration into other systems.