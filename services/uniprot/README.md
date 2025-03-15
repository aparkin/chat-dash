# UniProt Service

A service for querying and analyzing protein data from the UniProt Knowledge Base.

## Overview

The UniProt service provides a comprehensive interface to the [UniProt Knowledge Base](https://www.uniprot.org/), enabling users to search for proteins, retrieve protein information, and analyze protein data through both direct queries and natural language interactions.

## Features

- **Protein Search**: Find proteins matching specific criteria using UniProt query syntax
- **Natural Language Queries**: Ask questions about proteins in plain English
- **Protein Lookup**: Retrieve detailed information for specific proteins by accession
- **Result Analysis**: Get biological context and insights for query results
- **Dataset Creation**: Convert query results to datasets for further analysis

## Commands

### Natural Language Query

Ask a question about proteins in plain language:

```
uniprot: Find all human proteins associated with Alzheimer's disease
```

### Direct Query

Submit a structured query using UniProt query syntax:

```uniprot
{
  "query": "gene:BRCA1 AND organism:\"Homo sapiens\"",
  "fields": ["accession", "id", "gene_names", "protein_name", "organism_name"],
  "size": 10
}
```

### Protein Lookup

Look up a specific protein by its UniProt accession:

```
uniprot.lookup P04637
```

### Query Execution

Execute a previously created query:

```
uniprot.search uniprot_query_20230501_120000_orig
```

### Dataset Conversion

Convert query results to a dataset for further analysis:

```
convert uniprot_query_20230501_120000_orig to dataset
```

### Service Information

Get information about the UniProt service:

```
tell me about uniprot
```

## Query Syntax

UniProt queries use a field-based syntax that allows for precise filtering:

- `gene:TP53` - Search by gene name
- `organism:"Homo sapiens"` - Search by organism
- `length:[100 TO 200]` - Filter by length range
- `disease:cancer` - Filter by disease association
- `reviewed:true` - Only include reviewed (SwissProt) entries
- `keyword:631` - Search by keyword (where 631 is "Signal")

Combine filters with Boolean operators:

- `AND`: Both conditions must be true
- `OR`: Either condition can be true
- `NOT`: Exclude matches

Examples:

```
gene:BRCA1 AND organism:"Homo sapiens"
(disease:cancer OR disease:diabetes) AND reviewed:true
```

## Fields

Common fields that can be used in queries and returned in results:

- `accession`: UniProt accession number
- `id`: UniProt entry name
- `gene_names`: Associated gene names
- `protein_name`: Protein name
- `organism_name`: Source organism
- `length`: Protein sequence length
- `reviewed`: Whether the entry is reviewed (SwissProt) or not (TrEMBL)
- `ec`: Enzyme Commission number
- `go`: Gene Ontology terms
- `disease`: Associated diseases
- `pathway`: Biological pathways

## Data Coverage

The service provides access to:

- Over 200 million protein sequences
- Both reviewed (SwissProt) and unreviewed (TrEMBL) entries
- Functional annotations, domain information, and disease associations
- Cross-references to other biological databases

## Implementation Details

The service uses the [UniProt REST API](https://www.uniprot.org/help/api) to fetch data, with appropriate caching and rate limiting to ensure efficient operation.

Natural language queries are processed using LLMs to extract structured query parameters from user questions, making it easier to find relevant proteins without knowing the exact query syntax.

## Usage Examples

### Finding proteins related to a disease

```
uniprot: Find human proteins associated with Alzheimer's disease
```

### Searching for specific enzyme types

```uniprot
{
  "query": "ec:2.7.11.* AND reviewed:true AND organism:\"Homo sapiens\"",
  "fields": ["accession", "id", "gene_names", "protein_name", "ec"],
  "size": 25
}
```

### Looking up a specific protein

```
uniprot.lookup P04637
```
This returns detailed information about the p53 tumor suppressor protein. 