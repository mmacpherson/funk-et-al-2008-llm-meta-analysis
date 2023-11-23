# GPT-Based Scientific Literature Meta-Analysis

This repository accompanies the blog post ["How Well Can GPT Do Scientific
Literature
Meta-analysis?"](http://finedataproducts.com/posts/2023-12-31-llm-based-metaanalysis/)
, exploring the use of GPT-4 for conducting meta-analysis of scientific
literature. The project investigates whether Large Language Models (LLMs) like
GPT-4 can assist in synthesizing research findings, potentially offering a more
efficient, consistent, and
cost-effective approach compared to traditional methods.

A prototype results browser app is available at
<https://llm-metaanalysis.finedataproducts.com/>, to view the output of the analysis on
a selection of open-access papers.

## Repository Structure

- `analyze_papers.py`: Workflow for analyzing papers using GPT models.
- `common.py`: Shared utility functions and classes.
- `data/`: Directory containing datasets and annotations.
- `decision_tree_chat.py`: Interactive decision tree for paper classification.
- `process_papers.py`: Workflow for initial processing of papers.
- `requirements.{in,txt}`: Python dependencies for the project.
- `results_browser.py`: A Streamlit app for browsing results.
- `scripts/`: Utility scripts for data processing and analysis.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mmacpherson/funk-et-al-2008-llm-meta-analysis.git
   cd funk-et-al-2008-llm-meta-analysis
   ```

2. Set up the environment and install python dependencies:
   ```bash
   make env # Requires `pyenv` and `pyenv-virtualenv`.
   ```

   (The included `requirements.txt` supplies the dependencies needed to run the
   analysis, if you prefer to manage your virtualenvs with something other than
   pyenv/pyenv-virtualenv.)

3. Configure OpenAI API access, by creating a file called `.env` with an entry like:
   ```bash
   OPENAI_API_KEY={key_here}
   ```

## Running the Analysis

### Processing Paper Text

To process PDF papers into a vector store, run this script:

```bash
python process_papers.py run
```

Run `python process_papers.py run --help` to see available arguments.

As provided here, this workflow assumes that you've looked up some set of papers
using Semantic Scholar's API, and stored them in a `papers` table with this
schema:

``` sql
CREATE TABLE papers (
        semantic_scholar_id TEXT PRIMARY KEY NOT NULL,
        semantic_scholar_json TEXT NOT NULL
    )
```

And a table `pdfs` with this schema, that contains the pdf content for each
paper:

``` sql
CREATE TABLE pdfs (
        doi TEXT PRIMARY KEY NOT NULL,
        pdf_content BLOB NOT NULL,
        pdf_md5 TEXT NOT NULL,
        direct INTEGER NOT NULL -- Treated as boolean; could we download directly from the open internet, aot UC?
    )
```

If those tables exist, the downstream sqlite tables and chromadb vector store
will be created automatically.

The sqlite database provided at
`data/funk-etal-2008.selected-open-access.db` contains example data.


### Running the LLM Meta-Analysis

To run the meta-analysis itself:

```bash
python analyze_papers.py run
```

Run `python analyze_papers.py run --help` to see available arguments. See e.g.
[`scripts/run-pilot-set`](scripts/run-pilot-set) for the command used to run the
analysis over our pilot/training set.


### Running the Results Browser App

```bash
streamlit run results_browser.py
```


## Contributing

This repository provides the code to reproduce the analysis described in the
accompanying blog post. As it is meant primarily for replication purposes,
active development is limited.

However, if you have any questions, comments, or suggestions, please feel free
to open an issue! I'm happy to answer questions about the methodology, discuss
the findings, or hear any ideas you may have for extending the analysis.

Contributions in the form of bug reports, feature requests, or pull requests are
also welcome, though I can't guarantee very active maintenance. I'm opening this
code in the hopes that others may find it useful or build upon it in their own
work.

## License

This project is licensed under the CC0 1.0 Universal License. For more details,
see the [LICENSE](LICENSE) file in this repository or visit [Creative Commons
CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
