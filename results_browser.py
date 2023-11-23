import base64
import json
import pathlib
import sqlite3
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as mt
import streamlit as st
import yaml
from metaflow import Run

import common

# Set the layout to wide mode
st.set_page_config(layout="wide")

args = sys.argv[1:]
ROOT = pathlib.Path("data").absolute()
if not args:
    DB = ROOT / "funk-etal-2008.db"
else:
    DB = pathlib.Path(args[0]).resolve()


# Processing
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS notes (
                    run_id TEXT,
                    doi TEXT,
                    note TEXT,
                    UNIQUE(run_id, doi))"""
    )
    conn.commit()
    conn.close()


def add_note(run_id, doi, note):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO notes (run_id, doi, note) VALUES (?, ?, ?)",
        (run_id, doi, note),
    )
    conn.commit()
    conn.close()


def get_note(run_id, doi):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT note FROM notes WHERE run_id = ? AND doi = ?",
        (run_id, doi),
    )
    note = c.fetchone()
    conn.close()
    if note:
        return note[0]
    return ""


def load_paper_metadata(doi):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT semantic_scholar_json FROM papers WHERE json_extract(semantic_scholar_json, '$.externalIds.DOI') = ?",
        (doi,),
    )

    return json.loads(c.fetchone()[0])


def load_pdf(doi):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT pdf_content FROM pdfs WHERE doi = ?", (doi,))

    result = c.fetchone()
    if result:
        return result[0]

    return None


def pdf_display(pdf_byted, width="100%", height=1700):
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<embed src="data:application/pdf;base64,{pdf_b64}" width="{width}" height="{height}" type="application/pdf" />'
    return pdf_display


# def pdf_display(pdf_bytes, width="100%", height="1700px"):
#     pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
#     pdf_html = f"""
#         <iframe src="https://mozilla.github.io/pdf.js/web/viewer.html?file=data:application/pdf;base64,{pdf_b64}"
#                 width="{width}" height="{height}" style="border: none;">
#         </iframe>
#     """
#     return pdf_html


def plot_confusion_matrix_heatmap(df, classes, remove_error_class=True):
    df = df.copy().rename(columns={"observed": "Observed", "expected": "Expected"})

    if remove_error_class:
        # df = df.query("observed != 'error'").query("expected != 'error'")
        classes = [e for e in classes if e != "error"]

    # Create the confusion matrix
    conf_matrix = pd.crosstab(
        df["Observed"],
        df["Expected"],
        # rownames=["Expected"],
        # colnames=["Observed"],
    )

    # Reorder the confusion matrix according to the classes list
    conf_matrix = conf_matrix.reindex(index=classes, columns=classes, fill_value=0)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        annot_kws={"fontsize": "x-large"},
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
    )
    # Making x and y axis labels boldface
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold", fontsize="x-large")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold", fontsize="x-large")

    # Rotating the x-axis labels to 45 degrees
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize="x-large"
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize="x-large")

    # Draw percentage.
    # total = conf_matrix.sum().sum()

    # def pct(e):
    #     return 100.0 * float(e) / total

    # for t in ax.texts:
    #     t.set_text(t.get_text() + f"\n({pct(t.get_text()):.1f}%)")

    st.pyplot(fig)


def confusion_matrix(y_true, y_pred, labels):
    matrix = defaultdict(int)  # matrix[(actual, predicted)] will store count
    for true, pred in zip(y_true, y_pred):
        matrix[(true, pred)] += 1

    # Now convert to 2D list in the order of 'labels'
    return [
        [matrix[(true_cls, pred_cls)] for pred_cls in labels] for true_cls in labels
    ]


def compute_classification_stats(df):
    y_true = df["expected"]
    y_pred = df["observed"]

    accuracy = mt.accuracy_score(y_true, y_pred)
    f1_score = mt.f1_score(y_true, y_pred, average="weighted")
    precision = mt.precision_score(y_true, y_pred, average="weighted")
    recall = mt.recall_score(y_true, y_pred, average="weighted")
    mcc = mt.matthews_corrcoef(y_true, y_pred)

    return {
        # "class": cls,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mcc": mcc,
    }


# Example usage:
# y_true = [1, 0, 1, 1, 0, 1]  # Replace with the true class labels
# y_pred = [1, 0, 0, 1, 0, 1]  # Replace with the predicted class labels by the classifier

# stats = compute_classification_stats(y_true, y_pred)
# print(stats)


# def compute_classification_stats(df, classes):
#     conf_matrix = confusion_matrix(df["expected"], df["observed"], classes)

#     stats = compute_classification_stats

#     # results = []
#     # for i, cls in enumerate(classes):
#     #     tp = conf_matrix[i][i]
#     #     fp = sum(conf_matrix[j][i] for j in range(len(classes))) - tp
#     #     fn = sum(conf_matrix[i]) - tp
#     #     tn = sum(sum(row) for row in conf_matrix) - (tp + fp + fn)

#     #     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     #     precision = tp / (tp + fp) if tp + fp != 0 else 0
#     #     recall = tp / (tp + fn) if tp + fn != 0 else 0
#     #     f1_score = (
#     #         2 * (precision * recall) / (precision + recall)
#     #         if precision + recall != 0
#     #         else 0
#     #     )

#     #     record = {
#     #         "class": cls,
#     #         "accuracy": accuracy,
#     #         "f1_score": f1_score,
#     #         "precision": precision,
#     #         "recall": recall,
#     #     }
#     #     results += [record]

#     return pd.DataFrame(results)


init_db()


ordered_classes = [
    "not-an-empirical-paper",
    "not-an-experiment",
    "traits-not-measured",
    "traits-NOT-integral-to-experimental-design",
    "traits-integral-to-experimental-design",
    "error",
]

ordered_classes_map = {
    "not-an-empirical-paper": "something-else",
    "not-an-experiment": "something-else",
    "traits-not-measured": "something-else",
    "traits-NOT-integral-to-experimental-design": "something-else",
    "traits-integral-to-experimental-design": "traits-integral-to-experimental-design",
    "error": "error",
}


# Sidebar content
st.sidebar.title("Experiments")

# Get list of result folders and create dropdown in sidebar
RUN_IDS = pd.read_sql(
    "select distinct run_id from analysis order by run_id desc", sqlite3.connect(DB)
).run_id.tolist()
selected_run_id = st.sidebar.selectbox("Select a run:", RUN_IDS)


assignments_files = {e.stem: e for e in ROOT.glob("*annotations.yaml")}
selected_assignments_file = st.sidebar.selectbox("Assignments File:", assignments_files)

with open(assignments_files[selected_assignments_file]) as f:
    assignments_df = common.assignments_to_df(yaml.safe_load(f)).set_index("doi")

analysis = (
    pd.read_sql(
        """
SELECT
  analysis.*
FROM analysis
WHERE
  analysis.run_id = ?
""",
        sqlite3.connect(DB),
        params=[selected_run_id],
    )
    .merge(assignments_df.rename(columns={"node": "expected"}), on="doi", how="left")
    .assign(
        # Handle unannotated case.
        expected=lambda f: f.expected.fillna("error"),
        states=lambda f: f.states_json.apply(json.loads),
        chat=lambda f: f.chat_json.apply(json.loads),
        answers=lambda f: f.answers_json.apply(json.loads),
        observed=lambda f: [e[-1] for e in f.states],
        is_correct=lambda f: f.expected == f.observed,
        doi_annot=lambda f: [
            f"{'✅' if t.is_correct else '❌'} {t.doi}" for t in f.itertuples()
        ],
    )
    .set_index("doi_annot")
)


show_only_incorrect = st.sidebar.checkbox("Show only misclassifications.", value=True)

# Then when you fetch or display your papers
if show_only_incorrect:
    # Filter your dataframe or list to show only incorrect papers
    selectable_dois = analysis.query("~is_correct").index
else:
    selectable_dois = analysis.index

# New sidebar control for filtering by 'expected' class
selected_classes = st.sidebar.multiselect("Filter by Expected Class:", ordered_classes)

# Applying the filters
if show_only_incorrect and selected_classes:
    selectable_dois = analysis.query(
        "~is_correct & expected in @selected_classes"
    ).index
elif show_only_incorrect:
    selectable_dois = analysis.query("~is_correct").index
elif selected_classes:
    selectable_dois = analysis.query("expected in @selected_classes").index
else:
    selectable_dois = analysis.index


# Show DOIs in a dropdown in the sidebar
selected_doi = st.sidebar.selectbox(
    "Select a result:", ["SUMMARY", *selectable_dois.tolist()]
)

st.title("🤖 LLM Meta-analysis Results Browser")

ref = None
col1, col2 = st.columns(2)
if selected_doi == "SUMMARY":
    just_doi = "SUMMARY"
    with col1:
        st.header("Classification Summary")

        _, col1a, _ = st.columns([1, 3, 1])
        with col1a:
            eno = analysis.loc[:, ["expected", "observed"]]
            st.table(compute_classification_stats(eno))  # , ordered_classes))
            plot_confusion_matrix_heatmap(eno, ordered_classes)

            # Compress earlier classes.
            eno_c = analysis.assign(
                expected=lambda f: f.expected.map(ordered_classes_map),
                observed=lambda f: f.observed.map(ordered_classes_map),
            ).loc[:, ["expected", "observed"]]
            compressed_classes = [
                "something-else",
                "traits-integral-to-experimental-design",
                "error",
            ]
            st.table(compute_classification_stats(eno_c))  # , compressed_classes))

            plot_confusion_matrix_heatmap(eno_c, compressed_classes)

    with col2:
        params = [
            "db_path",
            "vector_db_path",
            "tree_path",
            "embeddings_model",
            "papers_collection",
            "model",
            "max_paper_tokens",
            "temperature",
            "max_reprompts",
        ]
        data = Run(f"AnalyzePapersFlow/{selected_run_id}").data
        st.header("Run Params")
        run_data = {p: getattr(data, p) for p in params}
        st.table(run_data)
        st.write(repr(run_data))


else:
    record = analysis.loc[selected_doi]
    just_doi = record.doi

    # Load Semantic Scholar JSON for DOI from db.
    ref = load_paper_metadata(just_doi)

    with col1:
        dt_path = " → ".join(
            f"[<code>{e}</code>]" for e in ["START"] + record["states"]
        )
        st.header("Classification Info")
        st.markdown(
            f"""\
            <style>
                .big-table {{
                    font-size: 1.3em;
                }}
            </style>

            <table class="big-table">
                <tr>
                    <td><strong>DOI</strong></td>
                    <td>{selected_doi}</td>
                </tr>
                <tr>
                    <td><strong>Reference Class</strong></td>
                    <td>{record["expected"]}</td>
                </tr>
                <tr>
                    <td><strong>LLM's Class</strong></td>
                    <td>{record["observed"]}</td>
                </tr>
                <tr>
                    <td><strong>Path</strong></td>
                    <td>{dt_path}</td>
                </tr>
            """,
            unsafe_allow_html=True,
        )

        st.header("Paper Info")
        author_html = (
            "<ol>"
            + "".join([f"<li>{e['name']}</li>" for e in ref["authors"]])
            + "</ol>"
        )
        try:
            abstract_html = ref["abstract"].replace("\n", "<br>")
        except Exception:
            abstract_html = "<i>No abstract in Semantic Scholar record.</i>"
        st.markdown(
            f"""\
            <style>
                .p-table {{
                    font-size: 1.1em;
                }}
            </style>

            <table class="p-table">
                <tr>
                    <td><strong>Journal</strong></td>
                    <td>{ref["journal"].get("name", "MISSING JOURNAL NAME")}</td>
                </tr
                <tr>
                    <td><strong>Title</strong></td>
                    <td>{ref["title"]}</td>
                </tr
                <tr>
                    <td><strong>Authors</strong></td>
                    <td>{author_html}</td>
                </tr
                <tr>
                    <td><strong>Abstract</strong></td>
                    <td><span style="font-size: 1em;">{abstract_html}</span></td>
                </tr
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.header("LLM Responses")

        def render_table_with_html(messages):
            html_content = "<table>"
            html_content += "<tr><th>Role</th><th>Text</th></tr>"

            for m in messages:
                role = m["id"][-1]
                try:
                    # mj = json.loads(m.content)
                    mj = json.loads(
                        m["kwargs"]["content"]
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    text = "<table>"
                    text += "<tr><th>Field</th><th>Text</th></tr>"
                    for k, v in mj.items():
                        text += f"<tr><td>{k}</td><td>{v}</td></tr>"
                    text += "</table>"
                except ValueError:
                    text = m["kwargs"]["content"].replace("\n", "<br/>")

                html_content += f"<tr><td>{role}</td><td>{text}</td></tr>"

            html_content += "</table>"

            return html_content

        st.write(render_table_with_html(record["chat"]), unsafe_allow_html=True)

    # Put your right-side (PDF iframe) content here
    with col2:
        pdf_bytes = load_pdf(just_doi)

        st.markdown(pdf_display(pdf_bytes), unsafe_allow_html=True)

with col1:
    st.header("Notes")

    st.session_state["notes"] = get_note(selected_run_id, just_doi)
    get_note(selected_run_id, just_doi)
    user_input = st.text_area("", st.session_state["notes"], height=250)

    if st.button("Save Note"):
        add_note(selected_run_id, just_doi, user_input)

with st.expander("Much Details!"):
    colA, colB = st.columns(2)
    with colA:
        st.table(analysis.loc[:, ["doi", "expected", "observed"]])

    with colB:
        if ref is not None:
            st.write(ref)
