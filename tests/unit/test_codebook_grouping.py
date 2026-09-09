import pandas as pd
import pyarrow as pa

from merfishviewerxp.viewer.layers import all_codebook_ids, genes_by_codebook, split_table_by_codebook


def _genes_df():
    return pd.DataFrame(
        [
            {"gene_id": 1, "gene_name": "GENEA", "codebook_ids": "CB0"},
            {"gene_id": 2, "gene_name": "GENEB", "codebook_ids": "CB0"},
            {"gene_id": 3, "gene_name": "exo_GENEC", "codebook_ids": "CB1"},
            {"gene_id": 4, "gene_name": "SHARED", "codebook_ids": "CB0,CB1"},
        ]
    )


def test_all_codebook_ids():
    assert all_codebook_ids(_genes_df()) == ["CB0", "CB1"]


def test_genes_by_codebook_splits_correctly():
    grouped = genes_by_codebook(_genes_df())
    assert set(grouped.keys()) == {"CB0", "CB1"}
    assert set(grouped["CB0"]["gene_name"]) == {"GENEA", "GENEB", "SHARED"}
    assert set(grouped["CB1"]["gene_name"]) == {"exo_GENEC", "SHARED"}


def test_genes_by_codebook_does_not_substring_match():
    # a codebook id "CB1" must not match inside e.g. "CB10" for a hypothetical dataset
    df = pd.DataFrame([{"gene_id": 1, "gene_name": "X", "codebook_ids": "CB10"}])
    grouped = genes_by_codebook(df)
    assert "CB1" not in grouped
    assert "CB10" in grouped


def test_split_table_by_codebook():
    table = pa.table(
        {
            "spot_id": [1, 2, 3],
            "codebook_id": ["CB0", "CB1", "CB0"],
            "gene_name": ["GENEA", "exo_GENEC", "GENEB"],
        }
    )
    split = split_table_by_codebook(table, ["CB0", "CB1"])
    assert split["CB0"].column("spot_id").to_pylist() == [1, 3]
    assert split["CB1"].column("spot_id").to_pylist() == [2]


def test_split_table_by_codebook_includes_empty_entries_for_missing_ids():
    table = pa.table({"spot_id": [1], "codebook_id": ["CB0"], "gene_name": ["GENEA"]})
    split = split_table_by_codebook(table, ["CB0", "CB1"])
    assert split["CB0"].num_rows == 1
    assert split["CB1"].num_rows == 0
