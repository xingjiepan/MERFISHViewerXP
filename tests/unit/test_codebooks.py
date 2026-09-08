import pytest

from merfishviewerxp.adapters.merlin.codebooks import (
    discover_codebook_files,
    parse_codebook,
)
from merfishviewerxp.errors import DatasetValidationError

CODEBOOK_CSV = """name,id,bit1,bit2,bit3
Blank-1,nan,1,0,0
Blank-2,nan,0,1,0
GENEA,ENST1,1,1,0
GENEB,ENST2,0,1,1
"""


def _write_codebook(tmp_path, filename, content=CODEBOOK_CSV):
    path = tmp_path / filename
    path.write_text(content)
    return path


def test_parse_codebook_barcode_id_is_row_order(tmp_path):
    path = _write_codebook(tmp_path, "codebook_0_test.csv")
    parsed = parse_codebook(path)
    assert parsed.codebook_id == "CB0"
    assert parsed.codebook_index == 0
    assert list(parsed.table["barcode_id"]) == [0, 1, 2, 3]
    assert list(parsed.table["is_blank"]) == [True, True, False, False]


def test_blank_names_disambiguated_by_codebook(tmp_path):
    path0 = _write_codebook(tmp_path, "codebook_0_a.csv")
    path1 = _write_codebook(tmp_path, "codebook_1_b.csv")
    parsed0 = parse_codebook(path0)
    parsed1 = parse_codebook(path1)
    names0 = set(parsed0.table.loc[parsed0.table.is_blank, "gene_name"])
    names1 = set(parsed1.table.loc[parsed1.table.is_blank, "gene_name"])
    assert names0.isdisjoint(names1)
    assert "Blank-1 [CB0]" in names0
    assert "Blank-1 [CB1]" in names1


def test_real_gene_names_not_disambiguated(tmp_path):
    path = _write_codebook(tmp_path, "codebook_0_test.csv")
    parsed = parse_codebook(path)
    real = parsed.table.loc[~parsed.table.is_blank, "gene_name"]
    assert set(real) == {"GENEA", "GENEB"}


def test_discover_codebook_files_none_found_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        discover_codebook_files(tmp_path)


def test_content_hash_changes_with_content(tmp_path):
    path = _write_codebook(tmp_path, "codebook_0_test.csv")
    parsed_a = parse_codebook(path)
    path.write_text(CODEBOOK_CSV.replace("GENEA", "GENEC"))
    parsed_b = parse_codebook(path)
    assert parsed_a.content_hash != parsed_b.content_hash
