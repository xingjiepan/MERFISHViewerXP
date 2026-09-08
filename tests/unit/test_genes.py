from merfishviewerxp.model.genes import (
    deterministic_gene_id,
    gene_color_hex,
    gene_color_rgb,
    is_blank_name,
)


def test_is_blank_name_case_insensitive():
    assert is_blank_name("Blank-1")
    assert is_blank_name("BLANK_23")
    assert is_blank_name("blank")
    assert not is_blank_name("PDGFRA")


def test_gene_id_deterministic_across_calls():
    assert deterministic_gene_id("PDGFRA") == deterministic_gene_id("PDGFRA")


def test_gene_id_independent_of_unrelated_state():
    ids_a = [deterministic_gene_id(n) for n in ["A", "B", "C"]]
    ids_b = [deterministic_gene_id(n) for n in ["C", "B", "A"]]
    assert set(ids_a) == set(ids_b)
    assert deterministic_gene_id("A") == ids_a[0] == ids_b[2]


def test_gene_color_deterministic():
    assert gene_color_rgb("PDGFRA") == gene_color_rgb("PDGFRA")
    assert gene_color_hex("PDGFRA") == gene_color_hex("PDGFRA")


def test_gene_color_hex_format():
    h = gene_color_hex("SOMEGENE")
    assert h.startswith("#")
    assert len(h) == 7
    int(h[1:], 16)  # must be valid hex


def test_different_genes_usually_differ_in_color():
    names = [f"GENE{i}" for i in range(50)]
    colors = {gene_color_hex(n) for n in names}
    assert len(colors) > 40  # allow rare collisions, but expect mostly distinct
