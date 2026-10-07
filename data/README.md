# Authorized input schema

No participant data are included. Keep patient-derived files in this ignored directory or a separate approved location.

The historical loader expects these filenames. All arrays and metadata must already share the same subject/feature ordering. Matching row counts alone does not establish identity alignment.

| File | Required contents |
| --- | --- |
| `canonical_subjects.csv` | `ID` and `label`; HC=0, RA=1; one unique ID per row |
| `sers_mean3_139x1934.npy` | Subject × spectral channel matrix, before wavelength restriction; 139 × 1934 in the original analysis |
| `sers_wavenumbers.csv` | One numeric wavenumber per line, matching spectral columns |
| `transcript_fpkm_139x62354.npy` | Nonnegative FPKM matrix; 139 × 62354 originally |
| `transcript_count_139x62354.npy` | Aligned count matrix, loaded for compatibility although not used by primary feature selection |
| `transcript_feature_metadata.csv` | One row per transcript column, including `gene_id` and `gene_name` |
| `metabolomics_full_139x2388.npy` | Nonnegative intensity matrix; 139 × 2388 originally |
| `metabolomics_feature_metadata.csv` | One row per metabolite column, including `Compound_ID` |

The primary model restricts SERS to 68–4000 cm⁻¹, giving 1796 measured channels in the original data. SNV is performed within each spectrum. Transcript FPKM prevalence, variance screening, ANOVA feature selection, feature grouping and scaling are fitted within training data. The selected dimensions are 512 transcripts and 128 metabolites.

For pathway analysis, metabolite metadata additionally require `Level`, `KEGG_ID` and `KEGG_MapID`. Locally supply permitted KEGG reference files `genes_full.txt`, `gene_path_full.txt` and `path_names.txt` in `data/reference_kegg_20260916/`. Reference data are not redistributed in this package.

The synthetic generator preserves historical filenames for loader compatibility while creating artificial IDs, labels and smaller omics matrices. It writes `SYNTHETIC_ONLY.json` to prevent confusion with research data.
