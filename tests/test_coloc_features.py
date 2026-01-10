import unittest
import pandas as pd

from src.ng_accuracy.feature_blocks.coloc_block import compute_coloc_features


class ColocFeatureTest(unittest.TestCase):
    def setUp(self):
        self.nearest = pd.DataFrame(
            [
                {"studyLocusId": "l1", "nearestGeneId": "ENSG1", "nearestGeneId_base": "ENSG1"},
                {"studyLocusId": "l2", "nearestGeneId": "ENSG2", "nearestGeneId_base": "ENSG2"},
                {"studyLocusId": "l3", "nearestGeneId": "ENSG3", "nearestGeneId_base": "ENSG3"},
                {"studyLocusId": "l4", "nearestGeneId": "ENSG4", "nearestGeneId_base": "ENSG4"},
            ]
        )

        self.coloc_df = pd.DataFrame(
            [
                {"leftStudyLocusId": "l2", "rightStudyType": "eqtl", "h4": 0.2, "clpp": 0.1},
                {"leftStudyLocusId": "l3", "rightStudyType": "pqtl", "h4": 0.7, "clpp": 0.4, "qtlGeneId": "ENSGX.1"},
                {"leftStudyLocusId": "l3", "rightStudyType": "pqtl", "h4": 0.5, "clpp": 0.2, "qtlGeneId": "ENSGX.1"},
                {"leftStudyLocusId": "l4", "rightStudyType": "pqtl", "h4": 0.9, "clpp": 0.05, "qtlGeneId": "ENSG4"},
            ]
        )

    def test_coloc_status_and_no_nas(self):
        result = compute_coloc_features(self.coloc_df, self.nearest)
        self.assertFalse(result.filter(like="coloc_").isna().any().any())

        status_cols = [
            "coloc_status_no_pairs",
            "coloc_status_pairs_no_mapped_gene",
            "coloc_status_mapped_gene_no_nearest",
            "coloc_status_nearest_match",
        ]
        self.assertTrue(((result[status_cols].sum(axis=1)) == 1).all())

        status_by_id = dict(zip(result["studyLocusId"], result[status_cols].idxmax(axis=1)))
        self.assertEqual(status_by_id["l1"], "coloc_status_no_pairs")
        self.assertEqual(status_by_id["l2"], "coloc_status_pairs_no_mapped_gene")
        self.assertEqual(status_by_id["l3"], "coloc_status_mapped_gene_no_nearest")
        self.assertEqual(status_by_id["l4"], "coloc_status_nearest_match")

    def test_nearest_and_best_logic(self):
        result = compute_coloc_features(self.coloc_df, self.nearest)
        row_l3 = result[result["studyLocusId"] == "l3"].iloc[0]
        self.assertEqual(row_l3["coloc_best_gene_h4"], 0.7)
        self.assertEqual(row_l3["coloc_max_h4_nearest_gene"], 0.0)
        self.assertAlmostEqual(row_l3["coloc_nearest_vs_best_h4_ratio"], 0.0)

        row_l4 = result[result["studyLocusId"] == "l4"].iloc[0]
        self.assertEqual(row_l4["coloc_best_gene_h4"], 0.9)
        self.assertEqual(row_l4["coloc_max_h4_nearest_gene"], 0.9)
        self.assertGreater(row_l4["coloc_nearest_vs_best_h4_ratio"], 0.99)


if __name__ == "__main__":
    unittest.main()
