# ng_accuracy
Predict accuracy of "nearest gene" method for identifying causal genes

Steps 
1. Use file from https://zenodo.org/records/15594712 . This file contains the data and scripts for the preprint "Large language models identify causal genes in complex trait GWAS" .
2. Unzip file, find the opentargets_step* files in the data folder.
   They describe GWAS loci where the causal gene is known, in the column "symbol".
3. Find the predictions for causal gene by the "nearest gene" method in the file opentargets.nearest_gene.csv in the predictions folder.
4. Using these files, you can evaluate at every locus whether the causal gene prediction from the "nearest gene" method was correct or not.
5. Think about what features of the locus/region could be used to predict whether "nearest gene" identified the right causal gene.
6. Focus on features that can be extracted from the available files.
7. Split data into train/test, and create a simple logistic regression predictor for this task.
8. Report the test AUC for this predictor.
