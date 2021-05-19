import pandas as pd
col_list = ['PUBMEDID', 'DISEASE/TRAIT', 'REPORTED GENE(S)', 'MAPPED_GENE', 'MAPPED_TRAIT', 'P-VALUE', 'SNPS']
datf = pd.read_csv('./gwas_catalog_v1.0.2-associations_e100_r2021-05-05.tsv', sep='\t', usecols=col_list)
datf.to_csv('./gwas_used.csv', sep='\t')
