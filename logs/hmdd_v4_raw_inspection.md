# HMDD v4.0 raw-file inspection

- Inspection date: 2026-09-05
- File: `data/raw/hmdd/hmdd_v4_2023_07_original.txt`
- File size (bytes): 12849414
- SHA-256: `79492eb7cb2b5d9982fdcaa8a55b308ca969afa31a9238f740eb2a2d0bf93e51`
- Detected encoding: `utf-8-sig`
- Detected delimiter: `'\t'`
- Total physical rows including header: 53554
- Data rows excluding header: 53553
- Distinct row widths: [5]

## Header

```text
['code', 'PMID', 'miRNA', 'disease', 'description']
```

## First five raw data rows

```text
['other', '36948853', 'hsa-mir-40', 'Liver Cirrhosis', 'MiR-340 mediates the involvement of high mobility group box 1 in the pathogenesis of liver fibrosis']
['target gene', '36948479', 'hsa-mir-29b', 'Zika Virus Infection', 'Zika virus NS1 suppresses VE-cadherin via hsa-miR-29b-3p/DNMT3b/MMP-9 pathway in human brain microvascular endothelial cells.']
['exosome', '36948039', 'hsa-mir-130b', 'Wound Healing', 'ADSCs-derived exosomal H19 accelerates cutaneous wound healing via the miR-130b-3p/PPARγ/STAT3 axis']
['target gene', '36947996', 'hsa-mir-221', 'Stomach Neoplasms', 'Downregulation of miR-221-3p promotes the ferroptosis in gastric cancer cells via upregulation of ATF3 to mediate the transcription inhibition of GPX4 and HRD1.']
['other', '36947456', 'hsa-mir-497', 'Pancreatic Neoplasms', 'Long noncoding RNA CASC9 promotes pancreatic cancer progression by acting as a ceRNA of miR-497-5p to upregulate expression of CCND1.']
```

## First twelve physical lines

```text
code	PMID	miRNA	disease	description
other	36948853	hsa-mir-40	Liver Cirrhosis	MiR-340 mediates the involvement of high mobility group box 1 in the pathogenesis of liver fibrosis
target gene	36948479	hsa-mir-29b	Zika Virus Infection	Zika virus NS1 suppresses VE-cadherin via hsa-miR-29b-3p/DNMT3b/MMP-9 pathway in human brain microvascular endothelial cells.
exosome	36948039	hsa-mir-130b	Wound Healing	ADSCs-derived exosomal H19 accelerates cutaneous wound healing via the miR-130b-3p/PPARγ/STAT3 axis
target gene	36947996	hsa-mir-221	Stomach Neoplasms	Downregulation of miR-221-3p promotes the ferroptosis in gastric cancer cells via upregulation of ATF3 to mediate the transcription inhibition of GPX4 and HRD1.
other	36947456	hsa-mir-497	Pancreatic Neoplasms	Long noncoding RNA CASC9 promotes pancreatic cancer progression by acting as a ceRNA of miR-497-5p to upregulate expression of CCND1.
target gene	36947452	hsa-mir-511	Carcinoma, Non-Small-Cell Lung	Hsa_circ_0003528 promotes cell malignant transformation and immune escape via increasing oncogene PDL1 through sponging miR-511-3p in non-small cell lung cancer.
genetics_knock down_suppress	36946780	hsa-mir-449b	Liver Neoplasms	LncRNA SLC9A3-AS1 knockdown increases the sensitivity of liver cancer cell to triptolide by regulating miR-449b-5p-mediated glycolysis.
target gene	36946346	hsa-mir-155	Neuroblastoma	miR-155-5p alleviates lipopolysaccharide-induced inflammatory damage of human SH-SY5Y neuroblastoma cells by down-regulating SOCS1
target gene	36946310	hsa-mir-6315	Spinal Cord Injuries	miR-6315 silencing protects against spinal cord injury through the Smo and anti-ferroptosis pathway.
target gene	36946075	hsa-mir-145	Uterine Cervical Neoplasms	miR-145 inhibits aerobic glycolysis and cell proliferation of cervical cancer by acting on MYC.
circulation_biomarker_diagnosis_ns	36945942	hsa-mir-21	Hypertension, Pulmonary	Decreased circulating microRNA-21 and microRNA-143 are associated to pulmonary hypertension
```
