"""
 Myung-Joon Kwon
 2023-07-25
"""
from pathlib import Path

project_root = Path(__file__).parent
dataset_root = Path(r"/mnt/server19_hard0/mjkwon/forensicsDB")
Safire_dataset_root = Path(r"/mnt/server4_hard0/wonjun/dataset")
dataset_paths = {
    # Specify where are the roots of the datasets.
    'tampDPR': dataset_root / "tampDPR",

    'FantasticReality': dataset_root / "FantasticReality_v1",
    'IMD2020': dataset_root / "IMD2020",
    'CASIA': dataset_root / "CASIA",
    'tampCOCO': dataset_root / "tampCOCO",

    'NC16': dataset_root / "NC2016_Test",
    'Columbia': dataset_root / "Columbia Uncompressed Image Splicing Detection",
    'COVERAGE': dataset_root / "COVERAGE",
    'CocoGlide' : dataset_root / "CocoGlide",
    'RealTamper' : dataset_root / "realistic-tampering-dataset",

    'SafireMS' : Safire_dataset_root / "SafireMS",
    'SafireDPR' : Safire_dataset_root / "SafireDPR",

    'Arbitrary' : project_root / "inputs",
    'Arbitrary_outputs_binary': project_root / "outputs_binary",
    'Arbitrary_outputs_multi': project_root / "outputs_multi",
}
