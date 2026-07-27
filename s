[33me17eb0a[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m Complete dataset analysis, Fbank, and pretrained ECAPA integration
 .gitignore                            |   42 [32m+[m
 README.md                             |  163 [32m++++[m
 reports/dataset_split_report.txt      |  402 [32m+++++++++[m
 reports/group_relationship_report.txt |  466 [32m+++++++++++[m
 reports/low_utterance_speakers.csv    |  699 [32m++++++++++++++++[m
 reports/recommended_split_report.txt  |   38 [32m+[m
 reports/speaker_distribution.csv      | 1464 [32m+++++++++++++++++++++++++++++++++[m
 reports/speaker_group_overlap.csv     |   11 [32m+[m
 requirements.txt                      |    7 [32m+[m
 src/__init__.py                       |    1 [32m+[m
 src/analyze_manifest.py               | 1255 [32m++++++++++++++++++++++++++++[m
 src/benchmark_fbank.py                |  164 [32m++++[m
 src/build_manifest.py                 | 1243 [32m++++++++++++++++++++++++++++[m
 src/fbank.py                          |  273 [32m++++++[m
 src/fbank_dataset.py                  |  194 [32m+++++[m
 src/inspect_dataset.py                | 1001 [32m++++++++++++++++++++++[m
 src/speechbrain_frontend.py           |  257 [32m++++++[m
 src/test_fbank.py                     |  202 [32m+++++[m
 src/test_pretrained_ecapa.py          |  356 [32m++++++++[m
 19 files changed, 8238 insertions(+)
