# The code includes three inputs for RNN_Neo, and please comment out the other code when using

import subprocess

# ----------------------------------- 原始输入 ----------------------------------- #
# pep——length To be separated by commas 8,9,10,11
length = '9,10'
# patient_id Required parameter
patient_id = 'test1'
# hla MRequired parameter，To be separated by commas
hla = 'HLA-A*02:01'
# path
tumor_dna_1 = './test/SRR2602447_1.fastq.gz'
tumor_dna_2 = './test/SRR2602447_2.fastq.gz'
normal_dna_1 = './test/SRR2602449_1.fastq.gz'
normal_dna_2 = './test/SRR2602449_2.fastq.gz'
tumor_rna_1 = './test/SRR2603944_1.fastq.gz'
tumor_rna_2 = './test/SRR2603944_2.fastq.gz'
subprocess.run([
    'python',
    'RNN_Neo.py',
    '-l',length,
    '--id',patient_id,
    '--hla',hla,
    '--tumor_dna_1',tumor_dna_1,
    '--tumor_dna_2',tumor_dna_2,
    '--normal_dna_1',normal_dna_1,
    '--normal_dna_2',normal_dna_2,
    '--tumor_rna_1',tumor_rna_1,
    '--tumor_rna_2',tumor_rna_2
])

# ----------------------------------- pep输入 ---------------------------------- #

# patient_id = 'test2'
# hla = 'HLA-A*02:01,HLA-C*07:02'

# pep_path = './test/pep.csv'
# subprocess.run([
#     'python',
#     'RNN_Neo.py',
#     '--id',patient_id,
#     '--hla',hla,
#     '-p',pep_path
# ])

# ------------------------------ snp indel vcf输入 ----------------------------- #
# patient_id = 'test3'
# hla = 'HLA-A*02:01,HLA-C*07:02'
# snv = './test/snv.csv'
# indel = './test/indel.csv'
# tpm = './test/tumor_gene.csv'

# subprocess.run([
#     'python',
#     'RNN_Neo.py',
#     '--id',patient_id,
#     '--hla',hla,
#     '-g',tpm,
#     '--snv',snv,
#     '--indel',indel
# ])