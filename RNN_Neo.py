import sys, time, os, json, re
import numpy as np
import torch
import pandas as pd
import glob
from Bio import Entrez
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
import argparse
import subprocess
from Bio import SeqIO
import os
import multiprocessing
import shutil

def get_parameter():
    with open("parameter.json") as f:
        global parameter
        parameter = json.load(f)
        
def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def write_time_log(f_time, A, s):
    B = time.time()
    C = B - A
    C = int(C)
    f_time.write(s + str(C // 60) + '分' + str(C % 60) + '秒\n')
    f_time.flush()



def wes2mut(thread,patient_id,tumor1,tumor2,normal1,normal2):
    wes_path = parameter['work_path'] + '/calspace/' + patient_id + '/1_wes'
    create_path(wes_path)

    print(f'tumor_dna_1:{tumor1}')
    print(f'tumor_dna_2:{tumor2}')
    print(f'noemal_dna_1:{normal1}')
    print(f'noemal_dna_2:{normal2}')

    if not os.path.exists(normal1):
        print('Please confirm the wes data')
    if not os.path.exists(normal2):
        print('Please confirm the wes data')
    if not os.path.exists(tumor1):
        print('Please confirm the wes data')
    if not os.path.exists(tumor2):
        print('Please confirm the wes data')

    f_time = open(wes_path + '/wes_time.log', 'w+')

    f_time.write('*****wes to mutation start!*****\n')
    f_time.flush()
    A = time.time()

    # trimmomatic
    os.system('trimmomatic PE -threads '+ str(thread) + ' -phred33 ' + ' ' + normal1 + ' ' + normal2 + ' ' + wes_path + '/normal1.fq.gz ' + wes_path + '/trim_normal1.fq.gz' + ' ' + wes_path + '/normal2.fq.gz ' + wes_path + '/trim_normal2.fq.gz' + ' ILLUMINACLIP:' + parameter['ref_path'] + '/TruSeq3-PE.fa:2:30:10 SLIDINGWINDOW:5:20 LEADING:5 TRAILING:5 MINLEN:50')
    os.system('trimmomatic PE -threads '+ str(thread) + ' -phred33 ' + ' ' + tumor1 + ' ' + tumor2 + ' ' + wes_path + '/tumor1.fq.gz ' + wes_path + '/trim_tumor1.fq.gz' + ' ' + wes_path + '/tumor2.fq.gz ' + wes_path + '/trim_tumor2.fq.gz' + ' ILLUMINACLIP:' + parameter['ref_path'] + '/TruSeq3-PE.fa:2:30:10 SLIDINGWINDOW:5:20 LEADING:5 TRAILING:5 MINLEN:50')

    write_time_log(f_time, A, 'step1_trimmomatic_')

    # bwa
    os.system("bwa mem -t " + str(thread) + " -M -R '@RG\\tID:normal\\tSM:normal\\tLB:normal\\tPL:illumina' " + parameter['ref_path'] + "/hg38.fa " + wes_path + '/normal1.fq.gz ' + wes_path + '/normal2.fq.gz ' + '> ' + wes_path + '/normal.sam')
    os.system("bwa mem -t " + str(thread) + " -M -R '@RG\\tID:tumor\\tSM:tumor\\tLB:tumor\\tPL:illumina' " + parameter['ref_path'] + "/hg38.fa " + wes_path + '/tumor1.fq.gz ' + wes_path + '/tumor2.fq.gz ' + '> ' + wes_path + '/tumor.sam')

    write_time_log(f_time, A, 'step2_bwa_')

    #  sortsam
    os.system('gatk --java-options "-Xmx4G" SortSam -SO coordinate ' + '-I ' + wes_path + '/normal.sam -O ' + wes_path + '/normal_sorted.bam')
    os.system('gatk --java-options "-Xmx4G" SortSam -SO coordinate ' + '-I ' + wes_path + '/tumor.sam -O ' + wes_path + '/tumor_sorted.bam')

    write_time_log(f_time, A, 'step3_sortsam_')

    # MarkDuplicatesAndMerge
    os.system('gatk --java-options "-Xmx4G" MarkDuplicates ' + '-I ' + wes_path + '/tumor_sorted.bam ' + '-O ' + wes_path + '/tumor_sorted_marked.bam ' + '-M ' + wes_path + '/tumor_sorted_marked.metrics')
    os.system('gatk --java-options "-Xmx4G" MarkDuplicates ' + '-I ' + wes_path + '/normal_sorted.bam ' + '-O ' + wes_path + '/normal_sorted_marked.bam ' + '-M ' + wes_path + '/normal_sorted_marked.metrics')
    os.system('samtools index ' + wes_path + '/normal_sorted_marked.bam')
    os.system('samtools index ' + wes_path + '/tumor_sorted_marked.bam')

    write_time_log(f_time, A, 'step4_MarkDuplicatesAndMerge_')

    # bqsr
    os.system('gatk --java-options ' + '"-Xmx4G"' + ' BaseRecalibrator ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-I ' + wes_path + '/tumor_sorted_marked.bam ' + '--known-sites  ' + parameter['annotation_path'] + '/1000G_phase1.snps.high_confidence.hg38.vcf.gz ' + '--known-sites  ' + parameter['annotation_path'] + '/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz ' + '--known-sites  ' + parameter['annotation_path'] + '/dbsnp_138.hg38.vcf.gz ' + '-O ' + wes_path + '/tumor_sorted_marked_temp.table')
    os.system('gatk --java-options "-Xmx4G" ApplyBQSR ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-I ' + wes_path + '/tumor_sorted_marked.bam ' + '-O ' + wes_path + '/tumor_sorted_marked.recal.bam ' + '-bqsr ' + wes_path + '/tumor_sorted_marked_temp.table')

    os.system('gatk --java-options ' + '"-Xmx4G"' + ' BaseRecalibrator ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-I ' + wes_path + '/normal_sorted_marked.bam ' + '--known-sites  ' + parameter['annotation_path'] + '/1000G_phase1.snps.high_confidence.hg38.vcf.gz ' + '--known-sites  ' + parameter['annotation_path'] + '/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz ' + '--known-sites  ' + parameter['annotation_path'] + '/dbsnp_138.hg38.vcf.gz ' + '-O ' + wes_path + '/normal_sorted_marked_temp.table')
    os.system('gatk --java-options "-Xmx4G" ApplyBQSR ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-I ' + wes_path + '/normal_sorted_marked.bam ' + '-O ' + wes_path + '/normal_sorted_marked.recal.bam ' + '-bqsr ' + wes_path + '/normal_sorted_marked_temp.table')

    write_time_log(f_time, A, 'step5_bqsr_')
    # Mutect2
    os.system('gatk --java-options "-Xmx4G" Mutect2 -R ' + parameter['ref_path'] + '/hg38.fa -I ' + wes_path + '/tumor_sorted_marked.recal.bam ' + '-I ' + wes_path + '/normal_sorted_marked.recal.bam -tumor tumor -normal normal --germline-resource  ' + parameter['annotation_path'] + '/af-only-gnomad.hg38.vcf.gz --af-of-alleles-not-in-resource 0.0000025 --disable-read-filter MateOnSameContigOrNoMappedMateReadFilter --native-pair-hmm-threads ' + str(thread) + ' -O ' + wes_path + '/somatic_mutect2.vcf.gz')

    write_time_log(f_time, A, 'step6_Mutect2_')

    # EstimateCrossSampleContamination(wes_path)
    os.system('gatk --java-options "-Xmx4G" GetPileupSummaries ' + '-I ' + wes_path + '/tumor_sorted_marked.recal.bam ' + '-L  ' + parameter['annotation_path'] + '/small_exac_common_3.hg38.vcf.gz ' + '-V  ' + parameter[ 'annotation_path'] + '/small_exac_common_3.hg38.vcf.gz ' + '-O ' + wes_path + '/tumor_getpileupsummaries.table')
    os.system('gatk --java-options "-Xmx4G" CalculateContamination ' + '-I ' + wes_path + '/tumor_getpileupsummaries.table ' + '-O ' + wes_path + '/tumor_calculatecontamination.table')

    write_time_log(f_time, A, 'step7_EstimateCrossSampleContamination_')

    # FilterForConfidentSomaticCalls
    os.system('gatk --java-options "-Xmx4G" FilterMutectCalls ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-V ' + wes_path + '/somatic_mutect2.vcf.gz ' + '--contamination-table ' + wes_path + '/tumor_calculatecontamination.table ' + '-O ' + wes_path + '/tumor_somatic_mutect2_oncefiltered.vcf.gz')

    write_time_log(f_time, A, 'step8_FilterForConfidentSomaticCalls_')

    # selectPassedSNPselectPassedINDEL
    os.system('gatk --java-options "-Xmx4G" SelectVariants ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-V ' + wes_path + '/tumor_somatic_mutect2_oncefiltered.vcf.gz ' + '-O ' + wes_path + '/tumor_oncefiltered_pass_snp.vcf ' + '--exclude-filtered true ' + '-select-type SNP')
    os.system('gatk --java-options "-Xmx4G" SelectVariants ' + '-R ' + parameter['ref_path'] + '/hg38.fa ' + '-V ' + wes_path + '/tumor_somatic_mutect2_oncefiltered.vcf.gz ' + '-O ' + wes_path + '/tumor_oncefiltered_pass_indel.vcf ' + '--exclude-filtered true ' + '-select-type INDEL')

    write_time_log(f_time, A, 'step9_selectPassed_')

    # annotatePassedSNP
    os.system('perl ' + parameter['annovar_path'] + '/table_annovar.pl ' + '' + wes_path + '/tumor_oncefiltered_pass_snp.vcf ' + '' + parameter['annovar_path'] + '/humandb ' + '--buildver hg38 ' + '-out ' + wes_path + '/tumor_pass_snp ' + '--protocol refGene ' + '--operation g ' + '--nastring . ' + '--vcfinput')
    # annotatePassedINDEL
    os.system('perl ' + parameter['annovar_path'] + '/table_annovar.pl ' + '' + wes_path + '/tumor_oncefiltered_pass_indel.vcf ' + '' + parameter['annovar_path'] + '/humandb ' + '--buildver hg38 ' + '-out ' + wes_path + '/turmor_pass_indel ' + '--protocol refGene ' + '--operation g ' + '--nastring . ' + '--vcfinput')

    write_time_log(f_time, A, 'step11_annovar_')

    # delete
    f1 = wes_path + '/*normal*'
    f2 = wes_path + '/*trim*'
    f3 = wes_path + '/*.table'
    f4 = wes_path + '/*.sam'
    f6 = wes_path + '/*hg38_multianno*'
    f7 = wes_path + '/*tumor_sorted*'
    f8 = wes_path + '/*.avinput*'
    f9 = wes_path + '/*.orig'
    f10 = wes_path + '/*.fa'
    f11 = wes_path + '/*refGene.log'
    f12 = wes_path + '/*refGene.variant_function'
    f13 = wes_path + '/*.fq.gz'
    f14 = wes_path + '/somatic_mutect2.vcf.gz*'
    f15 = wes_path + '/tumor_oncefiltered_pass*'
    os.system('rm -rf ' + f1 + ' ' + f2 + ' ' + f3 + ' ' + f4 + ' ' + f6 + ' ' + f7 + ' ' + f8)
    os.system('rm -rf ' + f9 + ' ' + f10 + ' ' + f11 + ' ' + f12 + ' ' + f13 + ' ' + f14 + ' ' + f15)

    f_time.write('Call mutation finish!\n')
    f_time.flush()
    f_time.close()

def rna2tpm(thread,patient_id,tumor1,tumor2):
    rna_path = parameter['work_path'] + '/calspace/' + patient_id + '/2_rna'
    create_path(rna_path)

    print(f'tumor_rna_1:{tumor1}')
    print(f'tumor_rna_2:{tumor2}')

    if not os.path.exists(tumor1):
        print('Please confirm the rna-seq data')
    if not os.path.exists(tumor2):
        print('Please confirm the rna-seq data')

    f_time = open(rna_path + '/rna_time.log', 'w+')

    f_time.write('*****rna to tpm start!*****\n')
    f_time.flush()
    A = time.time()

    i1 = tumor1
    i2 = tumor2

    o1 = rna_path+'/R1.clean.fastq.gz'
    o2 = rna_path+'/R2.clean.fastq.gz'
    # fastp
    subprocess.run(args=['fastp','-i',i1,'-I',i2,'-o',o1,'-O',o2,'-w',str(thread)])
    write_time_log(f_time, A, 'step1_fastp_')

    i1 = o1
    i2 = o2

    # kallisto index
    idx = parameter['ref_path']+'/Homo_sapiens.GRCh38.cdna.all.index'
    i = parameter['ref_path']+'/Homo_sapiens.GRCh38.cdna.all.fa.gz'
    if not os.path.exists(idx):
        subprocess.run(args=['kallisto','index',i,'-i',idx])

    o = rna_path
    # kallisto 
    subprocess.run(args=['kallisto','quant','-i',parameter['ref_path']+'/Homo_sapiens.GRCh38.cdna.all.index','-o',o,'-t',str(thread),i1,i2])
    write_time_log(f_time, A, 'step2_kallisto_')

    # id merge
    tpm = rna_path + '/abundance.tsv'
    df = pd.read_csv(tpm, sep='\t')
    df = df[df['tpm'] != 0]
    ensembl_ids = list(df['target_id'])
    df['gene_name'] = get_id(ensembl_ids,'transcript_id','gene_name',parameter['ref_path']+'/hg38.gtf')
    df = df.dropna()
    df = df[['gene_name','tpm']]
    df.columns = ['gene_name', 'gene_tpm']
    df.to_csv(rna_path+'/tumor_gene.csv',index=False)
    write_time_log(f_time, A, 'step3_trans_id_')

    f1 = os.path.join(rna_path, '*.clean.fastq.gz')
    print(f1)
    os.system('rm -rf ' +f1)

    f_time.write('rna2tpm finish!\n')
    f_time.flush()
    f_time.close()

def get_id(id_list,id_from,trans_to,gtf_file):
    with open(gtf_file,'r') as gff:
        result = []
        hash_dict = dict()
        for line in gff:
            line1=line.strip().split('\t',8)
            try:
                Name = line1[8]
            except:
                continue
            try:
                from_type = eval(Name.split(id_from)[1].split(';')[0])
                #hash methods, it need a large memory!!
                to_type = eval(Name.split(trans_to)[1].split(';')[0])
                hash_dict[from_type] = to_type
            except:
                continue
        for item in id_list:
            try:
                result.append(hash_dict[item])
            except:
                result.append(None)
                continue
        return(result)



def get_snv_pep(pep_length,hla_list,patient_id,n=10,snv=None):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    create_path(neo_path)
    snv_path = neo_path + '/snv'
    create_path(snv_path)
    rna_path = parameter['work_path'] + '/calspace/' + patient_id + '/2_rna'
    wes_path = parameter['work_path'] + '/calspace/' + patient_id + '/1_wes'

    if not snv:
        snv_mut_vcf_anno = wes_path + '/tumor_pass_snp.refGene.exonic_variant_function'
        snv_mut_vcf_anno_screened = snv_path + '/snv_screened_muts.txt'
        snv_file_fo = open(snv_mut_vcf_anno, 'r')
        snv_file_fo_lines = snv_file_fo.readlines()
        snv_file_fo.close()
        muttype_mutinfo_ls = [x.strip().split('\t')[1:3] for x in snv_file_fo_lines]
        muttype_mutinfo_ls = [[x[0], x[1].split(',')[0]] for x in muttype_mutinfo_ls]

        muttype_mutinfo_filtered_ls = [x for x in muttype_mutinfo_ls if x[0] == 'nonsynonymous SNV']

        muttype_mutinfo_filtered_ls = [x for x in muttype_mutinfo_filtered_ls if not 'X' in x[1].split(':')[-1]]
        fw = open(snv_mut_vcf_anno_screened, 'w+')
        for i in muttype_mutinfo_filtered_ls:
            i_word = '\t'.join(i) + '\n'
            fw.write(i_word)
            fw.flush()
        fw.close()
    else:
        pass


    def get_seq(email, id_ls):
        Entrez.email = email  
        id_ls = list(set(id_ls))
        id_str = ",".join(id_ls)
        print('gene_length {}'.format(len(id_ls)))
        handle = Entrez.efetch(db="nucleotide", id=id_str, rettype="fasta_cds_aa", retmode="text")
        id_seq_dict = {}
        for line in handle:
            if line.startswith(">"):
                seq_id = line.strip()
                id_seq_dict[seq_id] = ''
            else:
                id_seq_dict[seq_id] += line.strip()
        handle.close()    
        return id_seq_dict 

    def id2df(email,id_ls):
        fa = get_seq(email,id_ls)
        df = pd.DataFrame(list(fa.items()),columns=['info','seq'])   
        nm_pattern = r"(NM_\d+)"
        gene_pattern = r"\[gene=(\w+)\]"
        df['nm_id'] = df['info'].str.extract(nm_pattern)
        df['gene'] = df['info'].str.extract(gene_pattern)
        df = df[['nm_id','gene','seq']]
        df['len']=df['seq'].str.len()
        max_len_indices = df.groupby(['nm_id', 'gene'])['len'].idxmax()
        df = df.loc[max_len_indices]
        df = df[['nm_id','seq']]
        print('download seq {}'.format(len(df)))
        return df
    
    # download
    Entrez.email = "1911013@tongji.edu.cn"
    snv_mut_vcf_anno_screened = snv_path +'/snv_screened_muts.txt'
    fr = open(snv_mut_vcf_anno_screened)
    lines = fr.readlines()
    fr.close()
    lines = [x.strip().split('\t') for x in lines]


    nm_ls = []
    for j in lines:
        nmid = j[1].split(":")[1]
        if not nmid in nm_ls:
            nm_ls.append(nmid)


    email = '1911013@tongji.edu.cn'
    info = id2df(email,nm_ls)
    info.to_csv(snv_path + '/protein_fasta.info.txt',sep='\t', header=False, index=False)
    print('Protein sequence is now available')

    ##
    snv_mut_vcf_anno_screened = snv_path +'/snv_screened_muts.txt'
    fr = open(snv_mut_vcf_anno_screened)
    lines = fr.readlines()
    fr.close()
    muttype_mutinfo_filtered_ls = [x.strip().split('\t') for x in lines]

    nm_pro_dict = {}
    with open(snv_path+'/protein_fasta.info.txt') as fo:
        nm_fa_lines = fo.readlines()
        for nm_fa_line in nm_fa_lines:
            nm_fa_line = nm_fa_line.strip().split('\t')
            nm_pro_dict[nm_fa_line[0]] = nm_fa_line[1]
    info = []
    for i in muttype_mutinfo_filtered_ls:
        mut_info = i[1].split(':')
        nmid = mut_info[1]
        protein = ''
        s = int(re.findall('\d+',mut_info[-1])[0])
        neo_aa = mut_info[-1][-1]
        if nmid in nm_pro_dict:
            protein = nm_pro_dict[nmid]
        for j in pep_length:
            if protein:
                pep_raw , pep_neo =  snv_n(protein, s, neo_aa, int(j)-1)
                _ , pep_neo_long =  snv_n(protein, s, neo_aa, int(j)+n-1)
                pep_raw_slide = slide_window(pep_raw,int(j))
                pep_neo_slide = slide_window(pep_neo,int(j))

            for raw, neo in zip(pep_raw_slide,pep_neo_slide):
                long_pep, start, end= extract_sequence(neo, pep_neo_long, n)
                info.append([i[0], i[1], raw, neo, long_pep,str(start), str(end), str(j)])
            
            

    mut_pep_info = snv_path+ '/mut_pep_info.txt'
    fw = open(mut_pep_info,'w+')


    for i in info:
        fw.write('\t'.join(i)+'\n')
        fw.flush()
    fw.close()



    gene_tpm = pd.read_csv(rna_path + '/tumor_gene.csv')
    dict_gene_tpm = gene_tpm.groupby('gene_name')['gene_tpm'].mean().to_dict()


    hla = ';'.join(hla_list)


    all_in_one_ls = []
    with open(snv_path + '/mut_pep_info.txt','r') as fo_mut_pep_info:
        mut_pep_info = fo_mut_pep_info.readlines()
    mut_pep_info = [x.strip().split('\t') for x in mut_pep_info]
    dict_pepid_info = {}
    for i in mut_pep_info:
        Mut_protein=i[3]
        Raw_protein=i[2]
        long_mut_protein=i[4]
        geneid = i[1].split(':')[0]
        length = i[-1]
        start = i[-3]
        end = i[-2]
        #nmid = i[1].split(':')[1]
        #mut_type = i[0]
        if geneid in dict_gene_tpm:
            te = dict_gene_tpm[geneid]
        else:
            te = 0.000
        all_in_one_ls.append([Mut_protein,Raw_protein,long_mut_protein,start,end,geneid,str(te),hla,length])

    fw = open(snv_path + '/neo_snv_result.txt','w+')
    title = ['Pep_seq','Raw_Pep_seq','long_mut_protein','start','end','Gene','Gene_tpm','hla','length']
    fw.write('\t'.join(title)+'\n')
    for i in all_in_one_ls:
        fw.write('\t'.join(str(j) for j in i)+'\n')
        fw.flush()
    fw.close()

def get_indel_pep(pep_length,hla_list,patient_id,n=10,indel=None):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    create_path(neo_path)
    indel_path = neo_path + '/indel'
    create_path(indel_path)
    rna_path = parameter['work_path'] + '/calspace/' + patient_id + '/2_rna'
    wes_path = parameter['work_path'] + '/calspace/' + patient_id + '/1_wes'

    if not indel:
        indel_mut_vcf_anno = wes_path +'/turmor_pass_indel.refGene.exonic_variant_function'
        indel_mut_vcf_anno_screened = indel_path +'/indel_screened_muts.txt'

        indel_file_fo = open(indel_mut_vcf_anno,'r')
        indel_file_fo_lines = indel_file_fo.readlines()
        indel_file_fo.close()
        muttype_mutinfo_ls = [x.strip().split('\t')[1:3] for x in indel_file_fo_lines]
        muttype_mutinfo_ls = [[x[0],x[1].split(',')[0]] for x in muttype_mutinfo_ls]

        muttype_mutinfo_filtered_ls = [x for x in muttype_mutinfo_ls if not x[0] in ['unknown','startloss','stopgain']]


        muttype_mutinfo_filtered_ls = [x for x in muttype_mutinfo_filtered_ls if not 'X' in x[1].split(':')[-1]]


        fw = open(indel_mut_vcf_anno_screened,'w+')
        for i in muttype_mutinfo_filtered_ls:
            i_word = '\t'.join(i)+'\n'
            fw.write(i_word)
            fw.flush()
        fw.close()
    else:
        pass

    Entrez.email = "1911013@tongji.edu.cn"
    indel_mut_vcf_anno_screened = indel_path +'/indel_screened_muts.txt'
    fr = open(indel_mut_vcf_anno_screened)
    lines = fr.readlines()
    fr.close()
    lines = [x.strip().split('\t') for x in lines]

    def get_seq(email, id_ls):
        Entrez.email = email  
        id_ls = list(set(id_ls))
        id_str = ",".join(id_ls)
        print('gene_length {}'.format(len(id_ls)))
        handle = Entrez.efetch(db="nucleotide", id=id_str, rettype="fasta_cds_aa", retmode="text")
        id_seq_dict = {}
        for line in handle:
            if line.startswith(">"):
                seq_id = line.strip()
                id_seq_dict[seq_id] = ''
            else:
                id_seq_dict[seq_id] += line.strip()
        handle.close()    
        return id_seq_dict 

    def id2df(email,id_ls):
        fa = get_seq(email,id_ls)
        df = pd.DataFrame(list(fa.items()),columns=['info','seq'])   
        nm_pattern = r"(NM_\d+)"
        gene_pattern = r"\[gene=(\w+)\]"
        df['nm_id'] = df['info'].str.extract(nm_pattern)
        df['gene'] = df['info'].str.extract(gene_pattern)
        df = df[['nm_id','gene','seq']]
        df['len']=df['seq'].str.len()
        max_len_indices = df.groupby(['nm_id', 'gene'])['len'].idxmax()
        df = df.loc[max_len_indices]
        df = df[['nm_id','seq']]
        print('download seq {}'.format(len(df)))
        return df
    

    nm_ls = []
    for j in lines:
        nmid = j[1].split(":")[1]
        if not nmid in nm_ls:
            nm_ls.append(nmid)
            
    email = '1911013@tongji.edu.cn'
    info = id2df(email,nm_ls)
    info.to_csv(indel_path + '/protein_fasta.info.txt',sep='\t', header=False, index=False)
    print('Protein sequence is now available')


    def get_seq(email, id_ls):
        Entrez.email = email  
        id_ls = list(set(id_ls))
        id_str = ",".join(id_ls)
        print('gene_length {}'.format(len(id_ls)))
        handle = Entrez.efetch(db="nucleotide", id=id_str, rettype="fasta_cds_na", retmode="text")
        id_seq_dict = {}
        for line in handle:
            if line.startswith(">"):
                seq_id = line.strip()
                id_seq_dict[seq_id] = ''
            else:
                id_seq_dict[seq_id] += line.strip()
        handle.close()    
        return id_seq_dict 

    def id2df(email,id_ls):
        fa = get_seq(email,id_ls)
        df = pd.DataFrame(list(fa.items()),columns=['info','seq'])   
        nm_pattern = r"(NM_\d+)"
        gene_pattern = r"\[gene=(\w+)\]"
        df['nm_id'] = df['info'].str.extract(nm_pattern)
        df['gene'] = df['info'].str.extract(gene_pattern)
        df = df[['nm_id','gene','seq']]
        df['len']=df['seq'].str.len()
        max_len_indices = df.groupby(['nm_id', 'gene'])['len'].idxmax()
        df = df.loc[max_len_indices]
        df = df[['nm_id','seq']]
        print('download seq {}'.format(len(df)))
        return df
    
    email = '1911013@tongji.edu.cn'
    info = id2df(email,nm_ls)
    info.to_csv(indel_path + '/gene_fasta.info.txt',sep='\t', header=False, index=False)
    print('Gene sequence is now available')

    indel_mut_vcf_anno_screened = indel_path +'/indel_screened_muts.txt'
    fr = open(indel_mut_vcf_anno_screened)
    lines = fr.readlines()
    fr.close()
    muttype_mutinfo_filtered_ls = [x.strip().split('\t') for x in lines]


    nm_pro_dict = {}
    with open(indel_path+'/protein_fasta.info.txt') as fo:
        nm_fa_lines = fo.readlines()
        for nm_fa_line in nm_fa_lines:
            nm_fa_line = nm_fa_line.strip().split('\t')
            nm_pro_dict[nm_fa_line[0]] = nm_fa_line[1]

    nm_rna_dict = {}
    with open(indel_path+'/gene_fasta.info.txt') as fo:
        nm_fa_lines = fo.readlines()
        for nm_fa_line in nm_fa_lines:
            nm_fa_line = nm_fa_line.strip().split('\t')
            nm_rna_dict[nm_fa_line[0]] = nm_fa_line[1]

    info = []
    for j in muttype_mutinfo_filtered_ls:
        

        if j[0] in ['nonframeshift insertion', 'nonframeshift deletion', 'stoploss']:
            

            mut_info = j[1].split(':')
            if mut_info[1] in nm_pro_dict:
                protein = nm_pro_dict[mut_info[1]]
            else:
                protein = ''
                print(mut_info[1]+' has no seq !!')
            

            if j[0] == 'nonframeshift insertion' and 'delins' not in mut_info[-1]:
                

                if ('ins' not in mut_info[-1]) or ('_' not in mut_info[-1]) or ('*' in mut_info[-1]):
                    print(j)
                

                else:
                    ins_seq = mut_info[-1].split('ins')[-1]
                    s = re.findall('\d+',mut_info[-1])
                    s0 = int(s[0])
                    s1 = int(s[1])
                    
                    protein_mutted = protein[:s0] + ins_seq + protein[s1-1:]
                    pep_space = ''
                    
                    if s0 < 8:
                        start = 0
                    else:
                        start = s0-8
                    
                    if s1 + 7 > len(protein):
                        end = len(protein)
                    else:
                        end = s1 + 7
                    
                    ins_site = str(s0) + ',' + str(s1)
                    pep_space = protein[start:s0] + ins_seq + protein[s1-1:end]
                    
                    info.append(j+[protein,protein_mutted,ins_site,pep_space])
                    # print(pep_space)
                    

            elif j[0] == 'nonframeshift deletion' and 'delins' not in mut_info[-1]:
                    

                if '_' not in mut_info[-1]:
                    
                    s = int(re.findall('\d+',mut_info[-1])[0])
        
                    if s < 9:
                        start = 0
                    else:
                        start = s-9
                    
                    if s + 8 >= len(protein):
                        end = len(protein)
                    else:
                        end = s + 8
                    
                    protein_mutted = protein[:s-1] + protein[s:]
                    pep_space = ''
                    
                    del_site = str(s)
                    pep_space = protein[start:s-1] + protein[s:end]
                    
                    info.append(j+[protein,protein_mutted,del_site,pep_space])
                    # print(pep_space)
                

                else:
                    
                    s = re.findall('\d+',mut_info[-1])
                    
                    s0 = int(s[0])
                    s1 = int(s[1])
                    
                    if s0 < 9:
                        start = 0
                    else:
                        start = s0-9
                    
                    if s1 + 8 >= len(protein):
                        end = len(protein)
                    else:
                        end = s1 + 8
                    
                    protein_mutted = protein[:s0-1] + protein[s1:]
                    pep_space = ''
                    
                    del_site = str(s0) + ',' + str(s1)
                    pep_space = protein[start:s0-1] + protein[s1:end]
                    
                    info.append(j+[protein,protein_mutted,del_site,pep_space])
                    # print(pep_space)
            

            elif j[0] == 'stoploss':
                

                if '_' not in mut_info[-1] and 'delins' in mut_info[-1]:
                    ins_seq = mut_info[-1].split('ins')[-1]
                    ins_seq = ins_seq[:-1]
                    protein_mutted = protein + ins_seq
                    s = int(re.findall('\d+',mut_info[-1])[0])
                    change_site = str(s)
                    if len(protein) > 8:
                        pep_space = protein[-8:] + ins_seq
                    else:
                        pep_space = protein + ins_seq
                    info.append(j+[protein,protein_mutted,change_site,pep_space])
                    # print(j,pep_space)
            
  
                elif '_' in mut_info[-1]:
                    ins_seq = mut_info[-1].split('ins')[-1]
                    ins_seq = ins_seq[:-1]
                    protein_mutted = protein + ins_seq
                    s = int(re.findall('\d+',mut_info[-1])[0])
                    change_site = str(s)
                    if len(protein) > 8:
                        pep_space = protein[-8:] + ins_seq
                    else:
                        pep_space = protein + ins_seq
                    info.append(j+[protein,protein_mutted,change_site,pep_space])
                    # print(j,pep_space)
                
                else:
                    print(j,'error')
                    continue
                
            

            elif j[0] in ['nonframeshift insertion','nonframeshift deletion'] and 'delins' in mut_info[-1]:
                

                if "_" in mut_info[-1]:
                    s = re.findall('\d+',mut_info[-1])
                    s0 = int(s[0])
                    s1 = int(s[1])
                    
                    ins = mut_info[-1].split('delins')[1]
                    
                    if ins[-1] == '*':
                        ins = ins[:-1]
                    
                    if s0 < 9:
                        start = 0
                    else:
                        start = s0-9
                    
                    if s1 + 8 >= len(protein):
                        end = len(protein)
                    else:
                        end = s1 + 8
                    
                    protein_mutted = protein[:s0-1] + ins + protein[s1:]
                    
                    del_site = str(s0) + ',' + str(s1)
                    pep_space = protein[start:s0-1] + ins + protein[s1:end]
                    
                    info.append(j+[protein,protein_mutted,del_site,pep_space])
                

                else:
                    s = int(re.findall('\d+',mut_info[-1])[0])
                    ins = mut_info[-1].split('delins')[1]
                    
                    if s < 9:
                        start = 0
                    else:
                        start = s-9
                    
                    if s + 8 >= len(protein):
                        end = len(protein)
                    else:
                        end = s + 8
                    
                    protein_mutted = protein[:s-1] + ins + protein[s:]
                    del_site = str(s)
                    pep_space = protein[start:s-1] + ins + protein[s:end]
                    
                    info.append(j+[protein,protein_mutted,del_site,pep_space])


            else:
                print('other')
                print(j)
                continue
                

        elif j[0] in ['frameshift insertion', 'frameshift deletion']:
            mut_info = j[1].split(':')
            if mut_info[1] in nm_pro_dict and mut_info[1] in nm_rna_dict:
                protein = nm_pro_dict[mut_info[1]]
                cdna = nm_rna_dict[mut_info[1]]
                

                if j[0] == 'frameshift insertion':
                    
                    ### dup型
                    if 'dup' in mut_info[3]:
                        
                        dup_base = mut_info[3].split('dup')[-1]
                        dup_cDNA_site = mut_info[3].split('dup')[0].split('.')[-1]
                        dup_cDNA_site = int(dup_cDNA_site)
                        

                        if cdna[dup_cDNA_site-1] == dup_base:
                            
                            cDNA_new = cdna[:dup_cDNA_site]+dup_base+cdna[dup_cDNA_site:]
                            protein_old = protein
                            protein_new = translate_cDNA_to_protein(cDNA_new)
                            
                            for k in range(min(len(protein_old),len(protein_new))):
                                if protein_old[k] != protein_new[k]:
                                    change_site = k+1
                                    raw_aa = protein_old[k]
                                    new_aa = protein_new[k]
                                    break
                            
                            if change_site > 9:
                                pep_space = protein_new[change_site-9:]
                            else:
                                pep_space = protein_new
                            
                            info.append(j+[protein_old,protein_new,str(change_site),pep_space])
                                                            
                        else:
                            print('base info error: '+str(j))
                            continue
                    
      
                    elif '_' in mut_info[3]:
                        ins_base_site = int(mut_info[3].split('.')[1].split('_')[0])
                        ins_base = mut_info[3].split('ins')[1]
                        cDNA_new = cdna[:ins_base_site]+ins_base+cdna[ins_base_site:]
                        
                        protein_old = protein
                        protein_new = translate_cDNA_to_protein(cDNA_new)
                        
                        for k in range(min(len(protein_old),len(protein_new))):
                            if protein_old[k] != protein_new[k]:
                                change_site = k+1
                                raw_aa = protein_old[k]
                                new_aa = protein_new[k]
                                break
                            
                        if change_site > 9:
                            pep_space = protein_new[change_site-9:]
                        else:
                            pep_space = protein_new
                        
                        info.append(j+[protein_old,protein_new,str(change_site),pep_space])
                    

                    else:
                        print('unknow type: '+str(j))
                    

                if j[0] == 'frameshift deletion':
                    
 
                    if '_' not in mut_info[3]:
                        del_base_site = int(mut_info[3].split('.')[1].split('del')[0])
                        ins_base = mut_info[3].split('del')[1]
                        cDNA_new = cdna[:del_base_site-1]+cdna[del_base_site:]
                        
                        protein_old = protein
                        protein_new = translate_cDNA_to_protein(cDNA_new)
                        
                        for k in range(min(len(protein_old),len(protein_new))):
                            if protein_old[k] != protein_new[k]:
                                change_site = k+1
                                raw_aa = protein_old[k]
                                new_aa = protein_new[k]
                                break
                            
                        if change_site > 9:
                            pep_space = protein_new[change_site-9:]
                        else:
                            pep_space = protein_new
                        
                        info.append(j+[protein_old,protein_new,str(change_site),pep_space])
                        # print(change_site,pep_space)
                    

                    else:
                        del_start_site = mut_info[3].split('.')[1].split('del')[0].split('_')[0]
                        del_end_site = mut_info[3].split('.')[1].split('del')[0].split('_')[1]
                        
                        del_start_site = int(del_start_site)
                        del_end_site = int(del_end_site)
                        
                        cDNA_new = cdna[:del_start_site-1]+cdna[del_end_site:]
                        
                        protein_old = protein
                        protein_new = translate_cDNA_to_protein(cDNA_new)
                        
                        for k in range(min(len(protein_old),len(protein_new))):
                            if protein_old[k] != protein_new[k]:
                                change_site = k+1
                                raw_aa = protein_old[k]
                                new_aa = protein_new[k]
                                break
                            
                        if change_site > 9:
                            pep_space = protein_new[change_site-9:]
                        else:
                            pep_space = protein_new
                        
                        info.append(j+[protein_old,protein_new,str(change_site),pep_space])
                        # print(change_site,pep_space)
                        # print(j)
            else:
                # print(mut_info[1]+' has no seq !!')
                continue
        

        else:
            print('other mut type: '+str(j))


    mut_pep_info = indel_path +'/mut_pep_info.txt'
    fw = open(mut_pep_info,'w+')

    info_add = []
    for i in range(len(info)):
        info_add.append(info[i]+['peptide'+str(i+1)])

    for i in info_add:
        fw.write('\t'.join(i)+'\n')
        fw.flush()
    fw.close()


    with open(indel_path+'/mut_pep_info.txt','r') as fo_mut_pep_info:
        mut_pep_info = fo_mut_pep_info.readlines()
    mut_pep_info = [x.strip().split('\t') for x in mut_pep_info]


    info = []
    for i in mut_pep_info:
        neo = i[5]
        long = i[3]
        for j in pep_length:
            pep_neo_slide = slide_window(neo,int(j))
            for slide_neo in pep_neo_slide:
                long_pep, start, end = extract_sequence(slide_neo, long, n)
                info.append([i[0], i[1], slide_neo, long_pep, str(start), str(end),str(j)])
            
            

    mut_pep_info = indel_path+'/mut_pep_info.txt'
    fw = open(mut_pep_info,'w+')

    for i in info:
        fw.write('\t'.join(i)+'\n')
        fw.flush()
    fw.close()

    gene_tpm = pd.read_csv(rna_path + '/tumor_gene.csv')
    dict_gene_tpm = gene_tpm.groupby('gene_name')['gene_tpm'].mean().to_dict()


    hla = ';'.join(hla_list)


    all_in_one_ls = []
    with open(indel_path + '/mut_pep_info.txt','r') as fo_mut_pep_info:
        mut_pep_info = fo_mut_pep_info.readlines()
    mut_pep_info = [x.strip().split('\t') for x in mut_pep_info]
    for i in mut_pep_info:
        Mut_protein=i[2]
        long_mut_protein=i[3]
        geneid = i[1].split(':')[0]
        start = i[-3]
        end = i[-2]
        #nmid = i[1].split(':')[1]
        #mut_type = i[0]
        length = i[-1]
        if geneid in dict_gene_tpm:
            te = dict_gene_tpm[geneid]
        else:
            te = 0.000
        all_in_one_ls.append([Mut_protein,long_mut_protein,start,end,geneid,str(te),hla,length])

    fw = open(indel_path + '/neo_indel_result.txt','w+')
    title = ['Pep_seq','long_mut_protein','start','end','Gene','Gene_tpm','hla','length']
    fw.write('\t'.join(title)+'\n')
    for i in all_in_one_ls:
        fw.write('\t'.join(str(j) for j in i)+'\n')
        fw.flush()
    fw.close()

def snv_indel_combine(patient_id):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    snv_path = neo_path + '/snv'
    indel_path = neo_path + '/indel'

    snv_file = snv_path + '/neo_snv_result.txt'
    snv_file = pd.read_csv(snv_file, sep='\t')
    snv_file['SNV/INDEL'] = 'SNV'

    indel_file = indel_path + '/neo_indel_result.txt'
    indel_file = pd.read_csv(indel_file, sep='\t')
    indel_file['SNV/INDEL'] = 'INDEL'

    merged_df = snv_file.merge(indel_file, how='outer',on=['Pep_seq','long_mut_protein','start','end','Gene','Gene_tpm','hla','SNV/INDEL','length'])
    merged_df.columns = ['pep','wt_pep','long_pep','start','end','gene','gene_tpm','hla','length','SNV/INDEL']
    merged_df = merged_df[['pep','long_pep','start','end','gene','gene_tpm','length','SNV/INDEL']]
    print(f'neo_result drop before {len(merged_df)}')
    merged_df = merged_df.drop_duplicates()
    print(f'neo_result drop after {len(merged_df)}')
    output = neo_path + '/neo_result.csv'
    merged_df.to_csv(output, index=False)



def execute_command(command,file=None):
    if file:
        with open(file, 'w') as f:
            subprocess.run(command,stdout=f)
            print(f"Executing: {command}")
    else:
        subprocess.run(command)
        print(f"Executing: {command}")

def merge_files(output_dir, merged_file_name):

    merged_file_path = os.path.join(output_dir, merged_file_name)


    with open(merged_file_path, 'w') as merged_file:

        split_files = [f for f in os.listdir(output_dir) if f.endswith('.out')]
        def sort_key(filename):
            return int(''.join(re.findall(r'\d+', filename)))
        sorted_split_files = sorted(split_files, key=sort_key)

        for split_file in sorted_split_files:
            print(split_file)

            split_file_path = os.path.join(output_dir, split_file)

            with open(split_file_path, 'r') as split_file_handle:

                merged_file.write(split_file_handle.read())

    print(f'All files have been merged into {merged_file_path}')

def split_fasta_file(input_file, num_files, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    seq_records = list(SeqIO.parse(input_file, "fasta"))
    

    records_per_file = len(seq_records) // num_files

    remainder = len(seq_records) % num_files
    record_counts = [records_per_file + 1] * remainder + [records_per_file] * (num_files - remainder)
    

    file_index = 0
    

    for i in range(num_files):
        output_file = os.path.join(output_dir, f"split_{i+1}.fsa")
        with open(output_file, "w") as f:

            for j in range(record_counts[i]):
                if file_index < len(seq_records):
                    record = seq_records[file_index]
                    file_index += 1
                    f.write(f">{record.id}\n{record.seq}\n")
    
    print(f"FASTA file has been successfully split into {num_files} files in the '{output_dir}' directory.")

def split_csv_file(input_file, n, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    df = pd.read_csv(input_file)
    

    chunk_size = len(df) // n
    remainder = len(df) % n
    

    for i in range(n):
        start = i * chunk_size
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size
        

        chunk_df = df.iloc[start:end]
        

        output_file = os.path.join(output_dir, f'split_{i+1}.csv')
        chunk_df.to_csv(output_file, index=False)

def merge_csv_file(directory,merged_file_name):

    csv_files = [f for f in os.listdir(directory) if f.endswith('.out.csv')]
    

    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.csv')[0]))
    

    merged_df = pd.DataFrame()
    

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    

    merged_df.to_csv(merged_file_name, index=False)


def execute_batch_commands(commands):

    processes = []
    for command in commands:
        p = multiprocessing.Process(target=execute_command, args=(command,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("Batch commands have been executed.")

def feature_input(pep_length,hla_list,patient_id):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    feature_path = parameter['work_path'] + '/calspace/' + patient_id + '/5_feature'
    create_path(feature_path)
    bigmhc_path = feature_path + '/bigmhc'
    create_path(bigmhc_path)

    netchop_path = feature_path + '/netchop'
    create_path(netchop_path)

    netctlpan_path = feature_path + '/netctlpan'
    create_path(netctlpan_path)

    netmhcpan_path = feature_path + '/netmhcpan'
    create_path(netmhcpan_path)

    prime_path = feature_path + '/prime'
    create_path(prime_path)


    df = pd.read_csv(neo_path + '/neo_result.csv')
    df['hla'] = [hla_list] * len(df)
    # ---------------------------------- netchop --------------------------------- #

    f = open(netchop_path + '/netchop_input.fsa', 'w+')
    for _, row in df.iterrows():
        f.write('>pep'+'\n')
        f.write(row.long_pep + '\n')
    f.close()

    # --------------------------------- netctlpan -------------------------------- #

    def filter_len_n(df, n):
        df_filter = df[df['length'] == n]
        return df_filter
        

    for i in pep_length:
        df_filter = filter_len_n(df,int(i)) 
        f = open(netctlpan_path + '/netctlpan_input_{}.fsa'.format(i), 'w+')
        for _, row in df_filter.iterrows():
            f.writelines('>pep'+'\n')
            f.writelines(row.pep + '\n')
        f.close()
        
    # --------------------------------- netmhcpan -------------------------------- #

    for i in pep_length:
        df_filter = filter_len_n(df,int(i))  
        df_filter = df_filter['pep'] 
        df_filter = df_filter.drop_duplicates()
        df_filter.to_csv(netmhcpan_path + '/netmhcpan_mut_input_{}.txt'.format(i),sep='\t',index=False,header=False)

    # ---------------------------------- bigmhc ---------------------------------- #

    bigmhc = df[['hla','pep']]
    bigmhc = bigmhc.explode('hla')
    bigmhc.columns = ['mhc', 'pep']
    bigmhc = bigmhc.drop_duplicates()
    bigmhc.to_csv(bigmhc_path + '/bigmhc_mut_input.csv', index=False)

    # ----------------------------------- prime ---------------------------------- #

    prime = df['pep']
    prime = prime.drop_duplicates()
    prime.to_csv(prime_path + '/prime_input.txt', index=False, sep = '\t', header = False)

def feature_cal(pep_length,hla_list,patient_id):
    feature_path = parameter['work_path'] + '/calspace/' + patient_id +  '/5_feature'
    bigmhc_path = feature_path + '/bigmhc'
    netchop_path = feature_path + '/netchop'
    netctlpan_path = feature_path + '/netctlpan'
    netmhcpan_path = feature_path + '/netmhcpan'
    prime_path = feature_path + '/prime'

    # ---------------------------------- log --------------------------------- #
    f_time = open(feature_path + '/feature_time.log','w+')

    f_time.write('***** feature cal start!*****\n')
    f_time.flush()
    A = time.time()


    # --------------------------------- netctlpan -------------------------------- #

    tpm_path = netctlpan_path + '/tpm'
    create_path(tpm_path)
    net_hla = [item.replace('*', '') for item in hla_list] 
    net_hla = ','.join(net_hla)
    #HLA-A11:01,HLA-B37:01,HLA-C06:02

    for j in pep_length :
        commands=[]
        o_list =[]
        i = netctlpan_path+'/netctlpan_input_{}.fsa'.format(j)
        split_fasta_file(i,10,tpm_path)
        for split_file in os.listdir(tpm_path):
            if split_file.endswith('.fsa'):
                fasta_path = os.path.join(tpm_path, split_file)
                o = os.path.join(tpm_path, split_file+'.out')
                fasta_path = os.path.normpath(fasta_path)
                o = os.path.normpath(o)
                command = ['netCTLpan','-l',str(j),'-a',net_hla,fasta_path]
                commands.append(command)
                o_list.append(o)
        processes = []
        for command ,o in zip(commands,o_list):
            p = multiprocessing.Process(target=execute_command, args=(command,o,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        o = netctlpan_path+'/netctlpan_out_{}.txt'.format(j)
        merge_files(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("netctlpan commands have been executed.")
    write_time_log(f_time, A, 'netctlpan Finish!')

    # ---------------------------------- netchop --------------------------------- #

    tpm_path = netchop_path + '/tpm'
    create_path(tpm_path)
    commands=[]
    o_list =[]
    i = netchop_path+'/netchop_input.fsa'
    split_fasta_file(i,10,tpm_path)
    for split_file in os.listdir(tpm_path):
        if split_file.endswith('.fsa'):
            fasta_path = os.path.join(tpm_path, split_file)
            o = os.path.join(tpm_path, split_file+'.out')
            fasta_path = os.path.normpath(fasta_path)
            o = os.path.normpath(o)
            command = ['netchop',fasta_path]
            commands.append(command)
            o_list.append(o)
    processes = []
    for command ,o in zip(commands,o_list):
        p = multiprocessing.Process(target=execute_command, args=(command,o,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    o = netchop_path+'/netchop_out.txt'
    merge_files(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("netchop commands have been executed.")
    write_time_log(f_time, A, 'netchop Finish!')


    # --------------------------------- netmhcpan -------------------------------- #
    net_hla = [item.replace('*', '') for item in hla_list] 
    net_hla = ','.join(net_hla)
    #HLA-A11:01,HLA-B37:01,HLA-C06:02

    for j in pep_length:
        i = netmhcpan_path+'/netmhcpan_mut_input_{}.txt'.format(j)
        o = netmhcpan_path+'/netmhcpan_mut_out_{}.txt'.format(j)
        os.system('netMHCpan -l ' + str(j) + ' -a ' + net_hla + ' -p ' + i + ' > ' + o)

    write_time_log(f_time, A, 'netmhcpan Finish!')

    # ---------------------------------- bigmhc ---------------------------------- #

    tpm_path = bigmhc_path + '/tpm'
    create_path(tpm_path)
    commands=[]
    i = bigmhc_path+'/bigmhc_mut_input.csv'
    split_csv_file(i,4,tpm_path)
    for split_file in os.listdir(tpm_path):
        if split_file.endswith('.csv'):
            fasta_path = os.path.join(tpm_path, split_file)
            fasta_path = os.path.normpath(fasta_path)
            o = os.path.join(tpm_path, split_file+'.out.csv')
            o = os.path.normpath(o)
            command = ['python',parameter['bigmhc_path']+'/predict.py','-i='+fasta_path,'-o='+o,'-m=im','-b=32','-d=cpu']
            commands.append(command)
    

    batch_size = 2

    while commands:
        batch_commands = commands[:batch_size]
        execute_batch_commands(batch_commands)
        commands = commands[batch_size:]
    
    o = bigmhc_path+'/bigmhc_mut_im.csv'
    merge_csv_file(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("bigmhc commands have been executed.")
    write_time_log(f_time, A, 'bigmhc Finish!')


    # ----------------------------------- prime ---------------------------------- #

    #PRIME -i test/test.txt -o test/out.txt -a A0101,A2501,B0801,B1801

    i = prime_path+'/prime_input.txt'
    o = prime_path+'/prime_output.txt'

    pri_hla = [item.replace('*', '') for item in hla_list] 
    pri_hla = [item.replace(':', '') for item in pri_hla] 
    print(pri_hla)
    pri_hla = [item.split('-')[1] for item in pri_hla] 
    pri_hla = ','.join(pri_hla)
    #A1101,B3701,C0602

    os.system('PRIME ' + '-i '+ i + ' -o ' + o + ' -a ' + pri_hla )

    write_time_log(f_time, A, 'prime Finish!')

    f_time.write('***** feature cal Finish!*****\n')
    f_time.flush()
    f_time.close()

def feature_merge(pep_length,hla_list,patient_id):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    feature_path = parameter['work_path'] + '/calspace/' + patient_id + '/5_feature'
    bigmhc_path = feature_path + '/bigmhc'
    netchop_path = feature_path + '/netchop'
    netctlpan_path = feature_path + '/netctlpan'
    netmhcpan_path = feature_path + '/netmhcpan'
    prime_path = feature_path + '/prime'

    df = pd.read_csv(neo_path + '/neo_result.csv')
    print('The num of SNV and INDEL {}'.format(len(df)))

    df['hla'] = [hla_list] * len(df)
    df = df.explode('hla')
    # ------------------------------ netchop result ------------------------------ #

    o = netchop_path+'/netchop_out.txt'
    info = dict()
    pep_num = 0

    with open(o, 'r') as file:
        for line in file:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            if line:
                if line.startswith('#'):
                    continue
                if line.startswith('----'):
                    continue
                if line.startswith('NetChop'):
                    continue
                if line.startswith('Number'):
                    continue
                if 'pos' in line:
                    pep_num += 1
                    continue
                pos = int(line.split(' ')[0])
                score = float(line.split(' ')[3])
                key = ','.join([str(pep_num), str(pos)])
                info[key] = score
    
    for index, row in df.iterrows():
        key = ','.join([str(index+1), str(row.start)])
        df.loc[index,'netchop_score'] = info.get(key)

    # ----------------------------- netctlpan result ----------------------------- #

    info = []
    for j in pep_length:
        o = netctlpan_path+'/netctlpan_out_{}.txt'.format(j)
        with open(o, 'r') as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)
                if line:
                    if line.startswith('#'):
                        continue
                    if line.startswith('----'):
                        continue
                    if line.startswith('Number'):
                        continue
                    if 'HLA' in line:
                        hla = line.split(' ')[2]
                        pep = line.split(' ')[3]
                        score = float(line.split(' ')[5])
                        info.append([hla, pep, score])

    info = pd.DataFrame(info)
    info.columns = ['hla', 'pep', 'tap_score']

    df = df.merge(info,how='left',on=['pep','hla']) 

    # --------------------------------- netmhcpan -------------------------------- #

    info = []
    for j in pep_length:
        o = netmhcpan_path+'/netmhcpan_mut_out_{}.txt'.format(j)
        with open(o, 'r') as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)
                if line:
                    if line.startswith('#'):
                        continue
                    if line.startswith('----'):
                        continue
                    if 'Distance to training data' in line:
                        continue
                    if 'Pos' in line:
                        continue
                    if line.startswith('Protein PEPLIST'):
                        continue
                    if 'HLA' in line:
                        hla = line.split(' ')[1]
                        pep = line.split(' ')[2]
                        rank = float(line.split(' ')[12])
                        info.append([hla, pep, rank])
    info = pd.DataFrame(info)
    info.columns = ['hla', 'pep', 'netmhcpan_mut_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 

    # ---------------------------------- bigmhc ---------------------------------- #

    o = bigmhc_path+'/bigmhc_mut_im.csv'
    info = pd.read_csv(o)
    info = info[['mhc', 'pep', 'BigMHC_IM']]
    info.columns = ['hla', 'pep', 'bigmhc_mut_im']
    df = df.merge(info,how='left',on=['pep','hla']) 

    # ----------------------------------- prime ---------------------------------- #

    o = prime_path+'/prime_output.txt'
    info = pd.read_csv(o, sep='\t', comment='#' )

    pattern = re.compile(r'%Rank_[ABC]\d+')
    matching_columns = [col for col in info.columns if pattern.match(col)]

    info = info[['Peptide'] + matching_columns]

    info = pd.melt(info, id_vars=['Peptide'], var_name='hla', value_name='prime_rank')

    f1 = lambda x: x.split('_')[-1]
    f2 = lambda x: 'HLA-'+list(x)[0]+'*'+str(''.join(list(x)[1:3]))+':'+str(''.join(list(x)[3:]))
    info['hla'] = info['hla'].apply(f1)
    info['hla'] = info['hla'].apply(f2)

    info.columns = ['pep', 'hla', 'prime_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 


    o = prime_path+'/prime_output.txt'
    info = pd.read_csv(o, sep='\t', comment='#' )
    pattern = re.compile(r'%RankBinding_[ABC]\d+')
    matching_columns = [col for col in info.columns if pattern.match(col)]

    info = info[['Peptide'] + matching_columns]

    info = pd.melt(info, id_vars=['Peptide'], var_name='hla', value_name='mixmhcpred_rank')

    f1 = lambda x: x.split('_')[-1]
    f2 = lambda x: 'HLA-'+list(x)[0]+'*'+str(''.join(list(x)[1:3]))+':'+str(''.join(list(x)[3:]))
    info['hla'] = info['hla'].apply(f1)
    info['hla'] = info['hla'].apply(f2)

    info.columns = ['pep', 'hla', 'mixmhcpred_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 
    df.to_csv(parameter['work_path'] + '/calspace/'+ patient_id +'/df.csv',index=False)
    print('The num of SNV and INDEL after featured {}'.format(len(df)))





def snv_n(protein, s, neo_aa, n:int):

    if s >= n + 1 and s <= len(protein) - n:
        pep_space_raw = protein[s - n - 1:s + n]
        pep_space_neo = protein[s - n - 1:s - 1] + neo_aa + protein[s:s + n]

    elif s < n + 1 and s <= len(protein) - n:
        pep_space_raw = protein[:s + n]
        pep_space_neo = protein[:s - 1] + neo_aa + protein[s:s + n]

    elif s >= n + 1 and s > len(protein) - n:
        pep_space_raw = protein[s - n - 1:]
        pep_space_neo = protein[s - n - 1:s - 1] + neo_aa + protein[s:]

    else:
        pep_space_raw = protein[:]
        pep_space_neo = protein[:s - 1] + neo_aa + protein[s:]
    return pep_space_raw, pep_space_neo

def slide_window(seq, step):
    return [seq[i:i + step] for i in range(0, len(seq) - step + 1)]

def extract_sequence(sub_seq, long_seq, n:int):
    start = long_seq.index(sub_seq) - n
    if start < 0:
        start = 0
    end = start + len(sub_seq) + 2 * n
    if end > len(long_seq):
        end = len(long_seq)
    extracted_seq = long_seq[start:end]
    return extracted_seq, extracted_seq.index(sub_seq)+1, extracted_seq.index(sub_seq)+len(sub_seq)


codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def translate_cDNA_to_protein(cDNA_sequence):
    protein_sequence = ''
    start_codon = 'ATG'
    # Ensure the sequence starts with the start codon 'ATG'
    if cDNA_sequence.startswith(start_codon):
        # Iterate through the sequence in steps of 3 (codon size)
        for i in range(0, len(cDNA_sequence), 3):
            codon = cDNA_sequence[i:i+3]
            # Check if the codon exists in the codon table
            if codon in codon_table:
                amino_acid = codon_table[codon]
                # Stop translation if a stop codon is encountered
                if amino_acid == '*':
                    break
                protein_sequence += amino_acid
    return protein_sequence




####RNN PREDIECT

def makedata(path, cols, batch_size, shuffle_):
    dat = pd.read_csv(path)
    dat = dat[cols]
    dim = dat.shape[1]-2
    data = dat.iloc[:,2:].to_numpy().astype(np.float32).reshape((dat.shape[0],dim,1))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle = shuffle_)
    return dat,data_loader

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def predict_epoch(model, data_loader, device):
    model.eval() 
    y_prob_val_list = []
    with torch.no_grad():
        for _, (data) in enumerate(data_loader):
            data = data.to(device)
            y_prob_val = model(data)
            y_prob_val_list.extend(y_prob_val.cpu().detach().numpy())
    y_prob_val_list = [float(x) for sublist in y_prob_val_list for x in sublist]
    return y_prob_val_list

def get_rank(pep, y_prob):
    re={ 'pep':pep, 'y_prob':y_prob}
    df = pd.DataFrame(re)
    df = df.groupby(['pep'])['y_prob'].max().reset_index()
    df['rank'] = df['y_prob'].rank(ascending=False, method='min')
    return df

def bind_all_model(data_path,model_path, cols_list, input_size, hidden_size, output_size, batch_size, device):
    df = pd.DataFrame()
    for _, cols in enumerate(cols_list):
        for file in os.listdir(model_path):
            if file.endswith('{}_{}.pkl'.format(cols[-2], cols[-1])):
                print('****'+file)
                model = RNN(input_size, hidden_size, output_size).to(device)
                model.load_state_dict(torch.load(model_path+'/'+file,map_location=device))
                test_data,test_loader = makedata(path = data_path, cols = cols,  batch_size = batch_size, shuffle_ = False)
                y_prob_val_list = predict_epoch(model, test_loader, device)
                result = get_rank(test_data['pep'],y_prob_val_list)
                result.rename(columns={'rank': '{}_{}_rank'.format(cols[-2], cols[-1]),
                                      'y_prob': '{}_{}_prob'.format(cols[-2], cols[-1])},inplace=True)
                result.sort_values(by='{}_{}_rank'.format(cols[-2], cols[-1]) , inplace=True, ascending=True) 
                if df.empty:
                    df = pd.concat([df,result],axis=1)
                    print(len(df))
                else:
                    df = df.merge(result,how='left',on=['pep'])
                    print(len(df))
    if len(df)>2000:
        print('There are too many PEPs. Filter out PEPs that rank lower!')
        print(f'Number of pep before filtering {len(df)}')
        threshold=500
        df = df[(df['mixmhcpred_rank_prime_rank_rank'] <= threshold) | (df['netmhcpan_mut_rank_prime_rank_rank'] <= threshold) | 
                    (df['mixmhcpred_rank_bigmhc_mut_im_rank'] <= threshold) | (df['netmhcpan_mut_rank_bigmhc_mut_im_rank'] <= threshold)]
        print(f'Number of PEPs after filtering {len(df)}')
    print(f'Number of PEPs {len(df)}')
    return df

def sum_rank(df):
    selected_columns = [col for col in df.columns if col.endswith('prob')]
    print(selected_columns)
    df['sum'] = df[selected_columns].apply( lambda x: (x - x.mean()) / x.std(), axis=0).sum(axis=1)
    df['sum_rank'] = df['sum'].rank(method='min',ascending=False)
    return df

def predict_model(patient_id,pep=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    input_size, hidden_size, output_size = 1, 16, 1
    batch_size = 1024

    cols_list = [['pep','hla','gene_tpm','netchop_score','tap_score','mixmhcpred_rank','prime_rank'],
                ['pep','hla','gene_tpm','netchop_score','tap_score','netmhcpan_mut_rank','prime_rank'],
                ['pep','hla','gene_tpm','netchop_score','tap_score','mixmhcpred_rank','bigmhc_mut_im'],
                ['pep','hla','gene_tpm','netchop_score','tap_score','netmhcpan_mut_rank','bigmhc_mut_im']]

    model_path = './model'
    data_path = './calspace/'+ patient_id + '/df.csv'

    muti_RNN = bind_all_model(data_path, model_path, cols_list, input_size, hidden_size, output_size, batch_size, device)
    print(len(muti_RNN))
    print('bind done')

    re = sum_rank(muti_RNN)
    print(len(re))
    print('sum done')

    if not pep:
        df = pd.read_csv('./calspace/'+ patient_id + '/4_neo/neo_result.csv')
        df = df[['pep','gene','SNV/INDEL']]
        df = df.drop_duplicates()
        re = df.merge(re,how='inner',on=['pep'])
        print(len(re))
        print('merge done')
        re = re[['pep','gene','SNV/INDEL','mixmhcpred_rank_prime_rank_prob','mixmhcpred_rank_prime_rank_rank','netmhcpan_mut_rank_prime_rank_prob','netmhcpan_mut_rank_prime_rank_rank','mixmhcpred_rank_bigmhc_mut_im_prob','mixmhcpred_rank_bigmhc_mut_im_rank','netmhcpan_mut_rank_bigmhc_mut_im_prob','netmhcpan_mut_rank_bigmhc_mut_im_rank','sum','sum_rank']]
        re.columns = ['pep','gene','SNV/INDEL','RNN_MP_prob','RNN_MP_rank','RNN_NP_prob','RNN_NP_rank','RNN_MB_prob','RNN_MB_rank','RNN_NB_prob','RNN_NB_rank','RNN_voting','RNN_voting_rank']
        re.sort_values(by="RNN_voting" , inplace=True, ascending=False) 
    else:
        df = pd.read_csv(data_path)
        df = df[['pep','gene']]
        df = df.drop_duplicates()
        re = df.merge(re,how='inner',on=['pep'])
        print(len(re))
        print('merge done')
        re = re[['pep','gene','mixmhcpred_rank_prime_rank_prob','mixmhcpred_rank_prime_rank_rank','netmhcpan_mut_rank_prime_rank_prob','netmhcpan_mut_rank_prime_rank_rank','mixmhcpred_rank_bigmhc_mut_im_prob','mixmhcpred_rank_bigmhc_mut_im_rank','netmhcpan_mut_rank_bigmhc_mut_im_prob','netmhcpan_mut_rank_bigmhc_mut_im_rank','sum','sum_rank']]
        re.columns = ['pep','gene','RNN_MP_prob','RNN_MP_rank','RNN_NP_prob','RNN_NP_rank','RNN_MB_prob','RNN_MB_rank','RNN_NB_prob','RNN_NB_rank','RNN_voting','RNN_voting_rank']
        re.sort_values(by="RNN_voting" , inplace=True, ascending=False) 

    print(re.head(5))

    re.to_csv('./calspace/' + patient_id + '/'+patient_id+'_rank_RNN.csv', index=False)


def feature_input2(pep_path,hla_list,patient_id):
    feature_path = parameter['work_path'] + '/calspace/' + patient_id + '/5_feature'
    create_path(feature_path)
    bigmhc_path = feature_path + '/bigmhc'
    create_path(bigmhc_path)

    netchop_path = feature_path + '/netchop'
    create_path(netchop_path)

    netctlpan_path = feature_path + '/netctlpan'
    create_path(netctlpan_path)

    netmhcpan_path = feature_path + '/netmhcpan'
    create_path(netmhcpan_path)

    prime_path = feature_path + '/prime'
    create_path(prime_path)


    df = pd.read_csv(pep_path)
    df['length'] = df['pep'].apply(len)
    
    df['hla'] = [hla_list] * len(df)

    pep_length = list(set(df['length']))
    print(f'pep length: {pep_length}')
    print(df.head(5))

    # ---------------------------------- netchop --------------------------------- #

    f = open(netchop_path + '/netchop_input.fsa', 'w+')
    for _, row in df.iterrows():
        f.write('>pep'+'\n')
        f.write(row.pep + '\n')
    f.close()

    # --------------------------------- netctlpan -------------------------------- #

    def filter_len_n(df, n):
        df_filter = df[df['length'] == n]
        return df_filter
        

    for i in pep_length:
        df_filter = filter_len_n(df,int(i))   
        f = open(netctlpan_path + '/netctlpan_input_{}.fsa'.format(i), 'w+')
        for _, row in df_filter.iterrows():
            f.writelines('>pep'+'\n')
            f.writelines(row.pep + '\n')
        f.close()
        
    # --------------------------------- netmhcpan -------------------------------- #

    for i in pep_length:
        df_filter = filter_len_n(df,int(i))  
        df_filter = df_filter['pep'] 
        df_filter = df_filter.drop_duplicates()
        df_filter.to_csv(netmhcpan_path + '/netmhcpan_mut_input_{}.txt'.format(i),sep='\t',index=False,header=False)

    # ---------------------------------- bigmhc ---------------------------------- #

    bigmhc = df[['hla','pep']]
    bigmhc = bigmhc.explode('hla')
    bigmhc.columns = ['mhc', 'pep']
    bigmhc = bigmhc.drop_duplicates()
    bigmhc.to_csv(bigmhc_path + '/bigmhc_mut_input.csv', index=False)

    # ----------------------------------- prime ---------------------------------- #

    prime = df['pep']
    prime = prime.drop_duplicates()
    prime.to_csv(prime_path + '/prime_input.txt', index=False, sep = '\t', header = False)

def feature_cal2(pep_path,hla_list,patient_id):
    feature_path = parameter['work_path'] + '/calspace/' + patient_id +  '/5_feature'
    bigmhc_path = feature_path + '/bigmhc'
    netchop_path = feature_path + '/netchop'
    netctlpan_path = feature_path + '/netctlpan'
    netmhcpan_path = feature_path + '/netmhcpan'
    prime_path = feature_path + '/prime'

    df = pd.read_csv(pep_path)
    df['length'] = df['pep'].apply(len)
    pep_length = list(set(df['length']))

    # ---------------------------------- log--------------------------------- #
    f_time = open(feature_path + '/feature_time.log','w+')

    f_time.write('***** feature cal start!*****\n')
    f_time.flush()
    A = time.time()


    # --------------------------------- netctlpan -------------------------------- #

    tpm_path = netctlpan_path + '/tpm'
    create_path(tpm_path)
    net_hla = [item.replace('*', '') for item in hla_list] 
    net_hla = ','.join(net_hla)
    #HLA-A11:01,HLA-B37:01,HLA-C06:02

    for j in pep_length :
        commands=[]
        o_list =[]
        i = netctlpan_path+'/netctlpan_input_{}.fsa'.format(j)
        split_fasta_file(i,10,tpm_path)
        for split_file in os.listdir(tpm_path):
            if split_file.endswith('.fsa'):
                fasta_path = os.path.join(tpm_path, split_file)
                o = os.path.join(tpm_path, split_file+'.out')
                fasta_path = os.path.normpath(fasta_path)
                o = os.path.normpath(o)
                command = ['netCTLpan','-l',str(j),'-a',net_hla,fasta_path]
                commands.append(command)
                o_list.append(o)
        processes = []
        for command ,o in zip(commands,o_list):
            p = multiprocessing.Process(target=execute_command, args=(command,o,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        o = netctlpan_path+'/netctlpan_out_{}.txt'.format(j)
        merge_files(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("netctlpan commands have been executed.")
    write_time_log(f_time, A, 'netctlpan Finish!')

    # ---------------------------------- netchop --------------------------------- #

    tpm_path = netchop_path + '/tpm'
    create_path(tpm_path)
    commands=[]
    o_list =[]
    i = netchop_path+'/netchop_input.fsa'
    split_fasta_file(i,10,tpm_path)
    for split_file in os.listdir(tpm_path):
        if split_file.endswith('.fsa'):
            fasta_path = os.path.join(tpm_path, split_file)
            o = os.path.join(tpm_path, split_file+'.out')
            fasta_path = os.path.normpath(fasta_path)
            o = os.path.normpath(o)
            command = ['netchop',fasta_path]
            commands.append(command)
            o_list.append(o)
    processes = []
    for command ,o in zip(commands,o_list):
        p = multiprocessing.Process(target=execute_command, args=(command,o,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    o = netchop_path+'/netchop_out.txt'
    merge_files(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("netchop commands have been executed.")
    write_time_log(f_time, A, 'netchop Finish!')


    # --------------------------------- netmhcpan -------------------------------- #
    net_hla = [item.replace('*', '') for item in hla_list] 
    net_hla = ','.join(net_hla)
    #HLA-A11:01,HLA-B37:01,HLA-C06:02

    for j in pep_length:
        i = netmhcpan_path+'/netmhcpan_mut_input_{}.txt'.format(j)
        o = netmhcpan_path+'/netmhcpan_mut_out_{}.txt'.format(j)
        os.system('netMHCpan -l ' + str(j) + ' -a ' + net_hla + ' -p ' + i + ' > ' + o)

    write_time_log(f_time, A, 'netmhcpan Finish!')

    # ---------------------------------- bigmhc ---------------------------------- #

    tpm_path = bigmhc_path + '/tpm'
    create_path(tpm_path)
    commands=[]
    i = bigmhc_path+'/bigmhc_mut_input.csv'
    split_csv_file(i,4,tpm_path)
    for split_file in os.listdir(tpm_path):
        if split_file.endswith('.csv'):
            fasta_path = os.path.join(tpm_path, split_file)
            fasta_path = os.path.normpath(fasta_path)
            o = os.path.join(tpm_path, split_file+'.out.csv')
            o = os.path.normpath(o)
            command = ['python',parameter['bigmhc_path']+'/predict.py','-i='+fasta_path,'-o='+o,'-m=im','-b=32','-d=cpu']
            commands.append(command)
    

    batch_size = 2

    while commands:
        batch_commands = commands[:batch_size]
        execute_batch_commands(batch_commands)
        commands = commands[batch_size:]
    
    o = bigmhc_path+'/bigmhc_mut_im.csv'
    merge_csv_file(tpm_path,o)
    
    dir1 = tpm_path
    os.system('rm -rf '+dir1)
        
    print("bigmhc commands have been executed.")
    write_time_log(f_time, A, 'bigmhc Finish!')


    # ----------------------------------- prime ---------------------------------- #

    #PRIME -i test/test.txt -o test/out.txt -a A0101,A2501,B0801,B1801

    i = prime_path+'/prime_input.txt'
    o = prime_path+'/prime_output.txt'

    pri_hla = [item.replace('*', '') for item in hla_list] 
    pri_hla = [item.replace(':', '') for item in pri_hla] 
    pri_hla = [item.split('-')[1] for item in pri_hla] 
    pri_hla = ','.join(pri_hla)
    #A1101,B3701,C0602

    os.system('PRIME ' + '-i '+ i + ' -o ' + o + ' -a ' + pri_hla )

    write_time_log(f_time, A, 'prime Finish!')

    f_time.write('***** feature cal Finish!*****\n')
    f_time.flush()
    f_time.close()

def feature_merge2(pep_path,hla_list,patient_id):
    feature_path = parameter['work_path'] + '/calspace/' + patient_id + '/5_feature'
    bigmhc_path = feature_path + '/bigmhc'
    netchop_path = feature_path + '/netchop'
    netctlpan_path = feature_path + '/netctlpan'
    netmhcpan_path = feature_path + '/netmhcpan'
    prime_path = feature_path + '/prime'

    df = pd.read_csv(pep_path)
    df['hla'] = [hla_list] * len(df)
    df['length'] = df['pep'].apply(len)
    pep_length = list(set(df['length']))

    print('The num of SNV and INDEL {}'.format(len(df)))
    df = df.explode('hla')
    # ------------------------------ netchop result ------------------------------ #

    o = netchop_path+'/netchop_out.txt'
    info = dict()
    pep_num = 0

    with open(o, 'r') as file:
        for line in file:
            line = line.strip()
            line = re.sub(r"\s+", " ", line)
            if line:
                if line.startswith('#'):
                    continue
                if line.startswith('----'):
                    continue
                if line.startswith('NetChop'):
                    continue
                if line.startswith('Number'):
                    continue
                if 'pos' in line:
                    pep_num += 1
                    continue
                pos = int(line.split(' ')[0])
                score = float(line.split(' ')[3])
                key = ','.join([str(pep_num), str(pos)])
                info[key] = score
    
    for index, _ in df.iterrows():
        key = ','.join([str(index+1), str(1)])
        df.loc[index,'netchop_score'] = info.get(key)

    # ----------------------------- netctlpan result ----------------------------- #

    info = []
    for j in pep_length:
        o = netctlpan_path+'/netctlpan_out_{}.txt'.format(j)
        with open(o, 'r') as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)
                if line:
                    if line.startswith('#'):
                        continue
                    if line.startswith('----'):
                        continue
                    if line.startswith('Number'):
                        continue
                    if 'HLA' in line:
                        hla = line.split(' ')[2]
                        pep = line.split(' ')[3]
                        score = float(line.split(' ')[5])
                        info.append([hla, pep, score])

    info = pd.DataFrame(info)
    info.columns = ['hla', 'pep', 'tap_score']

    df = df.merge(info,how='left',on=['pep','hla']) 

    # --------------------------------- netmhcpan -------------------------------- #

    info = []
    for j in pep_length:
        o = netmhcpan_path+'/netmhcpan_mut_out_{}.txt'.format(j)
        with open(o, 'r') as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)
                if line:
                    if line.startswith('#'):
                        continue
                    if line.startswith('----'):
                        continue
                    if 'Distance to training data' in line:
                        continue
                    if 'Pos' in line:
                        continue
                    if line.startswith('Protein PEPLIST'):
                        continue
                    if 'HLA' in line:
                        hla = line.split(' ')[1]
                        pep = line.split(' ')[2]
                        rank = float(line.split(' ')[12])
                        info.append([hla, pep, rank])
    info = pd.DataFrame(info)
    info.columns = ['hla', 'pep', 'netmhcpan_mut_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 

    # ---------------------------------- bigmhc ---------------------------------- #

    o = bigmhc_path+'/bigmhc_mut_im.csv'
    info = pd.read_csv(o)
    info = info[['mhc', 'pep', 'BigMHC_IM']]
    info.columns = ['hla', 'pep', 'bigmhc_mut_im']
    df = df.merge(info,how='left',on=['pep','hla']) 

    # ----------------------------------- prime ---------------------------------- #

    o = prime_path+'/prime_output.txt'
    info = pd.read_csv(o, sep='\t', comment='#' )

    pattern = re.compile(r'%Rank_[ABC]\d+')
    matching_columns = [col for col in info.columns if pattern.match(col)]

    info = info[['Peptide'] + matching_columns]

    info = pd.melt(info, id_vars=['Peptide'], var_name='hla', value_name='prime_rank')

    f1 = lambda x: x.split('_')[-1]
    f2 = lambda x: 'HLA-'+list(x)[0]+'*'+str(''.join(list(x)[1:3]))+':'+str(''.join(list(x)[3:]))
    info['hla'] = info['hla'].apply(f1)
    info['hla'] = info['hla'].apply(f2)

    info.columns = ['pep', 'hla', 'prime_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 


    o = prime_path+'/prime_output.txt'
    info = pd.read_csv(o, sep='\t', comment='#' )
    pattern = re.compile(r'%RankBinding_[ABC]\d+')
    matching_columns = [col for col in info.columns if pattern.match(col)]

    info = info[['Peptide'] + matching_columns]

    info = pd.melt(info, id_vars=['Peptide'], var_name='hla', value_name='mixmhcpred_rank')

    f1 = lambda x: x.split('_')[-1]
    f2 = lambda x: 'HLA-'+list(x)[0]+'*'+str(''.join(list(x)[1:3]))+':'+str(''.join(list(x)[3:]))
    info['hla'] = info['hla'].apply(f1)
    info['hla'] = info['hla'].apply(f2)

    info.columns = ['pep', 'hla', 'mixmhcpred_rank']
    df = df.merge(info,how='left',on=['pep','hla']) 
    df.to_csv(parameter['work_path'] + '/calspace/'+ patient_id +'/df.csv',index=False)
    print('The num of SNV and INDEL after featured {}'.format(len(df)))


def mv_file(snv,indel,patient_id):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    create_path(neo_path)
    
    if snv:
        snv_path = neo_path + '/snv'
        create_path(snv_path)
        snv_new = snv_path + '/snv_screened_muts.txt'
        df = pd.read_csv(snv,header=None)
        df.to_csv(snv_new,header=None,index=False,sep='\t')
    if indel:
        indel_path = neo_path + '/indel'
        create_path(indel_path)
        indel_new = indel_path + '/indel_screened_muts.txt'
        df = pd.read_csv(indel,header=None)
        df.to_csv(indel_new,header=None,index=False,sep='\t')
        
def snv_indel_combine2(snv,indel,patient_id):
    neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
    snv_path = neo_path + '/snv'
    indel_path = neo_path + '/indel'
    
    if snv is not None and indel is not None:

        snv_file = snv_path + '/neo_snv_result.txt'
        snv_file = pd.read_csv(snv_file, sep='\t')
        snv_file['SNV/INDEL'] = 'SNV'

        indel_file = indel_path + '/neo_indel_result.txt'
        indel_file = pd.read_csv(indel_file, sep='\t')
        indel_file['SNV/INDEL'] = 'INDEL'

        merged_df = snv_file.merge(indel_file, how='outer',on=['Pep_seq','long_mut_protein','start','end','Gene','Gene_tpm','hla','SNV/INDEL','length'])
        merged_df.columns = ['pep','wt_pep','long_pep','start','end','gene','gene_tpm','hla','length','SNV/INDEL']
        merged_df = merged_df[['pep','long_pep','start','end','gene','gene_tpm','length','SNV/INDEL']]
        print(f'neo_result drop before {len(merged_df)}')
        merged_df = merged_df.drop_duplicates()
        print(f'neo_result drop after {len(merged_df)}')
        output = neo_path + '/neo_result.csv'
        merged_df.to_csv(output, index=False)
        print('snv and indel pep done')
    elif snv:
        snv_file = snv_path + '/neo_snv_result.txt'
        snv_file = pd.read_csv(snv_file, sep='\t')
        snv_file['SNV/INDEL'] = 'SNV'

        snv_file.columns = ['pep','wt_pep','long_pep','start','end','gene','gene_tpm','hla','length','SNV/INDEL']
        print(f'neo_result drop before {len(snv_file)}')
        snv_file = snv_file.drop_duplicates()
        print(f'neo_result drop after {len(snv_file)}')
        snv_file = snv_file[['pep','long_pep','start','end','gene','gene_tpm','length','SNV/INDEL']]
        output = neo_path + '/neo_result.csv'
        snv_file.to_csv(output, index=False)
        print('snv pep done')
    else:

        indel_file = indel_path + '/neo_indel_result.txt'
        indel_file = pd.read_csv(indel_file, sep='\t')
        indel_file['SNV/INDEL'] = 'INDEL'
        indel_file.columns = ['pep','long_pep','start','end','gene','gene_tpm','hla','length','SNV/INDEL']
        print(f'neo_result drop before {len(indel_file)}')
        indel_file = indel_file.drop_duplicates()
        print(f'neo_result drop after {len(indel_file)}')
        indel_file = indel_file[['pep','long_pep','start','end','gene','gene_tpm','length','SNV/INDEL']]
        output = neo_path + '/neo_result.csv'
        indel_file.to_csv(output, index=False)
        print('indel pep done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--length",help="pep length:8,9,10,11",metavar='',default="8,9,10,11",type=str)
    parser.add_argument("-t","--thread",help="thread:64",default=64,metavar='',type=int)
    parser.add_argument("-p","--pep_csv",help="pep_csv path",metavar='')
    parser.add_argument("--snv",help="mut_vcf path",metavar='')
    parser.add_argument("--indel",help="mut_vcf path",metavar='')
    parser.add_argument("-g","--gene_tpm",metavar='',help="gene_tpm path")
    parser.add_argument("--id",help="patient id",required=True)
    parser.add_argument("--tumor_dna_1",help="path of tumor_dna_1",metavar='')
    parser.add_argument("--tumor_dna_2",help="path of tumor_dna_2",metavar='')
    parser.add_argument("--normal_dna_1",help="path of normal_dna_1",metavar='')
    parser.add_argument("--normal_dna_2",help="path of normal_dna_2",metavar='')
    parser.add_argument("--tumor_rna_1",help="path of tumor_rna_1",metavar='')
    parser.add_argument("--tumor_rna_2",help="path of tumor_rna_2",metavar='')
    parser.add_argument("--hla",help="hla: HLA-A*11:01,HLA-B*46:01",required=True)
    args = parser.parse_args()
    # hla_list = ['HLA-A*11:01','HLA-B*46:01','HLA-C*01:02']
    patient_id = args.id
    thread=args.thread
    get_parameter()
    calspace_path = parameter['work_path'] + '/calspace/'+patient_id
    create_path(calspace_path)
    print('work path: {}'.format(parameter['work_path']))
    print('patient id {}'.format(patient_id))
    print('thread:{}'.format(str(thread)))

    tumor_dna_1 = args.tumor_dna_1
    tumor_dna_2 = args.tumor_dna_2
    normal_dna_1 = args.normal_dna_1
    normal_dna_2 = args.normal_dna_2
    tumor_rna_1 = args.tumor_rna_1
    tumor_rna_2 = args.tumor_rna_1


    if args.pep_csv is None:
        if args.gene_tpm:
            tpm_path = args.gene_tpm
            df = pd.read_csv(tpm_path)
            rna_path = parameter['work_path'] + '/calspace/' + patient_id + '/2_rna'
            create_path(rna_path)
            df.to_csv(rna_path+'/tumor_gene.csv',index=False)
            print('**********tpm read**********')
        else: 
            print('**********rna2tpm start**********')
            rna2tpm(thread,patient_id,tumor_rna_1,tumor_rna_2)
    else:
        pass


    if args.snv is None and args.indel is None and args.pep_csv is None:
        print('**********wes2mut start**********')
        wes2mut(thread,patient_id,tumor_dna_1,tumor_dna_2,normal_dna_1,normal_dna_2)

        print('**********get pep start**********')
        pep_length=list(args.length.split(","))
        print(f'pep length: {pep_length}')

        hla_list = args.hla.split(',')
        print(f'hla list: {hla_list}')

        get_snv_pep(pep_length,hla_list,patient_id)
        get_indel_pep(pep_length,hla_list,patient_id)
        snv_indel_combine(patient_id)

        print('**********feature start**********')
        feature_input(pep_length,hla_list,patient_id)
        feature_cal(pep_length,hla_list,patient_id)
        feature_merge(pep_length,hla_list,patient_id)

        print('**********predict start**********')
        predict_model(patient_id)

    elif args.pep_csv:
        print('**********feature start**********')
        hla_list = args.hla.split(',')
        print(f'hla list: {hla_list}')

        feature_input2(args.pep_csv,hla_list,patient_id)
        feature_cal2(args.pep_csv,hla_list,patient_id)
        feature_merge2(args.pep_csv,hla_list,patient_id)

        print('**********predict start**********')
        predict_model(patient_id,args.pep_csv)
    else:
        rna_path = parameter['work_path'] + '/calspace/' + patient_id + '/2_rna'
        tpm_path = rna_path+'/tumor_gene.csv'
        if not os.path.exists(tpm_path):
            print('Missing TPM file, Missing TPM file !!!!!!')
        else:
            hla_list = args.hla.split(',')
            print(f'hla list: {hla_list}')

            pep_length=list(args.length.split(","))
            print(f'pep length: {pep_length}')

            snv = args.snv
            indel = args.indel

            mv_file(snv,indel,patient_id)
            
            if snv:
                get_snv_pep(pep_length,hla_list,patient_id,n=10,snv=snv)
            if indel:
                get_indel_pep(pep_length,hla_list,patient_id,n=10,indel=indel)

            snv_indel_combine2(snv,indel,patient_id)

            print('**********feature start**********')
            neo_path = parameter['work_path'] + '/calspace/' + patient_id + '/4_neo'
            pep_path = neo_path+'/neo_result.csv'
            feature_input2(pep_path,hla_list,patient_id)
            feature_cal2(pep_path,hla_list,patient_id)
            feature_merge2(pep_path,hla_list,patient_id)

            print('**********predict start**********')
            predict_model(patient_id)
    print('**********ALL done**********')



if __name__ == '__main__':
    print('+----------------------------------------------+')
    print('|     ____  _   ___   __     _   ____________  |')
    print('|    / __ \/ | / / | / /    / | / / ____/ __ \\ |')
    print('|   / /_/ /  |/ /  |/ /    /  |/ / __/ / / / / |')
    print('|  / _, _/ /|  / /|  /    / /|  / /___/ /_/ /  |')
    print('| /_/ |_/_/ |_/_/ |_/____/_/ |_/_____/\____/   |')
    print('|                  /_____/                     |')
    print('+----------------------------------------------+')
    main()
