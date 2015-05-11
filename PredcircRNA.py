# -*- coding: utf-8 -*-
import sys
import numpy as np
import gzip
import random
import os
import bz2
import cPickle
import pdb
import urllib
import argparse
from modshogun import CombinedFeatures, RealFeatures, BinaryLabels, MulticlassLabels
from modshogun import CombinedKernel, PolyKernel, CustomKernel
from modshogun import MKLClassification, LibSVM, MKLMulticlass, MulticlassAccuracy, MulticlassLibSVM, MulticlassOneVsRestStrategy, KernelMulticlassMachine
from modshogun import CombinedKernel, GaussianKernel, LinearKernel,PolyKernel
from modshogun import LibSVMFile, CSVFile
from modshogun import SerializableAsciiFile

FEATURE_DIR = 'features/'
DATA_DIR = 'data/'

def read_phastCons(phscore_file):
    if not os.path.exists(phscore_file):
        print 'downloading phylop file'
        url_add = 'http://rth.dk/resources/mirnasponge/data/placental_phylop46way.tar.gz'
        urllib.urlretrieve(url_add, phscore_file)

    fp = gzip.open(phscore_file, 'r')
    phscore_dict = {}
    tmp_dict = {}
    key = 'chr1'
    for line in fp:
        values = line.split()
        if line.find('chr') != -1:
            if len(tmp_dict):
                print key
                phscore_dict[key] = tmp_dict
                key = values[0]
                tmp_dict = {}
        else:
            tmp_list = []
            for index in range(1, len(values)):
                tmp_list.append(float(values[index]))
            
            tmp_dict[values[0]] = tmp_list
    phscore_dict[key] = tmp_dict    
    fp.close
    
    return phscore_dict

def point_overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def get_score_base_position(phscore_dict, chr_name, start, end):
    chr_cons = phscore_dict[chr_name]
    score = []
    for key in chr_cons.keys():
        values = key.split('_')
        cons_start = int(values[0])
        cons_end = int(values[1])
        resu = point_overlap(cons_start, cons_end, start, end)
        if resu:
            cons_score = chr_cons[key]
            if cons_start <= start and end <= cons_end:
                start_index = start - cons_start
                end_index = end - cons_start
            elif cons_start > start and end <= cons_end:
                start_index = 0
                end_index = end - cons_start
            elif cons_start <= start and end > cons_end:
                start_index =  start - cons_start
                end_index =  cons_end - cons_start
            elif start < cons_start and  cons_end < end:
                start_index = 0
                end_index = cons_end - cons_start        
          
            score = cons_score[start_index:end_index + 1]
            #is_exist = True
            break
        
        #if is_exist:
    return score


def get_tris():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com   
           
def get_tri_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    for val in tris:
        num = seq.count(val)
        tri_feature.append(float(num)/seq_len)
    return tri_feature

def get_seq_based_position(chr_n, start, end, whole_seq):
    seq = whole_seq[chr_n]
    extract_seq = seq[start - 1: end]
    return extract_seq

def calculate_GC_percent(seq):
    G_num = seq.count('G')
    C_num = seq.count('C')
    GC = G_num + C_num
    percent = float(GC)/len(seq)
    return percent

def read_data(data_file):
    data = []
    fea_len = 32768
    fp = open(data_file, 'r')
    for line in fp:
        tmp_data = [0] * fea_len
        values = line.split()
        for value in values:
            val = value.split(':')
            tmp_data[int(val[0])] = float(val[1])
        data.append(tmp_data)
    fp.close()
    return data

def keep_important_features_for_graph(fea_importance_file, graph_file, outfile):
    data = read_data(graph_file)
    fea_imp = []
    fp = open(fea_importance_file, 'r')
    num_line = 0
    for line in fp:
        values = line.split()
        fea_imp.append(int(values[1]))
        if num_line >99:
            break
        num_line = num_line + 1

    fp.close()
    data = np.array(data)
    data = data.transpose()
    train = data[fea_imp, :]
    data= []
    fw = open(outfile, 'w')
    train = train.transpose()
    for vals in train:
        for val in vals:
            fw.write('%0.6f\t' %val)
        fw.write('\n')


    fw.close()


def get_graph_feature(graph_file_feature):
    graph_feaures = []
    fp = open(graph_file_feature, 'r')
    for line in fp:
        values = line.split()
        tmp_fea = []
        for val in values:
            tmp_fea.append(float(val))
        graph_feaures.append(tmp_fea)
                
    fp.close()
    return graph_feaures

def get_processed_conservation_score(score, len_seq):
    conservation_feature = []
    con_num = 6
    thres = 0.3
    lower_threshold = 0.6
    higher_threshold = 0.9
    #len_seq = len(score)
    score_array = np.array(score)
    mean_score =  score_array.mean()
    mean_std =  score_array.std()
    max_score = score_array.max()
    lower_num = np.where( score_array > lower_threshold )[0].size
    higher_num = np.where( score_array <= higher_threshold)[0].size
    higher_num_03 = np.where( score_array >= 0.3)[0].size
    higher_num_09 = np.where( score_array >= higher_threshold)[0].size
    con_arr = (score_array >= thres)
    con_str=''
    for val in con_arr:
        if val:
            con_str = con_str + '1'
        else:
            con_str = con_str + '0'
    sat7_len = con_str.count('1111111')        
    sat_len = con_str.count('111111') 
    sat5_len = con_str.count('11111') 
    sat4_len = con_str.count('1111') 
    sat8_len = con_str.count('11111111')    
    conservation_feature.append(mean_score)
    conservation_feature.append(max_score)
    conservation_feature.append(mean_std)
    conservation_feature.append(float(lower_num)/len_seq)
    conservation_feature.append(float(higher_num)/len_seq)
    conservation_feature.append(float(higher_num_03)/len_seq)
    conservation_feature.append(float(higher_num_09)/len_seq)
    conservation_feature.append(float(sat_len)/len_seq)
    conservation_feature.append(float(sat5_len)/len_seq)
    conservation_feature.append(float(sat4_len)/len_seq)
    conservation_feature.append(float(sat7_len)/len_seq)
    conservation_feature.append(float(sat8_len)/len_seq)
    return conservation_feature

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))
    
def get_seq_for_circularRNA(circbase, whole_seq):
    fp = open(circbase, 'r')
    fw = open('circularna_fasta', 'w')
    for line in fp:
        values = line.split()
        chr_n = values[0]
        start = int(values[1])
        end = int(values[2])
        circurna_name = values[3]
        strand = values[5]
        seq = whole_seq[chr_n]
        extract_seq = seq[start: end]
        extract_seq = extract_seq.upper()
        if strand == '-':
            extract_seq =  reverse_complement(extract_seq)
        
        fw.write('>%s\t%s\t%s\t%s\t%s\n' %(circurna_name, chr_n, start, end, strand))
        fw.write('%s\n'%extract_seq)
            
    fp.close()
    fw.close()
    
def get_annotation_dictory_based_gencode(gencode_file):
    fp = gzip.open(gencode_file, 'r')
    all_gene_dict = {}
    for line in fp:
        line = line.replace(';', '')
        line = line.replace('"', '')
        values = line.split()
        if values[0][0:3] == 'chr' and values[2] == 'transcript':
            #dict_value = (int(values[3]), int(values[4]))
            chr = values[0]            
            strand = values[6]
            #gene_type = values[13]
            #gene_name = values[17]
            transcript_type = values[19]
            trainscript_id = values[11]
            #value_type = gene_name + ':' + gene_type 
            key = chr + '_' + strand
            all_gene_dict.setdefault(transcript_type, []).append((int(values[3]), int(values[4]), key, trainscript_id))
            
    
    fp.close()
    return all_gene_dict


def get_nagetive_data_mRNA(gencode_file,  whole_seq):
    nega_num = 1951
    all_gene_dict = get_annotation_dictory_based_gencode(gencode_file)
    
    for key in all_gene_dict.keys():
        print key
        if key == 'miRNA':
            continue
        genes = all_gene_dict[key]
        fw = open('training_nega_' + key, 'w')
        num_gene = len(genes)
        rand_inds = random.shuffle(range(0, num_gene))
        select_inds = rand_inds[:nega_num]
        for index in select_inds:
            val = genes[index]
            chr_strand = val[2].split('_')
            chr_n = chr_strand[0]
            strand = chr_strand[1]
            trans_id = val[3]
            start = val[0]
            end = val[1]
            if whole_seq.has_key(chr_n):
                seq = whole_seq[chr_n]
                extract_seq = seq[start - 1: end]
                if strand == '-':
                    extract_seq =  reverse_complement(extract_seq)
    
                fw.write('>%s\t%s\t%s\t%s\t%s\n' %(trans_id, chr_n, start, end, strand))  
                fw.write('%s\n'%extract_seq)


        fw.close()
    

def get_UAU_GCG_component(seq):
    compo_per = []
    AG_num = seq.count('AG')
    GT_num = seq.count('GT')
    GTAG_num = seq.count('GTAG')
    AGGT_num = seq.count('AGGT')
    compo_per.append(float(AGGT_num)/len(seq))
    compo_per.append(float(AG_num)/len(seq))
    compo_per.append(float(GT_num)/len(seq))
    compo_per.append(float(GTAG_num)/len(seq))
    return compo_per

def read_alu_database(alu_file):
    alu_dict = {}
    fp = gzip.open(alu_file, 'r')
    for line in fp:
        values = line.split()
        chr_name = values[0]
        strand = values[-1]
        key = chr_name + strand
        posi = values[1] + '_' + values[2]
        alu_dict.setdefault(key, []).append((posi, values[3]))
    
    fp.close()
    return alu_dict

def extract_alu_type(alu_database, start, end, chrN, strand):
    key = chrN + strand
    alu_list = []
    for val in alu_database[key]:
        posi = val[0].split('_')
        base_start = int(posi[0])
        base_end = int(posi[1])
        if point_overlap(base_start, base_end, start, end):
            alu_list.append(val[1])
    return alu_list

'''        
def extract_alu_feature(alu_database, start, end, chrN, strand):
    key = chrN + strand
    for val in alu_database[key]:
        posi = val[0].split('_')
        base_start = int(posi[0])
        base_end = int(posi[1])
        if point_overlap(base_start, base_end, start, end):
            return val[1]     
    return ''
'''
def extract_alu_feature(fasta_file, alu_database):
    include_alu_type = []
    falu = open(DATA_DIR + 'alu_type', 'r')
    for line in falu:
        include_alu_type.append(line[:-1])
    falu.close()
    
    fp = open(fasta_file, 'r')
    fw = open(FEATURE_DIR + fasta_file + '_alu', 'w')
    for line in fp:
        values = line.split()
        if line[0] == '>':
            alu_score = [0] * 37
            chr_name = values[1]
            start = int(values[2])
            end = int(values[3]) - 1
            len_seq = end - start + 1
            strand = values[4]
            alu_list = extract_alu_type(alu_database, start, end, chr_name, strand)
            if len(alu_list):
                for val in alu_list:
                    if val in include_alu_type:
                        posi = include_alu_type.index(val)
                        alu_score[posi] = alu_score[posi] + 1
            for score in alu_score:
                fw.write('%d\t' %(score))
            fw.write('\n')
    fw.close()
    fp.close()


def get_dinucletide():
    nucle_com = []
    chars = ['A', 'C', 'G', 'T']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    return  nucle_com 

def get_di_nucleotide_composition(dis, seq):
    seq_len = len(seq)
    di_feature = []
    for val in dis:
        num = seq.count(val)
        di_feature.append(float(num)/seq_len)
    return di_feature
    
def extract_feature_conservation_CCF(data_file, phscore_dict):
    fp= open(data_file, 'r')
    cons_file = FEATURE_DIR + data_file + '_cons'
    fw = open(cons_file, 'w')
    tri_file = FEATURE_DIR + data_file + '_tri'
    fw_tri = open(tri_file, 'w')
    tris = get_tris()
    for line in fp:
        values = line.split()
        if line[0] == '>':
            chr_name = values[1]
            start = int(values[2]) 
            end = int(values[3]) - 1
            len_seq = end - start + 1
            score = get_score_base_position(phscore_dict, chr_name, start, end)
            if len(score) < 1:
                cons_feature = [0] *12 
            else:
                cons_feature = get_processed_conservation_score(score, len_seq)
        else:
            seq = line[:-1]
            compo_perc = get_UAU_GCG_component(seq)
            gc_content = calculate_GC_percent(seq)
            rna_len = len(seq)
            for val in cons_feature:
                fw.write('%0.3f\t' %val)
            fw.write('\n')
            tri_fea = get_tri_nucleotide_composition(tris, seq)
            #tri_fea = get_di_nucleotide_composition(tris, seq)   
            for val in tri_fea:
                fw_tri.write('%0.3f\t' %val)
            for val in compo_perc:
                fw_tri.write('%0.3f\t' %val)
            fw_tri.write('%0.3f\t%d\t' %(gc_content, rna_len))
 
            fw_tri.write('\n') 
                        
    fp.close()
    fw.close()
    fw_tri.close()
    
    return cons_file, tri_file

def parse_tandem_feature(tandem_file, out_file):
    fp = open(tandem_file, 'r')

    parameter_flag = False
    fw = open(out_file + '_tmp', 'w')
    for line in fp:
        if line.find('Sequence:') != -1:
            fw.write('>%s' %line)
            parameter_flag = False
            continue

        if line.find('Parameters:') != -1:
            parameter_flag = True
            continue

        if parameter_flag and len(line) > 5 :
            fw.write('%s' %line)
    fp.close()
    fw.close()
    fw1 = open(out_file, 'w')
    fp1 = open(out_file + '_tmp', 'r')
    first_flag = True
    tmp = [0] * 12
    for line in fp1:
        values = line.split()
        if line[0] == '>':
            if not first_flag:
                fw1.write('%d\t' %tmp[0])
                if tmp[0] > 0:
                    for val in tmp[1:]:
                        fw1.write('%0.3f\t' %(float(val)/tmp[0]))
                else:
                    for val in tmp[1:]:
                        fw1.write('%0.3f\t' %val)
                fw1.write('\n')        
            seq_name = values[1]
            tmp = [0] * 12
            #fw1.write('%s:\t' %seq_name)
            first_flag = False
        else:
            #print line
            ext_vals = values[2:13]
            #print len(ext_vals)
            tmp[0] = tmp[0] + 1
            for index in range(11):
                tmp[index + 1] = tmp[index + 1] + float(ext_vals[index])
    
    fw1.write('%d\t' %tmp[0])
    if tmp[0] > 0:
        for val in tmp[1:]:
            fw1.write('%0.3f\t' %(float(val)/tmp[0]))
    else:
        for val in tmp[1:]:
            fw1.write('%0.3f\t' %val)
    fw1.write('\n')     
    
    os .remove(out_file + '_tmp')
                
    fp1.close()  
    fw1.close()

def read_snp_database(snp_file):
    snp_dict = {}
    chr_focus = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
     'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
     'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
     'chr22', 'chrX', 'chrY']
    strand_dict = {'1':'+', '-1':'-'}
    if not os.path.exists(snp_file):
        url_add = 'http://rth.dk/resources/mirnasponge/data/snp_mart.gz'
        urllib.urlretrieve(url_add, snp_file) 
    fp = gzip.open(snp_file, 'r')
    for line in fp:
        values = line.split()
        if len(values) < 5:
            continue
        #print line
        chr_name = 'chr' + values[2]
        if chr_name not in chr_focus:
            continue
        strand = strand_dict[values[-1]]
        key = chr_name + strand
        posi = int(values[3])
        snp_dict.setdefault(key, []).append(posi)
    
    fp.close()
    for key in snp_dict.keys():
        snp_dict[key].sort() 
   
    return snp_dict

def extract_snp_feature(fasta_file, snp_database):  
    fp = open(fasta_file, 'r')
    fw = open(FEATURE_DIR + fasta_file + '_snp', 'w')
    for line in fp:
        values = line.split()
        if line[0] == '>':
            chr_name = values[1]
            start = int(values[2])
            end = int(values[3])
            len_seq = end - start + 1
            strand = values[4]
            key = chr_name + strand
            num_snp = 0
            snp_list = snp_database[key]
            #snp_list.sort()
            for val in snp_list:
                if val > end:
                    break
                if val <= end and val >= start:
                    num_snp = num_snp + 1
            fw.write('%0.6f\t' %(float(num_snp)/len_seq))
            fw.write('\n')
    fw.close()         
    fp.close()

def get_alu_repeat_ORF_snp(seq_file):
    alu_file = FEATURE_DIR + seq_file + '_alu'
    fp = open(alu_file, 'r')
    alu_val = []
    for line in fp:
        values = line.split()
        sum_alu = 0
        for val in values:
            sum_alu = sum_alu + int(val)
        alu_val.append(sum_alu)

    fp.close()

    '''rna_len = []
    fp2= open(seq_file, 'r')
    for line in fp2:
        if line[0] == '>':
            continue
        else:
            rna_len.append(len(line[:-1]))

    fp2.close()
    '''
    os.remove(alu_file)
    
    orf_fea ={}
    orf_file = FEATURE_DIR +seq_file + '_ORF'
    fp = open(orf_file, 'r')
    #pdb.set_trace()
    for line in fp:
        values = line.split()
        gene_name = values[0]
        orf_len = int(values[2]) - int(values[1])
        orf_fea[gene_name] = orf_len
    os.remove(orf_file)
        
    snp_fea = []
    snp_file = FEATURE_DIR + seq_file + '_snp'
    fp = open(snp_file, 'r')
    for line in fp:
        values = line.split() 
        snp_fea.append(float(values[0]))

    fp.close()
    os.remove(snp_file)
    
    repeat_fea = []
    repeat_file = FEATURE_DIR + seq_file + '_repeat'
    fp1 = open(repeat_file, 'r')
    for line in fp1:
        values = line.split()
        repeat_fea.append(float(values[0]))
        
    fp1.close()
    os.remove(repeat_file)
    
    fp1 = open(seq_file, 'r')
    whole_fea_file = FEATURE_DIR + seq_file + '_repeat_orf_alu_snp'
    fw = open(whole_fea_file, 'w')
    index = 0
    for line in fp1:
        if line[0] == '>':
            name = line.split()[0]
            name = name[1:]
        else:
            #values = line.split()
            rna_length = len(line[:-1])
            repeat_length = repeat_fea[index]
            if orf_fea.has_key(name):
                orf_length = orf_fea[name]
            else:
                orf_length = 0
            snp_length = snp_fea[index]
            #pdb.set_trace()
            fw.write('%0.6f\t%0.6f\t%d\t%0.6f\t%0.6f\n' %(float(alu_val[index])/rna_length, float(repeat_length)/rna_length, orf_length, float(orf_length)/rna_length, snp_length))
            index = index + 1

    fw.close()
    
    return whole_fea_file

def run_trftandem_cmd(fasta_file):
    cli_str = 'data/trf404.linux64 ' + fasta_file + ' 2 7 7 80 10 50 500 -f -d'
    fex = os.popen(cli_str, 'r')
    fex.close()
    
    filelist = [ f for f in os.listdir(".") if f.endswith(".html") ]
    for f in filelist:
        os.remove(f)     
    
    trf_out_file = fasta_file + '.2.7.7.80.10.50.500.dat'
    out_file = FEATURE_DIR + fasta_file + '_repeat'
    parse_tandem_feature(trf_out_file, out_file)
    
    os.remove(trf_out_file)

def run_graphprot(fasta_file):
    gz_outfile = fasta_file + '.gz'
    cli_str = 'perl GraphProt-1.0.1/fasta2shrep_gspan.pl –seq-graph-t -nostr -stdout -fasta ' + fasta_file + '| gzip > ' + gz_outfile
    #cli_str = 'perl GraphProt-1.0.1/fasta2shrep_gspan.pl –seq-graph-t -nostr -stdout -fasta ' + fasta_file + '| gzip > ' + gz_outfile
    fex = os.popen(cli_str, 'r')
    fex.close()

    cli_str = 'GraphProt-1.0.1/EDeN/EDeN -a FEATURE -i ' + gz_outfile
    fex = os.popen(cli_str, 'r')
    fex.close()
    os.remove(gz_outfile)
    graph_out_file = FEATURE_DIR +fasta_file + '_graph_feature'
    keep_important_features_for_graph(FEATURE_DIR + 'graph_feature_importance', gz_outfile + '.feature', graph_out_file)
    os.remove(gz_outfile + '.feature')
    return graph_out_file
    
def run_txCdsPredict(fasta_file):
    out_file = FEATURE_DIR + fasta_file + '_ORF'
    cli_str = 'data/txCdsPredict ' + fasta_file + ' ' + out_file
    fex = os.popen(cli_str, 'r')
    fex.close()   
        
def get_hg19_sequence():
    chr_focus = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
     'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14',
     'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
     'chr22', 'chrX', 'chrY']
    
    sequences = {}
    dir1 = DATA_DIR + 'hg19_seq/'
    if not os.path.exists(dir1):
        os.mkdir(dir1)
    #for chr_name in chr_foc
    for chr_name in chr_focus:
        file_name = chr_name + '.fa.gz'
        if not os.path.exists(dir1 + file_name):
            print 'download genome sequence file'
            cli_str = 'rsync -avzP rsync://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/' + chr_name + '.fa.gz ' + dir1
            fex = os.popen(cli_str, 'r')
            fex.close()
        
        print 'file %s' %file_name
        fp = gzip.open(dir1 + file_name, 'r')
        sequence = ''
        for line in fp:
            if line[0] == '>':
                name = line.split()[0]
            else:
                sequence = sequence + line.split()[0]
        sequences[chr_name] =  sequence 
        fp.close()
    
    return sequences

def get_seq_for_RNA_bed(RNA_bed_file):
    whole_seq = get_hg19_sequence()
    fp = open(RNA_bed_file, 'r')
    fasta_file = RNA_bed_file + '_fasta'
    fw = open(fasta_file, 'w')
    for line in fp:
        values = line.split()
        chr_n = values[0]
        start = int(values[1])
        end = int(values[2])
        gene_name = values[4]
        strand = values[3]
        seq = whole_seq[chr_n]
        extract_seq = seq[start: end]
        extract_seq = extract_seq.upper()
        if strand == '-':
            extract_seq =  reverse_complement(extract_seq)
        
        fw.write('>%s\t%s\t%s\t%s\t%s\n' %(gene_name, chr_n, start, end, strand))
        fw.write('%s\n'%extract_seq)
            
    fp.close()
    fw.close()

def read_feature_data(data_file):
    data = []
    fp = open(data_file, 'r')
    for line in fp:
        values = line.split()
        tmp_data = []
        for value in values:
            tmp_data.append(float(value))
        data.append(tmp_data)
    fp.close()
    return np.array(data)

def get_normalized_values_by_column(array, fea_length, save_file_name):
    max_col =[-100000] * fea_length
    min_col = [100000] * fea_length
    for values in array:
        for index in range(len(values)):
            if values[index] > max_col[index]:
                max_col[index] = values[index]
            if values[index] < min_col[index]:
                min_col[index] = values[index]
    for values in array:
        for index in range(len(values)):
            if max_col[index] == min_col[index]:
                continue
            values[index] = float(values[index] - min_col[index])/(max_col[index] - min_col[index]) 
            
    fw = open(save_file_name, 'w')
    for val in min_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    for val in max_col:
        fw.write('%f\t' %val)
    fw.write('\n')
    fw.close()

def get_normalized_given_max_min(array, save_file_name):
    normalized_data = np.zeros(array.shape)
    tmp_data = np.loadtxt(save_file_name)
    min_col = tmp_data[0, :]
    max_col = tmp_data[1, :]
    for x in xrange(array.shape[0]):
        for y in xrange(array.shape[1]):  
            normalized_data[x][y] = float(array[x][y] - min_col[y])/(max_col[y] - min_col[y])
    return normalized_data

def save_model(filename, myobj):
    try:
        f = bz2.BZ2File(filename, 'wb')
    except:
        print 'open model error'
        return

    cPickle.dump(myobj, f, protocol=2)
    f.close()


def load_model(filename):
    try:
        f = bz2.BZ2File(filename, 'rb')
    except:
        print 'loading model error'
        return

    myobj = cPickle.load(f)
    f.close()
    return myobj    

def fuse_multi_view_features(fasta_file):
    print 'extracting repeat feature'
    run_trftandem_cmd(fasta_file)
    
    print 'extracting ORF feature using txCdsPredict'
    run_txCdsPredict(fasta_file)
    
    print 'extracting ALU feature'
    alu_database_file = DATA_DIR + 'alu_hg19.bed.gz'
    alu_database = read_alu_database(alu_database_file)
    extract_alu_feature(fasta_file, alu_database)
    alu_database.clear()
    print 'extracting snp feature'
    snp_data_base_file = DATA_DIR + 'snp_mart.gz'
    snp_dict = read_snp_database(snp_data_base_file)
    extract_snp_feature(fasta_file, snp_dict)
    snp_dict.clear()
    
    other_feature_file = get_alu_repeat_ORF_snp(fasta_file)
    
    print 'extracting conservation feature'
    phscore_file = DATA_DIR + 'placental_phylop46way.tar.gz'
    phscore_dict = read_phastCons(phscore_file)
    cons_file, tri_file = extract_feature_conservation_CCF(fasta_file, phscore_dict)
    phscore_dict.clear()
    
    print 'extracting graph feature'
    graph_file = run_graphprot(fasta_file)
    
    return graph_file, cons_file, tri_file, other_feature_file

def predict_new_data(graph_file, cons_file, tri_file, other_feature_file):    
    print 'reading extracted features'
    graph_feature = read_feature_data(graph_file)
    graph_feature = get_normalized_given_max_min(graph_feature, 'models/grtaph_max_size')
    cons_feature = read_feature_data(cons_file)
    cons_feature = get_normalized_given_max_min(cons_feature, 'models/cons_max_size')
    CC_feature = read_feature_data(tri_file)
    CC_feature = get_normalized_given_max_min(CC_feature, 'models/tri_max_size')
    ATOS_feature = read_feature_data(other_feature_file)
    ATOS_feature = get_normalized_given_max_min(ATOS_feature, 'models/alu_max_size')
    
    width, C, epsilon, num_threads, mkl_epsilon, mkl_norm = 0.5, 1.2, 1e-5, 1, 0.001, 3.5
    kernel = CombinedKernel()
    feats_train = CombinedFeatures()
    feats_test = CombinedFeatures()   

    #pdb.set_trace() 
    subkfeats_train = RealFeatures()
    subkfeats_test = RealFeatures(np.transpose(np.array(graph_feature)))
    subkernel = GaussianKernel(10, width) 
    feats_test.append_feature_obj(subkfeats_test)
    
    fstream = SerializableAsciiFile("models/graph.dat", "r")
    status = subkfeats_train.load_serializable(fstream)
    feats_train.append_feature_obj(subkfeats_train)
    kernel.append_kernel(subkernel)   

    subkfeats_train = RealFeatures()
    subkfeats_test = RealFeatures(np.transpose(np.array(cons_feature)))
    subkernel = GaussianKernel(10, width) 
    feats_test.append_feature_obj(subkfeats_test)
    
    fstream = SerializableAsciiFile("models/cons.dat", "r")
    status = subkfeats_train.load_serializable(fstream)
    feats_train.append_feature_obj(subkfeats_train)
    kernel.append_kernel(subkernel) 

    subkfeats_train = RealFeatures()
    subkfeats_test = RealFeatures(np.transpose(np.array(CC_feature)))
    subkernel = GaussianKernel(10, width) 
    feats_test.append_feature_obj(subkfeats_test)
    
    fstream = SerializableAsciiFile("models/tri.dat", "r")
    status = subkfeats_train.load_serializable(fstream)
    feats_train.append_feature_obj(subkfeats_train)
    kernel.append_kernel(subkernel) 
    
    subkfeats_train = RealFeatures()
    subkfeats_test = RealFeatures(np.transpose(np.array(ATOS_feature)))
    subkernel = GaussianKernel(10, width) 
    feats_test.append_feature_obj(subkfeats_test)
    
    fstream = SerializableAsciiFile("models/alu.dat", "r")
    status = subkfeats_train.load_serializable(fstream)
    feats_train.append_feature_obj(subkfeats_train)
    kernel.append_kernel(subkernel)

    model_file = "models/mkl.dat"
    if not os.path.exists(model_file):
        print 'downloading model file'
        url_add = 'http://rth.dk/resources/mirnasponge/data/mkl.dat'
        urllib.urlretrieve(url_add, model_file) 
    print 'loading trained model'
    fstream = SerializableAsciiFile("models/mkl.dat", "r")
    new_mkl= MKLClassification()
    status = new_mkl.load_serializable(fstream)
    
    print 'model predicting'
    kernel.init(feats_train, feats_test)
    new_mkl.set_kernel(kernel)
    y_out =  new_mkl.apply().get_labels()
    
    return y_out


parser = argparse.ArgumentParser(description="""computational classification of circular RNA from other long non-coding RNA using hybrid features""")

parser.add_argument('--inputfile', help='BED input file for transcript candidates, it should be like:chromosome    start    end    gene1', default='')
parser.add_argument('--outputfile', help='BED result file with corresponding lncRNA type in last column', default='')  
args = parser.parse_args()

if __name__ == "__main__":
    ''' It should be run as follows:
    python PredcircRNA.py --inputfile=test_bed --outputfile=test_bed_final
   
    ''' 
    print 'extracting transcript sequences'
    #input_bed_file = sys.argv[1]
    #output_file = sys.argv[2]
    input_bed_file = args.inputfile
    output_file = args.outputfile
    get_seq_for_RNA_bed(input_bed_file)
    
    print 'feature extraction'
    fasta_file = input_bed_file + '_fasta'
    graph_file, cons_file, tri_file, other_feature_file = fuse_multi_view_features(fasta_file)

    print 'predicting BED-format input transcripts'
    y_out = predict_new_data(graph_file, cons_file, tri_file, other_feature_file)
    
    fw = open(output_file, 'w')
    fp = open(input_bed_file, 'r')
    index = 0
    for line in fp:
        if y_out[index] == 1:
            gene_type = 'circularRNA'
        else:
            gene_type = 'other lncRNA'
        fw.write(line[:-1] + '\t' + gene_type + '\n')
        index = index + 1
    fp.close()
    fw.close()


