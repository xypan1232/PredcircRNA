# PredcircRNA: computational classification of circular RNA from other long non-coding RNA using hybrid features

PredcircRNA, focused on distinguishing circularRNA from other lncRNAs using  
multiple kernel learning. Firstly we extracted different sources of discriminative features, including graph feature, conservation information and sequence 
compositions, ALU and tandem repeat, SNP density and open reading frame (ORF) from transcripts. Secondly, to better integrate features from different sources, we 
proposed a computational approach based on multiple kernel learning framework to fuse those heterogeneous features.
<br>

<b>Dependcy: </b><br>
1. GraphProt: http://www.bioinf.uni-freiburg.de/Software/GraphProt/ <br>
2. SHOTGUN: http://www.shogun-toolbox.org/  <br>
3. txCdsPredict: http://hgdownload.cse.ucsc.edu/admin/ <br>
4. Tandem repeats finder(trf): http://tandem.bu.edu/trf/trf.download.html <br>


<b>Input bed file format(such as test_bed): </b><br>
chr2	69304539	69318051	+	gene1 <br>
chr7	138593736	138597206	-	gene2 <br>
chr22	39134591	39137055	-	gene3 <br>

NOTICE: in the last column, we need have unique name (here is gene1, gene2...) for the transcript.<br><br>
<b>How to use the tool, the command as follows: </b><br>
python PredcircRNA.py --inputfile=test_bed --outputfile=test_bed_out
<br><br>
The output file have corresponding lncRNA type in last column.


<b>Reference</b> <br>
Xiaoyong Pan, Kai Xiong. PredcircRNA: computational classification of circular RNA from other long non-coding RNA using hybrid features. Mol. BioSyst., 2015, 11, 2219-2226
