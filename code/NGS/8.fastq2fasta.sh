for i in *.final.qc.fq;do fastq_to_fasta -i $i -o ${i%.*}.fa -Q 33 & done
