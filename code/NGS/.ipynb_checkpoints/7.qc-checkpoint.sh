for i in *final.fq;do fastq_quality_filter -q 20 -p 80 -Q 33 -i $i -o ${i%.*}.qc.fq & done
