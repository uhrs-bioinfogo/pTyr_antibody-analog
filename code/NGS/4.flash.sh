for i in *rm_primer.paired.1.fastq;do /home/jianghaoqiang/anaconda3/bin/flash $i ${i%%.*}.paired.2.fastq -d ./data_join/ -o ${i%%rm*} -t 3 & done
