use warnings;
use strict;

my $usage = qq{$0 primer outfile};
die "$usage\n" if scalar @ARGV != 2;
my ($primer,$out) = @ARGV;

open (PRIMER,$primer) or die "Can't open $primer\n";
open (OUT, ">$out");

while (<PRIMER>){
	chomp;
	my @l=split;
	my $p1=$l[0].".1.fq";
	my $p1_out=$l[0]."rm_primer.1.fq";
	my $p2=$l[0].".2.fq";
	my $p2_out=$l[0]."rm_primer.2.fq";
	my $seq="^".$l[1];
	print OUT "cutadapt -j 4 -g $seq -o $p1_out $p1\n";
	print OUT "cutadapt -j 4 -g $seq -o $p2_out $p2\n";
}
close PRIMER;

