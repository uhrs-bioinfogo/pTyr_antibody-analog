use warnings;
use strict;

my $usage = qq{$0 input_fastq1 input_fastq2 out_fastq1 out_fastq2};
die "$usage\n" if scalar @ARGV != 4;
my ($fastq1, $fastq2,$fq1,$fq2) = @ARGV;


open(FASTQ1, $fastq1) or die "Can't open $fastq1\n";
my %h1;
while (my $readid = <FASTQ1>) {
        chomp $readid;
        chomp (my $sequence  = <FASTQ1>);
        chomp (my $comment   = <FASTQ1>);
        chomp (my $quality   = <FASTQ1>);
	my @n=split / /,$readid;
	$h1{$n[0]}=$readid."\n".$sequence."\n".$comment."\n".$quality."\n";
}
close FASTQ1;

open(FASTQ2, $fastq2) or die "Can't open $fastq2\n";
my %h2;
while (my $readid = <FASTQ2>) {
	chomp $readid;
	chomp (my $sequence  = <FASTQ2>);
	chomp (my $comment   = <FASTQ2>);
	chomp (my $quality   = <FASTQ2>);
	my @n=split / /,$readid;
	$h2{$n[0]}=$readid."\n".$sequence."\n".$comment."\n".$quality."\n";
}
close FASTQ2;

open(OUT1,">$fq1") or die "Can't open $fq1\n";
open(OUT2,">$fq2") or die "Can't open $fq2\n";
foreach my $k (keys %h1){
	if (defined $h2{$k}){
		print OUT1 $h1{$k};
		print OUT2 $h2{$k};
	}
}
