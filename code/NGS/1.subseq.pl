use warnings;
use strict;

my $usage = qq{$0 input_fastq primer};
die "$usage\n" if scalar @ARGV != 2;
my ($fastq, $primer) = @ARGV;

my %p;
open (PRIMER,$primer) or die "Can't open $primer\n";

while (<PRIMER>){
	chomp;
	my @l=split;
	$p{$l[1]}=$l[0];
}
close PRIMER;

open(FASTQ, $fastq) or die "Can't open $fastq\n";
while (my $readid = <FASTQ>) {
        chomp $readid;
        chomp (my $sequence  = <FASTQ>);
        chomp (my $comment   = <FASTQ>);
        chomp (my $quality   = <FASTQ>);
	foreach my $pri (keys %p){
		if ($sequence=~/^$pri/){
			open (OUT,">>$p{$pri}");
			print OUT qq{$readid\n$sequence\n$comment\n$quality\n};
			close OUT;
		}
	}

}
close FASTQ;
