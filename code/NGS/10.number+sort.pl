use warnings;
use strict;
												
my $usage = qq{$0 input_peptide out_fastq1};
die "$usage\n" if scalar @ARGV != 2;
my ($peptide, $peptideout) = @ARGV;

open PEPTIDEOUT,">","$ARGV[1]" or die;
open(PEPTIDE, $peptide) or die "Can't open $peptide\n";
my %h1;
my $c = <PEPTIDE>;
chomp $c;
$h1{$c} = 1;
while (my $read = <PEPTIDE>)
{chomp $read;
	if (defined $h1{$read})
	{
	$h1{$read} += 1;
	next;
	}
	else
	{
	$h1{$read} = 1;
	}
}

my @key=keys %h1;
my %h2;
foreach my $key (@key)
	{
        open PEPTIDEOUT,">>","$ARGV[1]" or die;
        print PEPTIDEOUT "$key\t$h1{$key}\n";
        close PEPTIDEOUT;
	$h2{$key} = $h1{$key};
	}

open PEPTIDEOUT2,">","$ARGV[1]" or die;
foreach my $key ( sort { $h2{$b} <=> $h2{$a} } keys %h2 )
{

        my $value = $h2{$key};

        open PEPTIDEOUT2,">>","$ARGV[1]" or die;
        print PEPTIDEOUT2 "$key\t\t$h2{$key}\n";
        close PEPTIDEOUT2;


}
