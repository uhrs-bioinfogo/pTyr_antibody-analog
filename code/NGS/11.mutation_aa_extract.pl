use strict;
use warnings;

my $infile = $ARGV[0];#extendedFrags.fastq
my $outfile_forward = $ARGV[1];

open IN,  "<", $infile;
open OUT1, ">", $outfile_forward;

while (my $line = <IN>)
{#chomp $line;
	if($line =~ /^SASYKPVFV..ITDDLHFYVQDVETGTQLEKLMENMRNDIASHPPVEGSYAPRRGEFCIAK...GEW.RARVEKVE/)
	{my $a = substr ($line , 9 , 2 );
	 my $b = substr ($line , 61 , 3 );
	 my $c = substr ($line , 67 , 1 );
#	 my $d = substr ($line , 41 , 1 );
#	 my $e = substr ($line , 56 , 1 );
#	 my $f = substr ($line , 59 , 1 );

	 
	
	 print OUT1 $a,$b,$c,"\n";
	}
	else:
	{
	next;
	}
}
close IN;
close OUT1;
