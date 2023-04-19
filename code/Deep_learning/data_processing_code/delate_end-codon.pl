#!/usr/bin/perl
use strict;
use warnings;


my $usage = qq{$0 inputfile out};
die "$usage\n" if scalar @ARGV != 2;
my ($infile, $out) = @ARGV;

open IN,  "<", $infile;
open OUT, ">", $out;
while (my $line = <IN>)
{chomp $line;
my @line_split=split(/\t\t/,$line);
	if($line_split[0] =~ m/o/g)#~ m/C/g
	{
	next;
	}
	elsif($line_split[0] =~ m/u/g)
	{next;
#	 $line_split[0] =~ s/u/W/g
	}
	elsif($line_split[0] =~ m/a/g)
	{
	 $line_split[0] =~ s/a/Q/g;
	 print OUT $line_split[0],"\t\t",$line_split[1],"\n";
	}
	else
	{
	 print OUT $line,"\n";
	}
}
close OUT;