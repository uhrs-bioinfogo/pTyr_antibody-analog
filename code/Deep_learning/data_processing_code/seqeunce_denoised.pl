
use strict;
use warnings;

my $usage = qq{$0 input_R0 input_R1 input_R2 input_R3 input_R4 input_R1_R4_merge OUT_R0 OUT_R1 OUT_R2 OUT_R3 OUT_R4};
die "$usage\n" if scalar @ARGV != 11;
my ($R0,$R1,$R2,$R3,$R4,$input_R1_R4_merge,$OUT_R0,$OUT_R1,$OUT_R2,$OUT_R3,$OUT_R4) = @ARGV;

open ROUND0, "<", $R0;
open ROUND1, "<", $R1;
open ROUND2, "<", $R2;
open ROUND3, "<", $R3;
open ROUND4, "<", $R4;
open R1_R4_merge, "<",$input_R1_R4_merge;

open OUT_R0, ">", $OUT_R0;
open OUT_R1, ">", $OUT_R1;
open OUT_R2, ">", $OUT_R2;
open OUT_R3, ">", $OUT_R3;
open OUT_R4, ">", $OUT_R4;

my %R4;
my %R3;
my %R2;
my %R1;
my %R0;
my @numberR4;
my @numberR3;
my @numberR2;
my @numberR1;
my @number_merge;
my @numberR0;

while (my $liner0 = <ROUND0>)
{chomp $liner0;
 @numberR0 = split(/\t\t/,$liner0);
 $R0{$numberR0[0]} = $numberR0[1];
}
while (my $liner1 = <ROUND1>)
{chomp $liner1;
 @numberR1 = split(/\t\t/,$liner1);
 $R1{$numberR1[0]} = $numberR1[1];
}
 while(my $liner2 = <ROUND2>)
{chomp $liner2;
 @numberR2 = split(/\t\t/,$liner2);
 $R2{$numberR2[0]} = $numberR2[1];
}
while(my $liner3 = <ROUND3>)
{chomp $liner3;
 @numberR3 = split(/\t\t/,$liner3);
 $R3{$numberR3[0]} = $numberR3[1];
}
while(my $liner4 = <ROUND4>)
{chomp $liner4;
 @numberR4 = split(/\t\t/,$liner4);
 $R4{$numberR4[0]} = $numberR4[1];
}

foreach my $R0_key (keys %R0)
{if ($R0{$R0_key}<3)
    {
        if (defined $R1{$R0_key} and  $R1{$R0_key}<3)
        {
            if (defined $R2{$R0_key} and $R2{$R0_key}<3)
            {
                if (defined $R3{$R0_key} and $R3{$R0_key}<3)
                {
                    if (defined $R4{$R0_key} and $R4{$R0_key}<3)
                    {
                     print OUT_R0 $R0_key,"\t",$R0{$R0_key},"\n";
                     print OUT_R1 $R0_key,"\t",$R1{$R0_key},"\n";
                     print OUT_R2 $R0_key,"\t",$R2{$R0_key},"\n";
                     print OUT_R3 $R0_key,"\t",$R3{$R0_key},"\n";
                     print OUT_R4 $R0_key,"\t",$R4{$R0_key},"\n";

                    }

                }
            }
        }
    }
}
while(my $liner_merge = <R1_R4_merge>)
{chomp $liner_merge;
 @number_merge = split(/\t\t/,$liner_merge);
    if (defined $R0{$number_merge[0]} and $R0{$number_merge[0]} > 3)
    {print OUT_R0 $number_merge[0],"\t",$R0{$number_merge[0]},"\n";
        if (defined $R1{$number_merge[0]})
        {
        print OUT_R1 $number_merge[0],"\t",$R1{$number_merge[0]},"\n";
        }
        if (defined $R2{$number_merge[0]})
        {
        print OUT_R2 $number_merge[0],"\t",$R2{$number_merge[0]},"\n";
        }
        if (defined $R3{$number_merge[0]})
        {
        print OUT_R3 $number_merge[0],"\t",$R3{$number_merge[0]},"\n";
        }
        if (defined $R4{$number_merge[0]})
        {
        print OUT_R4 $number_merge[0],"\t",$R4{$number_merge[0]},"\n";
        }
    }
    elsif (defined $R1{$number_merge[0]} and $R1{$number_merge[0]} > 3)
    {print OUT_R1 $number_merge[0],"\t",$R1{$number_merge[0]},"\n";
        if (defined $R0{$number_merge[0]})
        {
        print OUT_R0 $number_merge[0],"\t",$R0{$number_merge[0]},"\n";
        }
        if (defined $R2{$number_merge[0]})
        {
        print OUT_R2 $number_merge[0],"\t",$R2{$number_merge[0]},"\n";
        }
        if (defined $R3{$number_merge[0]})
        {
        print OUT_R3 $number_merge[0],"\t",$R3{$number_merge[0]},"\n";
        }
        if (defined $R4{$number_merge[0]})
        {
        print OUT_R4 $number_merge[0],"\t",$R4{$number_merge[0]},"\n";
        }

    }
    elsif (defined $R2{$number_merge[0]} and $R2{$number_merge[0]} > 3)
    {print OUT_R2 $number_merge[0],"\t",$R2{$number_merge[0]},"\n";
        if (defined $R0{$number_merge[0]})
        {
        print OUT_R0 $number_merge[0],"\t",$R0{$number_merge[0]},"\n";
        }
        if (defined $R1{$number_merge[0]})
        {
        print OUT_R1 $number_merge[0],"\t",$R1{$number_merge[0]},"\n";
        }
        if (defined $R3{$number_merge[0]})
        {
        print OUT_R3 $number_merge[0],"\t",$R3{$number_merge[0]},"\n";
        }
        if (defined $R4{$number_merge[0]})
        {
        print OUT_R4 $number_merge[0],"\t",$R4{$number_merge[0]},"\n";
        }

    }
    elsif (defined $R3{$number_merge[0]} and $R3{$number_merge[0]} > 3)
    {print OUT_R3 $number_merge[0],"\t",$R3{$number_merge[0]},"\n";
        if (defined $R0{$number_merge[0]})
        {
        print OUT_R0 $number_merge[0],"\t",$R0{$number_merge[0]},"\n";
        }
        if (defined $R1{$number_merge[0]})
        {
        print OUT_R1 $number_merge[0],"\t",$R1{$number_merge[0]},"\n";
        }
        if (defined $R2{$number_merge[0]})
        {
        print OUT_R2 $number_merge[0],"\t",$R2{$number_merge[0]},"\n";
        }
        if (defined $R4{$number_merge[0]})
        {
        print OUT_R4 $number_merge[0],"\t",$R4{$number_merge[0]},"\n";
        }

    }
    elsif (defined $R4{$number_merge[0]} and $R4{$number_merge[0]} > 3)
    {print OUT_R4 $number_merge[0],"\t",$R4{$number_merge[0]},"\n";
        if (defined $R0{$number_merge[0]})
        {
        print OUT_R0 $number_merge[0],"\t",$R0{$number_merge[0]},"\n";
        }
        if (defined $R1{$number_merge[0]})
        {
        print OUT_R1 $number_merge[0],"\t",$R1{$number_merge[0]},"\n";
        }
        if (defined $R2{$number_merge[0]})
        {
        print OUT_R2 $number_merge[0],"\t",$R2{$number_merge[0]},"\n";
        }
        if (defined $R3{$number_merge[0]})
        {
        print OUT_R3 $number_merge[0],"\t",$R3{$number_merge[0]},"\n";
        }

    }
    else
    {
    next;
    }
}
#	if(defined $R1{$R1_key})
#	{
#		if($R1{$R1_key} > 4)
#		{
#		print OUT_R1 $R1_key,"\t",$R1{$R1_key},"\n";
#	    }
#
#	}
#	if(defined $R2{$R1_key})
#	{
#		if($R2{$R1_key} > 4)
#		{
#		print OUT_R2 $R1_key,"\t",$R2{$R1_key},"\n";
#	    }
#
#	}
#	if(defined $R3{$R1_key})
#	{
#		if($R3{$R1_key} > 4)
#		{
#		print OUT_R3 $R1_key,"\t",$R3{$R1_key},"\n";
#	    }
#
#	}
#	if(defined $R4{$R1_key})
#	{
#		if($R4{$R1_key} > 4)
#		{
#		print OUT_R4 $R1_key,"\t",$R4{$R1_key},"\n";
#	    }
#
#	}
#}
#	if (defined $R2{$numberR3[0]})
#	{
#		if($R2{$numberR3[0]} > 4)
#		 {print OUT_R2 $numberR3[0],"\t",$R2{$numberR3[0]},"\n";
#			if (defined $R1{$numberR3[0]})
#                        {
#                        print OUT_R1 $numberR3[0],"\t",$R1{$numberR3[0]},"\n";
#                        }
#		  print OUT_R3 $numberR3[0],"\t",$R3{$numberR3[0]},"\n";
#		  next;
#		 }
#	}
#	if ($R3{$numberR3[0]} > 4)
#	{print OUT_R3 $numberR3[0],"\t",$R3{$numberR3[0]},"\n";
#		if (defined $R1{$numberR3[0]})
#                {
#                print OUT_R1 $numberR3[0],"\t",$R1{$numberR3[0]},"\n";
#                }
#                if (defined $R2{$numberR3[0]})
#                {
#                print OUT_R2 $numberR3[0],"\t",$R2{$numberR3[0]},"\n";
#                }
#		next;
#	}
#}
close ROUND0;
close ROUND1;
close ROUND2;
close ROUND3;
close ROUND4;
close OUT_R0;
close OUT_R1;
close OUT_R2;
close OUT_R3;
close OUT_R4;
