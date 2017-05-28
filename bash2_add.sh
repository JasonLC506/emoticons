#!/bin/bash
par_a=("authorship" "bodyfat" "calhousing" "cpu" "elevators" "fried" "glass" "housing" "iris" "pendigits" "segment" "stock" "vehicle" "vowel" "wine" "wisconsin")
#par_b=("RankPairPref.py" "DecisionTreeWeight.py" "LabelWiseRanking.py" "SMPrank.py" "CADrank.py" "KNNPlackettLuce.py" "KNNMallows.py")
for((i=0;i<${#par_a[@]};i++))
{
  #for((j=0;j<${#par_b[@]};j++))
  #{
     a=${par_a[$i]}
     #b=${par_b[$j]}
     echo $a
     #echo $b
     nohup python logLinear.py $a >> logs/log_logLinear.py_${a} &
  #}
}
exit 0
