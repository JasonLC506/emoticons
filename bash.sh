#!/bin/bash
par_a=("nytimes" "wsj" "washington")
# par_b=(20 40 60 80 100)
for((i=0;i<${#par_a[@]};i++))
{
  #for((j=0;j<${#par_b[@]};j++))
  #{
     a=${par_a[$i]}
     # b=${par_b[$j]}
     echo $a
     # echo $b
     nohup python logLinear.py $a >> logs/log_logLinear_${a} &
  #}
}
exit 0
