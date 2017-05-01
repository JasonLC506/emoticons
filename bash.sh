#!/bin/bash
par_a=(10 20 30 40 50)
par_b=(20 40 60 80 100)
for((i=0;i<${#par_a[@]};i++))
{
  for((j=0;j<${#par_b[@]};j++))
  {
     a=${par_a[$i]}
     b=${par_b[$j]}
     echo $a
     echo $b
     nohup python CADrank.py $a $b >> logs/log_CADrank_${a}_${b} &
  }
}
exit 0
