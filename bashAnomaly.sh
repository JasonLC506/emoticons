#!/bin/bash
news_list=('nytimes' 'wsj' 'washington')
K_list=(20 40 60 80 100)
for((i=0;i<${#news_list[@]};i++))
{
  for((j=0;j<${#K_list[@]};j++))
  {
     news=${news_list[$i]}
     K=${K_list[$j]}
     echo $news
     echo $K
     nohup python AnomalyDetectProb_SMPrank.py $news $K >> logs/log_Anomaly_SMP_${news}_${K} &
  }
}
exit 0
