#!/bin/bash
news_list=('nytimes' 'wsj' 'washington')
init_prop_list=(0.3 0.5 0.7)
for((i=0;i<${#news_list[@]};i++))
{
  for((j=0;j<${#init_prop_list[@]};j++))
  {
     news=${news_list[$i]}
     init_prop=${init_prop_list[$j]}
     echo $news
     echo $init_prop
     nohup python timeSeriesPredict_single.py $news $init_prop >> logs/log_timeSeries_IMG_${news}_${init_prop} &
  }
}
exit 0
