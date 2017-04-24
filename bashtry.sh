#!/bin/bash
news_list=('nytimes' 'wsj' 'washington')
for((i=0;i<3;i++))
{
python test.py ${news_list[$i]}
}
exit 0
