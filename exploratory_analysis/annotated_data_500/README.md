
# Run the following in ../annotations/2.0


## raw annotations in ./output

1. analyze_round2.py --read --store $target_file --> creates $target_file.json (for easy processing) and annotation pretty_$target_file.txt (for easy reading)
  -> gets json and pretty

2. analyze_round2.py --iaa --source $target_file.json --> does some IAA analysis
  -> gets pairwise detailed agreement

3. experiment1.py --disentangle --source $target_file.json --> does some analysis on annotation stats on different levels per outlet type (left, center, right)
  -> gets item/entity_stats/agreement per outlet bias type

4. experiment2_entities.py --entities --source $target_file.json --> looks at the actual extracted entities per entity type across outlet types (left, center, right) 
  -> gets entities + agreement (*very preliminary*)


**output excludes Victoria**
**annotations for N=502 articles (including NOT_RELEVANTs)**
- 330 articles with 2 annotations!
- 140 articles with 3 annotations!
- 32 articles with 1 annotation!

  
5. utils.py --> creates majority votes

- articles with >=2 annotations where at least two agree 
- N = 416 (excluding NOT_RELEVANTs)

# Copy all article txts into ./articles

6. ./copy_articles.txt --source $target_file --> takes json, gets IDs, copies articles from ../annotations/2.0/articles
