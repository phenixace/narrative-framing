## Annotation files

### Article text and metadata with aggregated frame annotations

articles_metadata.tsv
- article ID
- outlet
- date of publication
- mbfc bias
- full article text
- frame annotations (5 colums)


### Frame and entity annotations

role_and_frame_annoations.tsv
- article ID
- extracted entity
- entity's stakeholder category
- entity's assigned role 
- mbfc bias
- frame annotations (5 colums)

### Unaggregated annotations

full_annotations.json
- annotations for  all 21 binary questions (paper appendix A)
- Raw annotations are given ordered by annotator ID (1-4), with -1 indicating that an annotator did not label an article.


## Question to frame mappings

The subset of questions which were verified as predictive for one of the five frames, and the only ones used in this
paper, are:

CO: CO1 CO2, CO3 
EC: EC1, EC2, EC3
HI: HI1, HI2, HI5
MO: MO1, MO2
RE: RE1, RE5

Hero: RE6
Villain: RE7
Victim: HI3

## Annotation instructions and codebook
The two pdf files contain the instructions given to annotators.
