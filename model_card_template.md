# Model Card
## Model Details
This model is part of a machine learning piipeline that includes data preprocessing and model saving to a pickle file.
It uses a OneHotEncoder for catergories, and a LabelBinarizer to target the raw Census data for training the model.
### Model Date
This model was created and run in May of 2026.
### Model Version
It is the first version of this model, there are currently no other versions of this model.
The requirements.txt details the version and list of python modules and packages used to generate the model.
The highlight of which would be the scikit-learn module which used version ```1.5.1``` and the pandas library which used
version ```2.2.2``` The base python version used was ```3.14``` This model was created for a "Deploying a Machine 
Learning Pipeline" class for Udacity.

### Model Type
The model run is a random forest classifier. It was trained and processed with the scikit-learn module from python. The 
seed used for reproducibility was ```67``` 

## Intended Use
The intended use of this model is predicting using US Census data whether someone annual income with be above 50,000
dollars based on various attributes gathered from census data. Which might be useful to researchers for factoring in
various socioeconomic conditions that might lead to a higher income bracket, such as being married, employment in the
private or public sector, etc.

This model ideally would not be used for smaller scale decisions, like determining an individual's creditworthiness,
or determining an individual's interest rates for loans, rather is should be for larger scale economic simulations.

## Training Data
The model was trained using US Census data, on adults actively living in the Unites States. The data is non-proprietary
and is openly available on the US Census site. The biases contained in the model, will lean towards United States habits
only, and should not be used with high confidence outside the United States for making conclusions.

## Evaluation Data
The same US Census data was used to evaluate the model. Polling a random slice in the data on unique entries to ensure
a wide demographic of groups from the census data. It is a selection of data that was, critically, not used during the
training as to not skew the evaluation and scoring results. The data was preprocessed to drop unknown values, and values
that did not match the label passed in, in this case, salary.

## Metrics

Top 3 Workclasses for Salary

| Workclass         | Precision | Recall | FBeta  | Count |
|-------------------|-----------|--------|--------|-------|
| Private           | 1.0000    | 0.9992 | 0.9996 | 5686  |
| Self-emp-not-inc  | 1.0000    | 1.0000 | 1.0000 | 644   |
| Federal-gov       | 1.0000    | 1.0000 | 1.0000 | 240   |
| ...               | ...       | ...    | ...    | ...   |

The model performance was measured using a precision, recall, and Fbeta. The random slice output for the model 
scored the highest across many categories. Most consistently the workclass: Private, with large recurring counts across
multiple demographic categories. Implying that private sector workers 

Many perfect scores in all metrics appeared so the data still appears inconclusive.

## Ethical Considerations
Models like this could be used as a predictive source for an individual's creditworthiness, comparing someone to the sum
of the entire United States, which could be used for predatory loans based solely on your race, sex, or origin-country.
Providing a source of systematic injustice towards underrepresented, or incorrectly represented demographics. Harming 
their ability to secure lines of credit, rent, or worse. Thus, why this model should only be used for larger scale
economic simulations, and to target which demographics of the United States might need to most financial assistance.

## Caveats and Recommendations
A Caveat identified was the large number of perfect scores for salary. The narrow range of salary data collected from 
Census data might be limiting the model's predictive capabilities. To fix this I would recommend finding a larger scoped
salary data source, possibly from other publicly available or anonymized and aggregated sources. In the future one could
compare other North American Census data, or even a different countries data to draw more conclusions on what
socioeconomic factors might affect salary in different parts of the world. 
