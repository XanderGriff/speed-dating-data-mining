# speed-dating-data-mining
A data mining experiment using Python (+data mining libs) &amp; Jupyter Notebook to gain insight into the inner workings of the dating process







CMPE 333 Project Report
By: Alexander Griff (10094348)

Table of Contents:

Table of Contents:	1
Section 1: Introduction:	2
Section 2: Approach:	2
Section 3: Procedure:	2
Phase 1: Background Research:	2
Phase 2: Data Understanding & Clustering	2
Phase 3: Data Preparation & Prediction in KNIME	3
Phase 4: Data Preparation & Prediction in Jupyter Notebook	4
Section 4: Conclusion	12
Bibliography:	13
Appendix A: KNIME Workflows	14
Appendix B: Jupyter Notebook Code & Results	17



Section 1: Introduction:
For this project, data mining principles and concepts will be applied to the process of speed dating. Using an extensive dataset taken from a speed dating event, the aim of this project is to produce a predictive model to accurately classify the compatibility of a pair of speed daters.

Section 2: Approach:
Based on course content and experience with course assignments, the Cross-Industry Standard Process for Data Mining (CRISP-DM) problem solving procedure has been adopted and adapted for this assignment. The main phases of the adapted CRISP-DM used for this project are as follows [1]:

Background Research
Data Understanding & Clustering
Data Preparation
Modeling
Evaluation
Deployment

Movement between phases works both forward and backward and movement can be multi step (ie. one can move from, say, the evaluation phase back to the business understanding phase). The flow of state between the phases is described above in Figure 1. To apply this approach in practice, the KNIME Analytics Platform, was proposed for use and used initially based on familiarity through previous course exercises. However, Jupyter Notebook was ultimately selected as the tool of choice for its versatility and integration with Python and accompanying libraries Pandas, Numpy and Scikit. 

Section 3: Procedure:
Phase 1: Background Research:
The first step in the procedure was to learn more about the dataset through background research. Although concepts covered within the bounds of the course syllabus provided an adequate technical background to devise an approach for this task, no contextual information was provided. The context of the dataset would  aid in informing initial hypotheses and conclusions.

This dataset was found to have been used as the basis of a research paper on gender-based preferences in dating, named Gender Differences in Mate Selection, that was published in 2006. The data was gathered over a two year period in a number of concentrated sessions from 2002-2004. Each pairing within a session lasted approximately four minutes in duration [3]. 

Phase 2: Data Understanding & Clustering
With access to and understanding of the dataset contextual information, the next step in the procedure involved finding out more about the actual dataset. Data formatting restrictions inherent to KNIME inhibited the application of clustering techniques on unprepared data (ie. formatting restrictions, datatype restrictions, etc.). Thus, the preliminary phase of data understanding came in the form of research. Luckily, the dataset was provided with an accompanying key-document. In conjunction with actually viewing the raw data, this document aided in achieving the sought-after semantic understanding. 

On top of learning about the basic information regarding attribute naming and ranges, a number of characteristics were identified from the key-document. Firstly, the preference scales varied between two different formats, which would provide a challenge of normalizing the two scales for comparison. Next, there were found to be slight variations (eg. different MC, clientele, etc.) between waves that might introduce some distorting factor into the dataset. This distortion would have to be considered when defining clusters and drawing conclusions based on wave differences. Also, a number of the fields defined in the sign-up phase were similar. The range of values within this attribute could therefore potentially be reduced to reduce complexity of computation and prediction. However, special attention would have to be paid to the reduction process to ensure that no pertinent data dimensionality was lost. From a contextual perspective, there was also the added information on the timing of the two post-event surveys, which were filled out the day after the event and three to four weeks after the event, respectively.

With a better understanding of the some of the challenges posed by the dataset characteristics, it was now necessary to take a look at the actual dataset. To view the dataset, the CSV file provided was read in KNIME using an interactive table. This provided a comprehensive view of the dataset. Many cells in the dataset were blank, posing another challenge of dealing with missing data, as described in the project outline. Also, some attributes had text-based inputs, which would provide a challenge of discretization. Finally, numerical values varied significantly. Normalization would therefore be of use in eliminating any error caused by this variation. However, normalization techniques would have to be employed to ensure data is not distorted (eg. SAT scores range from 400 to 1600, which would affect normalization outcomes if range was not considered). With an understanding of both the semantics of the dataset and the primary challenges for data preparation, it was possible to move forward to the data preparation phase.

Phase 3: Data Preparation & Prediction in KNIME
Based on the identified challenges, a KNIME workflow was initiated and implemented to process the data such that it would be more suitable for clustering and modeling techniques. Min/max normalization, over a range [0,1], and column filtering, whereby attributes that were less than 50% blank were filtered out, were implemented. Blank entries from all other columns were filled in with the most frequent attribute value. This preliminary blank-data algorithm was set to be used as the basis for testing with more sophisticated algorithms as the project continues. Discretization proved to be more of a challenge than in past exercises due to the range of answers (eg. 259 unique attribute values in field column), but was implemented using naive methodology of applying a binning node. An example preprocessing KNIME workflow can be seen below in Figure 2, and all workflows employed for data preparation can be found in Appendix A.


Figure 2: KNIME Workflow for String Manipulation

Phase 4: Data Preparation & Prediction in Jupyter Notebook
At this point in the project, the decision was made to switch tools and use Jupyter Notebook, rather than KNIME, based on its use in industry and its ease of implementation with code-based solutions in Python. Essentially, more customizability was needed than the KNIME platform could provide. Thus, all following work shown was done in Jupyter. While selected results are displayed throughout the report, more detailed and complete results, along with all code used to generate the results, can be found in Appendix B.

After importing the data into the Jupyter environment, it was possible to view a summary of attribute values. Using the describe() function, metrics including count, standard deviation, and mean were found for all numeric attributes. A sample of these metrics can be found below in Table 1:

Table 1: Describe Metrics for Dataset


iid
id
gender
Count
8378.000000
8377.000000
8378.000000
Mean
283.675937
8.960248
0.500597
Standard Deviation
158.583367
5.491329
0.500029
Minimum Value
1.000000
1.000000
0.000000
Maximum Value
552.000000
22.000000
1.000000

Having now taken a look at the preview of the dataset along with some cursory metrics, it was possible to go through the first iteration of the CRISP-DM procedure with the goal of establishing a baseline prediction model using a naive approach. The idea here being that any subsequent applied data processing techniques will be inherently more involved and thus more costly than a naive approach. Thus, if these techniques are to be applied, they must be justifiably useful. 

The naive approach, designed for simplicity of implementation rather than performance, is defined here as a series of steps to remove any problematic aspects of the dataset. Given that any missing values would prove problematic for prediction, any column attributes with missing data were removed. On top of that, it was theorized that rows from the series of waves in which a secondary preference scale was used would also have to be removed. However, on inspection of the missing-data-adjusted dataset, it was found that no columns pertinent to the scoring process remained. In fact, only 13 columns of the initial 195 columns had no missing data.

Clearly, there was a significant reduction to the information content of the dataset, and thus expectations for prediction accuracy were low. However, given as this was a baseline test, it would leave a great deal of room for improvement. This initially defined method for establishing a baseline predictive model, however, did not end up working as planned. Of the 13 attributes in this initial naive set, 3 attributes (match, dec_o and dec) were not available for model generation as they were either the target attribute for prediction or a direct function of the target attribute. This left the naive set with a mere 10 attributes, many of which were used procedurally during data collection and had no real influence on the experience of the speed daters (iid, idg, wave, round, position, etc.).

Thus, a more involved method would have to be adopted for establishing even the simplest of models. To do this, a predict method was crafted that takes as input a dataset and outputs a confusion matrix based on a Random Forest learning model applied to the input dataset. Random Forest learning was selected due to its strength in handling large numbers of attributes, which is one of the most prominent features of this dataset [4]. This is due to its inherent process of sampling with replacement, allowing it to see subsets of the dataset and determine which subsets are most prominent for prediction. The number of trees used in the model generation process was experimented with, and n=500 trees was settled on based on its optimality of prediction time and performance. 

Using this model, the initial dataset would be simplified based on the error messages given as the function fails due to dataset complications. For example, when the predict method was applied to the initial dataset, it provided an error message notifying that it could not take string values as attribute values. Thus, any column that had string values was removed. The next run complained about missing values, so missing values were imputed using each columns mean value without regard for appropriateness. When this new dataset was passed through the predict method, it succeeded in running, and provided the following confusion matrix with a predictive accuracy of approx. 84%: 


Figure 3: Baseline Confusion Matrix with 84% Prediction Accuracy

With a base level of predictive accuracy found to be about 84% it was now possible to move forward to the next iteration of the CRISP-DM process. That is to say, it was time to look at the dataset and see if improvements could be made to any of the steps taken to create the predictive model.

The first step was data preprocessing, which required a deeper analysis of the dataset documentation. Going through each attribute, it was possible to find more performant methods of imputation, discretization and normalization than applied in the naive approach. This involves the somewhat arduous task of checking the state of each attribute column for missing values, outliers, mistakes, and inconsistencies between waves. The table generated with the .describe() method above was particularly useful for determining outliers by checking the minimum and maximum values with the allowed values in the data key specifications. 

As there were almost 200 attributes in the dataset, some methodology had to be defined to approach preprocessing, so as to prioritize the most important features and minimize unnecessary work. The first metric that informed the methodology of approach was the percent of missing values in an attribute column. If the majority of the values of a given attribute were missing, there would be no point in trying to impute values as it increase the likelihood of inaccuracy to an unacceptable degree. Another metric to consider was attribute correlation. Ideally, attributes have high correlation to the attribute being predicted and low correlation with each other. If any attributes were extremely correlated with one another, it might be possible to remove them. However, as correlation is non-transitive, one cannot eliminate attributes based on high correlation alone, and must consider contextual information and percent- missing values of the attributes as well [5]. These two metrics were found using methods developed in Python, called find_correlations() and sorted_list_of_missing(), respectively. Although the results of these two methods were quite large and difficult to display, samples of the results from the correlation and missing value percentages are shown below in Tables 2 and 3:

Table 2: Three Attributes with the Highest Correlation with Match
Attribute
Correlation with Match
like_o         
0.305853
like
0.305723
fun_o
0.277700

Table 3: Three Attributes with the Highest Percent of Missing Values
Attribute
Correlation with Match
num_in_3
92%
numdat_3
82%
expnum
79%

During research into defining the acceptable percent of missing data, an paper was found on methods of imputation, which mentioned a technique called "Last observation carried forward," which entails using data from the past to inform missing follow-up data [6]. This paradigm was hypothesized to prove particularly useful for 'follow up' attributes where data was missing, as 'initial' attributes had much lower rates of missing data.

At this point, however, the correlation of attributes to one another had to be considered. Because the followup attributes measure the same metric varied over time, it was hypothesized that they had relatively high correlations with one another and, combined with the fact that they were missing a majority of the time based on observation, they could be eliminated rather than imputed. To confirm this was the case, a multiple step process was necessary. First, although initial observation of the percent-missing values of the followup attributes seemed relatively high, this observation had to be confirmed with actual data. Thus, the average percent-missing value was calculated for any attribute with the suffix '_2' or '_3'. A sample of the output from this calculation can be found below in Figure 4. 


Figure 4: Correlation Between Initial and Follow Up Attributes for attr1_1

Given that the percent-missing average is 50%, it's clear that a significant amount of values are missing from follow up attributes. If it can be proven that these attributes are correlated to their 'initial' counterparts, they can be sensibly eliminated. To do this, one of the attributes, attractiveness, was selected as a proxy for all attributes, and the attributes with the highest correlations were found for each of the five followup categories. Seeing as the most attributes most correlated with the 'initial' attribute were, for the most part, the 'follow up' attributes, these attributes could be eliminated from the dataset due to their high missing value rate and high correlation to initial attributes.

With followup attributes eliminated, the next step in the preprocessing phase involved normalizing the scores of the attribute values. Namely, the budget imposed on participants in Wave 12 and the different scales used in Waves 6-9. The budget limitation present in Wave 12, which dictated that a participant was able to "say yes" to only 50% of people they met during the session. This imposed inherent psychological conditions on the participants that was not present in other waves. Thus, there was no method for normalizing this data, and it had to be omitted.

The point scale difference in Waves 6 through 9 is a more complex issue to deal with, as the only change in conditions for the participants was the scale upon which they would rate their matches. Namely, the scoring was changed from a 1-100 to a 1-10 score for these rounds. Thus, it would seem as though normalization techniques could be applied. Although methods like z-score normalization were discussed in class, this problem was slightly different. Rather than normalizing for a scale, the goal was to normalize scoring methods. The scores could not simply be scaled by a factor of 10 due to differences in point allocation. The 1-10 scale involved simply ranking one's match out of 10 on each of the six judged attributes, while the 1-100 scale forced the participants to allocate a budget of 100 points across all six of the judged attributes. An argument could be made here that, similar to in Wave 12, a change in the nature of scoring could be a problematic influence on the way participants scored their matches. However, both scoring methods required the participants to rank their matches on a limited scale, so if one method of scoring could be converted to the other, they would, hypothetically, be quite similar indeed. To convert the 1-10 scores to 1-100 scores, an algorithm was generated that accounts for the 100 point budget. The algorithm works as follows: 

For each of the six categories, a 1-10 ranking is given. Thus, in a sense there are a total of 60 points available for allocation, although not all must be used. However, as the goal is simply to find the relative scores, the ratio of the fraction of the score of one category to the total number of points given in all categories would be an accurate representation of the relative score in that category. For example, given a score of [6,7,8,5,9,7] for each of the six categories, the algorithm would work as follows:

Total Score = 6 + 7 + 8 + 5 + 9 + 7 = 42

6 / 42 = 14%
7 / 42 = 16%
8 / 42 = 19%
5 / 42 = 12%
9 / 42 = 21%
7 / 42 = 16%

Given that there are 100 total points, point allocation in each category would exactly map to the percentages calculated above. Thus, the output of the algorithm would be: [14,16,19,12,21,16]. A particularly observant reader would notice that the total of the above allocation is 98. This is due to introduction of error made during the scaling process. However, minor discrepancies in overall point totals far outweigh prior
differences due to scoring inconsistencies. During testing of the algorithm, another inconsistency was found. As it turns out, despite the data key specifically denoting which columns used the different scoring method during Waves 6 through 9, some column attributes actually already used the scoring method from Waves 1 through 5 and Wave 10 onwards. However, the conversion algorithm is built in a way that does not to change the values of any of the relative scores, and thus it could be applied safely to the expected column attributes in case the alternate scoring method was used.

With adjustments made to the scoring system as detailed above, it was time to evaluate the changes to ensure that they had a net positive effect. The same methods of imputation and prediction for the naive model were applied to this newly preprocessed dataset. As seen below in Figure 5, the prediction accuracy improved by almost 2%, reflecting the improvements made to the dataset. It was also noted that the generated predictive model had a much tougher time dealing with predicting matches than non-matches. It was hypothesized that it might therefore prove useful to explore the rows where matches occur and see if any conclusions can be drawn from clustering methods.


Figure 5: Confusion Matrix with Predictive Accuracy for Updated Dataset

To further explore the relation of various attributes to matches, a number of methods were theorized to be useful based on the correlation values found using the find_correlations() method developed above. The first was to explore the correlations between matches and the six attributes used to score partner's attractiveness, sincerity, fun, etc.). Similarly, correlations would also be explored between matches and interests. Cursory explorations of the correlation between the score and interest attributes can be found below in Figures 6 and 7:


Figure 6: Correlation between Partner Scores and Match Attribute


Figure 7: Correlation between Interest Rankings and Match Attribute

In trying to understand correlations between various attributes, it was theorized that a possible method for imputation would be to add a degree of intelligence to the imputation process by first considering possible clusters and imputing based on the mean of those clusters, rather than the mean of the whole dataset. For example, considering that different genders or different cultural groups may have different correlations between the above interests and matches, missing values in these categories could be imputed from the means of these groups rather than the attribute mean for the entire dataset.

First, however, it would be necessary to observe whether or not belonging to these groups actually had an impact on the correlation values. To do this, the parallel visualization from course material was employed, with color coded lines to indicate belonging to either a gender or racial group. These visualizations can be found below in Figures 8 through 11:


Figure 8: Parallel Visualization of Partner Scored Attributes by Gender


Figure 9: Parallel Visualization of Interests by Gender


Figure 10: Parallel Visualization of Partner Scored Attributes by Race


Figure 11: Parallel Visualization of Interests by Race

Based on the differences in shape and magnitude in Figures 8 through 11, it is possible to come to the conclusion that there are different patterns of attribute correlation in different clusters of gender and race. Although some are significantly more pronounced, all attributes show some level of differentiation. It should be noted that, when dealing with gender or racial patterns, ethical consciousness must be considered. The gender and racial patterns described here in no way reflect any judgements or statements on external cultural groups as a whole, and are used exclusively to better impute data within these small and nonrepresentative subsets of the speed dating cohort.

Generally speaking, the preferences of the six partner scored attributes followed the same trends in both gender and racial groups, although with slight variations in magnitude and direction. Interests, on the other hand, differed quite significantly between both gender and racial groups. From a gender perspective, the greatest differences of correlations were in the tv, tv_sports, theatre, and movies attributes and the greatest similarities were between music, shopping, and yoga. From a racial perspective, each group held slightly different values, with hiking being the attribute with the greatest variance. Based on these observations, imputation can be broken down and tailored to specifically clustered demographics. To confirm that this change in process led to a neutral or better outcome, prediction accuracy found using the predict method from the naive approach was compared with that of the base model. A confusion matrix generated from this method is shown below in Figure 12:


Figure 11: Confusion Matrix for Further Improved Dataset

Clearly, the above improvements produced a marked increase in prediction accuracy, and can thus be considered successful. Some values, however, were still not imputed, as no mean could be calculated for the given cluster. Five attributes in particular, mn_sat, tuition, income, undergra and zipcode, all had above 10% and up to 64% missing values, while all other attributes were almost 99% or more complete. Thus, it was hypothesized that these five attributes would not be able to be imputed, and thus must be eliminated. To ensure that this did not negatively impact the model generation process, prediction accuracy was again calculated and compared to the base model. The generated confusion matrix generated is shown below in Figure 12:


Figure 12: Confusion Matrix for Final Improved Dataset

Section 4: Conclusion
Prediction accuracy with the five removed attributes removed increased again, this time by 0.4%. The rate of improvement of the generated models had been decreasing with each subsequent improvement as the dataset was continually improved, as seen below in Table 4. As the changes were reaching the point where they could be considered negligible, it seemed like an appropriate place to conclude the data mining process for this project and reflect on the progress made thus far.

Table 4: Prediction Accuracies of Models Generated Throughout the Data Mining Process
Data Mining Strategy
Prediction Accuracy
Naive Method
84.67%
Data Normalization & Imputation
85.98%
Data Imputation Based on Clustering
87.58%
Data Removal
88.08%

Using and extending upon information learned in class, this project ran the gamut on data mining topics and applications. It involved the use of classification, clustering, processing and visualization techniques to come to a greater understanding of the dataset and the underlying forces that dictate how partners will match in a speed dating environment. It provided insight on how different groups of people interact with one another, and how people set their values when in comes to choosing a partner. 

It also provided insight into how the data mining procedure takes place. This being the first attempt at a full scale data mining , it proved useful in solidifying conceptual understanding and leaving room for further improvements. Looking back on the project, there are a few specific extensional pursuits that may have proven useful if this project were to be done again. While the decision to use the Random Forest predictor algorithm was based in a technical understanding of the various classification algorithms, it may have been interesting to experiment with other methods learned in class, like SVM or MLP for example. It may also have proven beneficial to execute further kernel methods on the dataset to synthesize extra attributes, as done in course exercises. Lastly, while in this case all of the follow up data was removed for technically sound reasons, it would have been interesting to pursue alternative methods for dealing with and visualizing time series data as discussed briefly in the report.

Overall, significant progress was made on all fronts. This project has provided an engaging platform for learning and applying course concepts, and will continue to be of reference in further data mining projects. 
Bibliography:
[1]"Cross-industry standard process for data mining", En.wikipedia.org, 2017. [Online]. Available: https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining. [Accessed: 01- Nov- 2017].

[2]"Image: Cross-industry standard process for data mining - Wikipedia", Google.ca, 2017. [Online]. Available: https://www.google.ca/imgres?imgurl=https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/220px-CRISP-DM_Process_Diagram.png&imgrefurl=https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining&h=220&w=220&tbnid=jThVR0pmEeGC-M:&tbnh=160&tbnw=160&usg=__K1OSAMgVOvgQPPstWckhbRfwbI4=&vet=10ahUKEwjqr-_r4aDXAhWRyIMKHaO5C2cQ9QEIKjAA..i&docid=kUiUPHPMxtixYM&sa=X&ved=0ahUKEwjqr-_r4aDXAhWRyIMKHaO5C2cQ9QEIKjAA#h=220&imgdii=HGST_C4sOkea8M:&tbnh=160&tbnw=160&vet=10ahUKEwjqr-_r4aDXAhWRyIMKHaO5C2cQ9QEIKjAA..i&w=220. [Accessed: 01- Nov- 2017].

[3]R. Fisman, S. Iyengar, E. Kamenica and I. Simonson, "Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment", The Quarterly Journal of Economics, vol. 121, no. 2, pp. 673-697, 2006.

[4]P. Burger, "non-transitivity of correlation", Phillip Burger, 2017. [Online]. Available: http://www.phillipburger.net/wordpress/non-transitivity-property-of-correlation/. [Accessed: 02- Dec- 2017].

[5]H. Kang, "The prevention and handling of the missing data", 2017.

[6]"Random forests - classification description", Stat.berkeley.edu, 2017. [Online]. Available: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm. [Accessed: 02- Dec- 2017].





Appendix A: KNIME Workflows


Figure 2: KNIME Preprocessing Workflow


To determine the most sensible method of discretization, the first textual attribute, ‘field’, was singled out as a test subject to determine scope. It was found that the range of the attribute was 259 discrete values, as shown in Figure 4, meaning it may be difficult to use dictionary replace binning, which is shown in Figure 5. As discussed in the report, it may be necessary to remove the attributes until some discretization algorithm is found to be more suitable. This process is shown below in Figure 6.



Figure 3: Discretization Analysis Workflow


Figure 4: Discretization Analysis Table Output Sample


Figure 5: Dictionary Replace Discretization Inefficient Implementation Workflow


Figure 6: Discretization Elimination Workflow 
