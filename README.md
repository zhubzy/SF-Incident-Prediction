# Group Project Final Report <San Francisco Incident Prediction Project>

## Introduction
The San Francisco Incident Prediction project represents a pivotal application of data science to enhance urban safety and improve policing strategies. By predicting the nature of crime incidents using temporal and spatial data, this initiative stands at the forefront of a proactive approach to public safety, with the aim of being both innovative and impactful, that also explains why we chose this as our final project topic.

### Project Significance

The relevance of this project is underscored by its direct impact on law enforcement agencies, urban planners, and communities. Predictive modeling of crime incidents based on nuanced factors such as time and place allows for precision in public safety measures, offering a tailored strategy for addressing specific criminal activities.

#### Data Utilization

The methodical use of time and location data, complemented by advanced feature engineering techniques, illustrates an astute deployment of available information. This project exemplifies the shift towards data-driven decision-making in urban management, replacing intuition-based methods with evidence-based practices.

#### Technical Rigor

A diverse array of modeling techniques, including logistic regression, neural networks, and SVM classifiers, demonstrates a rigorous analytical process aimed at identifying the most effective model for crime prediction. This approach highlights the adaptability and scalability of our methods, with potential applications extending to other urban settings.

### Broader Impact

Effective predictive models have a profound impact on multiple facets of urban life:

- **Proactive Policing**: Enhanced resource allocation and preemptive intervention capabilities.
- **Community Trust**: Strengthening the bond between law enforcement and the public through transparent, evidence-based practices.
- **Policy Making**: Guiding the development of targeted legislation and community initiatives.
- **Urban Planning**: Informing safer city designs through strategic planning and environmental design considerations.

### Importance of Robust Predictive Models

The necessity for accuracy, adaptability, and efficiency in predictive modeling is paramount. Such models are instrumental in focusing efforts where they are most effective, adjusting to evolving crime trends, and optimizing resource use. In essence, the San Francisco Incident Prediction project is more than an exploration of crime prediction; it's a blueprint for enhancing public safety through the application of technology and analytics, contributing to the creation of safer and more resilient communities.


## Methods

### Data Exploration

- Explain the initial dataset
- Discuss the characteristics of the data
- Include any insights gleaned from the exploration
- **Results / Figures**: Insert relevant figures with legends

### Preprocessing
- Detail the steps taken to preprocess the data
- Mention any data cleaning or transformation performed
- **Steps / Sample output of your final dataframe**: Show a snapshot of the data after preprocessing

#### Data Preprocessing for "San Francisco Incident Reports" Analysis

Google Collab link (May not be able to view notebook on Github due to size): https://colab.research.google.com/github/zhubzy/SF-Incident-Prediction/blob/main/SF_Incident_Prediction.ipynb

This README highlights the steps for preprocessing the "San Francisco Incident Reports (2018-present)" dataset for classifying incidents into "violent" and "non-violent" categories based on location and time information. The document will guide through data cleaning and transformation to prepare the dataset for analysis using feedforward neural networks.

#### Data Cleaning

##### 1. Handle Missing Values

- **Numerical columns**: Identify and fill missing values in columns like `longitude` and `latitude`. we will use the mean or median of the column for filling in the missing values.
- **Categorical columns**: For columns with categorical data such as `Intersection`, we will fill missing entries with the most frequent value or a placeholder such as 'Unknown'.

##### 2. Remove Outliers

- Use statistical methods to identify and remove outliers from numerical columns like `longitude` and `latitude`. We will use the Interquartile Range (IQR) method for this purpose.

#### Data Transformation

##### 1. Normalization

- We will normalize numerical features to ensure they're on the same scale. This is crucial for features like `longitude` and `latitude`. We will use the z-score normalization to achieve this.

##### 2. Extract Target Variable

- We will manually classify incidents into "violent" and "non-violent" categories using the `Incident Category` column. For example, we may classify "Larceny Theft" as a "non-violent" act and "Assault" as a "violent" act in the column.

##### 3. Categorical Feature Encoding

- We can use one-hot encoding technique for converting categorical data into a binary vector, such as converting "non-violent" to `0` and "violent" to `1`.

#### Splitting

##### 1. Split the data

- After completing the data cleaning and transformation steps, we will split the dataset into training and testing sets to be used by our models.


### Model 1 - Logisitic Regression
- Describe the first model, its hypothesis, and algorithm
- Mention the training process

We first finished pre-processing by extracting the dattime in string to numerical categories (day, month, year, etc) then applying z-score standardization to all the appropriate columns before the data is ready for training.

We then begin to build and experiment with a simple logistic regression model.



### Model 2 - Neural Network

- Describe the second model, its hypothesis, and algorithm
- Mention the training process

The data we choose is the pre-processed dataframe df_balanced. We didn't use the one-hot encoded version of dataset because it significantly increases the dimensionality of the dataset and the kernel cannot handle that huge feature space. We tried to run on different servers but it always shows "Kernel Dead". 

The neural network we built has a relatively good performance on predicting whether the incident is violent or not. Aiming for better accuracy, we performad hyperparameter tuning to optimize the configuration settings, including the number of nodes, optimizer, learning rate, identifying the best combination that minimizes the loss function and improves the model's accuracy and generalization ability on unseen data. We didn't use k-fold cross validation because it cannot effectively improves the performance of the model while making the training process especially computationally expensive. Similarly, although feature expansion has the potential to uncover non-linear relationships and improve model performance, it also risks leading to overfitting, where the model becomes too tailored to the training data and performs poorly on unseen data. Furthermore, feature expansion can exponentially increase the dimensionality of the data, exacerbating issues with memory usage and computational efficiency.


### Model 3

- Describe the third model, its hypothesis, and algorithm
- Mention the training process



## Results

### Model 1 - Logistic Regression Results / Figures
- Present the results of Model 1
- Discuss the findings
- **Figures**: Include graphs or charts that support the results
  
Due to class imbalance, we achieved 89% accuracy with the prediction tasks but along with 0% precision and 0% recall. Upon inspecting the confusion matrix we observe that the model is predicting any incident to be "non-violent". In this case accuracy is not a good metric of performance due to the class imbalance and we want to figure out contributing factors for a crime to be violent. Further data processing is needed (resampling the dataset to be balanced).
![](logistic_reg_model_unbalanced_train_set_confusion_matrix.png)
*Figure X: Description of what the figure represents.*

<img src="logistic_reg_model_unbalanced_train_set_confusion_matrix.png" alt="Description of the image" width="300" height="200">


In a second attempt, we changed our sample technique to account for this. On the new training set, we include an even 50-50 split of both classes from resampling, and we end up with an F1 score of 0.56 for these new tasks.
![](logistic_reg_model_balanced_train_set_confusion_matrix.png)

We then experimented with more feature extraction. We added additional features into our dataset by one hot encoding the intersection (so our model knows what community the crime is happening), and this raised the accuracy to 62%. Adding temporal features such as isWeekend, timeOfDay (morning, afternoon, evening) did not help improve the results.

![](logistic_reg_model_balanced_with_intersection_confusion_matrix.png)
*Figure X: Description of what the figure represents.*
![](logistic_reg_model_balanced_with_intersection_and_temporal_features_confusion_matrix.png)
*Figure X: Description of what the figure represents.*


### Model 2 - Neural Network Results / Figures

- Present the results of Model 2
- Discuss the findings
- **Figures**: Include graphs or charts that support the results

Where does your model fit in the ftting graph, how does it coripare to your frst model?

Based on the graph for model loss，we have a promising model that is learning effectively, evidenced by the consistent downward trend in training loss. The model displays a commendable ability to minimize error on the training set, indicating a good fit. Despite some fluctuations in validation loss, the general proximity of the training and validation losses suggest that the model has the potential for strong generalization with further tuning. This foundational training showcases a solid starting point for a robust model that, with further refinement, is poised to offer reliable predictions.

![](neural_network_model_loss.png)
*Figure X: Description of what the figure represents.*

### Model 3 Results / Figures

- Present the results of Model 3
- Discuss the findings
- **Figures**: Include graphs or charts that support the results



## Discussion

- Provide a comprehensive analysis of the results
- Compare the models and discuss the merits and demerits of each



## Conclusion

- Summarize the main findings
- State the conclusions drawn from the analysis
- Suggest possible future work or improvements

### Model 1 (Logistic Regression): 

In conclusion, with some feature engineering, we are able to achieve a F1 score of 0.63 with our simple logistic regression model. Through this experiment, we found out that the broad location of crime (intersection) seems to be an important contributing factor to whether a crime is violent or not as one hot encoding it increased our performance metrics (F1 score) marginally.

Logistic regression assumes a linear relationship between the features and the log-odds of the target variable. If the relationship is non-linear, logistic regression may not perform well. For the next two models, we could incorporate non-linear transformations of the features or use more complex models like decision trees or neural networks. Crime patterns can vary significantly over time and across different locations. If the model doesn't account for these temporal and spatial dynamics adequately, its predictive performance may suffer. In addition, we can consider incorporating time and location-specific features or using techniques like spatial or temporal clustering.

For the next two models, we want to try SVM classifiers and neural networks, as they are well-suited for complex datasets with high-dimensional features. They can effectively handle datasets with a large number of features, which we would expect to have after more feature extraction and engineering just like we did this milestone with the intersection.

### Model 2 (Neural Network): 

Overall, we achieved an F1 score of about 0.58 which is slightly lower than that of the logistic regression model. This is due to the limit of memory when we add dimensions to our data. But compared to using the same feature space, we do see an increase in results compared to our logistic regression model due to the ability of a neural network to capture non-linear relationships.

We plan to employ and experiment with several SVM (Support Vector Machine) models in our exploration of crime data prediction. Firstly, SVMs are renowned for their effectiveness in handling high-dimensional data, which is particularly relevant given the extensive feature engineering we've undertaken, including location and temporal aspects. Their versatility in kernel choice allows us to experiment with various functions to model non-linear relationships that a simple logistic regression or even a neural network might struggle with. Moreover, SVM's ability to provide a robust decision boundary, thanks to its margin optimization approach, makes it ideal for complex prediction tasks like distinguishing between violent and non-violent crimes. This model is particularly resistant to overfitting, especially in high-dimensional spaces, due to its regularization mechanism. By choosing SVM as our next modeling approach, we aim to leverage its strengths in dealing with complex patterns and its comparative advantage over our previous models, thereby enriching our analysis and enhancing our predictive capabilities in the domain of crime prediction.

### Model 3 (SVM Classifier):

## Collaboration

- Detail the roles and contributions of each group member
- Mention any challenges faced and how they were overcome
- Acknowledge any assistance or resources

  # Collaboration

| NAME       | TEAM ROLE                                  | WHAT HAVE WE DONE? |
|------------|--------------------------------------------|--------------------|
| Zach Zhong | Team Leader, Project Manager, Coder        | Organized meetings, planned milestones, and assigned tasks to the team. Built the baseline model for logistic regression; Wrote and finetuned model 3; Experimented with feature engineering to improve the accuracy of our models. Brainstormed topics together. |
| Fangyu Zhu | Coder, Writer                              | Milestone 2 for Incident Location Data Exploration; Milestone 3 model 1- Logistic Regression collaborated working with Zach and Steven; Updates on writeup/readme, final review of the submissions based on milestone requirements; Final Submission Sections: Introduction; Methods/Results of Data Exploration; final submissions of group project (markdown transfer, formatting, etc.). Brainstormed topics together. |
| Jerry Gong | *Roles and contributions not listed*       | Brainstormed topics together. |
| Boyu Tian  | *Roles and contributions not listed*       | Brainstormed topics together. |
| Steven Xie | *Roles and contributions not listed*       | Brainstormed topics together. |

  
