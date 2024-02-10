# Data Preprocessing for "San Francisco Incident Reports" Analysis

This README highlights the steps for preprocessing the "San Francisco Incident Reports (2018-present)" dataset for classifying incidents into "violent" and "non-violent" categories. The document will guide through data cleaning and transformation to prepare the dataset for analysis using feedforward neural networks.

## Data Cleaning

### 1. Handle Missing Values
- **Numerical columns**: Identify and fill missing values in columns like `longitude` and `latitude`. we will use the mean or median of the column for filling in the missing values.
- **Categorical columns**: For columns with categorical data such as `Intersection`, we will fill missing entries with the most frequent value or a placeholder such as 'Unknown'.

### 2. Remove Outliers
- Use statistical methods to identify and remove outliers from numerical columns like `longitude` and `latitude`. We will use the Interquartile Range (IQR) method for this purpose.

## Data Transformation
### 1. Normalization
- We will normalize numerical features to ensure they're on the same scale. This is crucial for features like `longitude` and `latitude`. We will use the z-score normalization to achieve this.

### 2. Extract Target Variable
- We will manually classify incidents into "violent" and "non-violent" categories using the `Incident Category` column. For example, we may classify "Larceny Theft" as a "non-violent" act and "Assault" as a "violent" act in the column.

### 3. Categorical Feature Encoding
- We can use one-hot encoding technique for converting categorical data into a binary vector, such as converting "non-violent" to `0` and "violent" to `1`.

## Splitting
### 1. Split the data
- After completing the data cleaning and transformation steps, we will split the dataset into training and testing sets to be used by our models.






