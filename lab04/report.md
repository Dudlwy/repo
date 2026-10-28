1. Goal
The primary objective of this lab is to build a reproducible, leakage-free data pipeline for mixed numerical and categorical data using the UCI Adult dataset, perform strict schema validation checks, engineer derived features without contaminating the test set, and produce a processed dataset hash, configuration, and complete data card
2. Method
We took thousands of census income records containing worker information, checked all numbers and categories to find missing question marks and invalid values, and split the data into separate training and testing groups before doing any cleaning so the model could not peek at the test answers[cite: 2, 4]. We cleaned missing values using averages and most common entries, clipped extreme values, created two new helpful math features combining capital gains and working hours relative to age, checked for duplicate rows with conflicting income answers, saved our exact processor settings, and generated a unique security hash code to verify that our data pipeline can be run with a single command
3. Result

![alt text](image.png)
![alt text](image-1.png)

4. Conclussion