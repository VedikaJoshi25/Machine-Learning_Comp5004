# COMP5004 – Week 2: Titanic Data Processing

## Overview

This notebook demonstrates fundamental **data processing techniques** using the Titanic dataset.  
It covers:

- Handling missing data
- Encoding categorical variables
- Scaling numerical features
- Visualizing datasets

It also includes **coded visuals** of the Titanic and passenger distribution.

## Features

**1. Messy Dataset Simulation**  
- Random missing values in `age` and `fare`.
- Some categorical fields filled with `'Unknown'`.

**2. Missing Data Handling**  
- Fill missing values using **median** or **mean**, or drop rows.  
- Interactive choice allows experimenting with different strategies.

**3. Categorical Encoding**  
- One-Hot Encoding applied to `embark_town` column.

**4. Feature Scaling**  
- Standardization applied to `age` and `fare`.

**5. Coded Titanic Visuals**  
- Symbolic ship plot using Matplotlib patches.  
- Passenger distribution by class displayed as a bar chart.

**6. Cleaned Dataset**  
- Columns with too many missing values (`deck`, `who`) removed.  
- Dataset is ready for further machine learning tasks.

