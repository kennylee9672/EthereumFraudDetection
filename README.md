# Ethereum Fraud Detection

## Milestones

### Part 1. Data Prep (Saksham)
  - Preparation:
    - Focus on current database for now
  - Feature Evaluation:
    - PCA → find the good features (Kenny)
    - Heatmap (Akshay)
  - Balanced Data / Training Data prep:
    - 3-Way cross validation
    - SMOTE (Pratik)

### Part 2. Model Choice
  - Supervised:
    - RF, DT (Akshay)
  - Unsupervised: 
    - GMM, DBSCAN, Multiple Component Multivariate GMM (Pratik)

### Part 3. Model Evaluation (Kenny)
  - ROC, PRC (F1)


- convert string to numeric (S)
- normalized the data ()
- features explaination

### Notes:
 - Etherscan column verification: confirm data with etherscan (Currently all values are 0) 
    - ERC20 min val sent contract
    - ERC20 max val sent contract
    - ERC20 avg val sent contract
    - ERC20 avg time between sent tnx
    - ERC20 avg time between rec tnx
    - ERC20 avg time between rec 2 tnx
    - ERC20 avg time between contract tnx
