# North Chicago Township Home Sale Analysis

## I. Executive Summary
This project focuses on investigating the determinants of single-family home sale prices in North Chicago, particularly within affluent neighborhoods like Gold Coast, Magnificent Miles, and Lincoln Park, challenging the traditional emphasis on location by employing a linear regression model on various continuous predictors. The study aims to provide in-depth insights beyond the common mantra of "location, location, location," benefiting prospective homebuyers, sellers, and real estate professionals in understanding the nuanced dynamics affecting property prices in the North Chicago Township.

## II. Data Description
<img width="846" alt="image" src="https://github.com/sallylee0801/Northern-Chicago-Township-Home-Sale-Analysis/assets/121594845/3c517b28-59ef-402c-986a-0355cafcf2dc">

This data consists of 403 single-family homes sold in the North Chicago township of Cook County between 2018 and 2020.

## III. Exploratively Data Analysis
- Initial analyses revealed a right-skewed distribution of Sale Prices, prompting a transformation to Log Sale Price for alignment with regression model assumptions.

<img width="468" alt="image" src="https://github.com/sallylee0801/Northern-Chicago-Township-Home-Sale-Analysis/assets/121594845/9035892e-f10d-4845-8b1f-fec86358d8bd">

- The ensuing multiple linear regression model, incorporating eight predictors, demonstrated robust performance with a high Coefficient of Determination (R-squared) of 0.8384.

<img width="468" alt="image" src="https://github.com/sallylee0801/Northern-Chicago-Township-Home-Sale-Analysis/assets/121594845/0287cb50-6d88-4dbb-b109-bb70296780a9">

- Building Square Feet emerged as the most influential predictor, followed by Land Acre and Full Baths, as confirmed by both regression coefficients and Shapley values.

<img width="282" alt="image" src="https://github.com/sallylee0801/Northern-Chicago-Township-Home-Sale-Analysis/assets/121594845/e79482c1-9725-42d5-ad03-af5b2e29ec4b">

- Utilizing median values, we predicted a Sale Price of $886.03, supported by a confidence interval providing a nuanced range for anticipated property values.

These findings collectively empower stakeholders with actionable insights, whether optimizing pricing strategies, understanding predictor impacts, or navigating the intricacies of the North Chicago Township real estate market.
