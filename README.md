# king-county-housing

Aaron, Brent & Lucas
December 6th 2019 

## Background:
The factors and variables that affect and influence housing prices are of particular interest to many different groups of people. Often times, most of us non-housing experts have to rely on the opinions of others to determine what the true value of a house is on the open market. Sometimes, however, these evaluations are inaccurate due to new, emerging market trends and/or over generalizations and antiquated assumptions about first-time home buyers that are no longer relevant.

## Objective:
For this project, we decided to look at three main assumptions routinely expressed by real-estate experts that may influence the value of a given house. Using data provided by King County Department of Assessments (link https://info.kingcounty.gov/assessor/DataDownload/default.aspx) for the 2018 year, we decided to test the validity of these three claims:
Does higher square footage increase home sale price?
Does the house filling a higher percentage of the lot decrease sale price?
Is the cost per square foot of a duplex lower than for a single family home?

## Method: 
We began by for our three null hypotheses that we wished to reject:
Square footage has no effect on the housing prices.
1. Having a house that fills a great percentage of the square lot does not affect the houses value.
There is no significant difference between the cost per square foot of a duplex and that of a single family home.
2. After this, we visited the aforementioned website and downloaded the ZIP files: ‘Parcel,’ ‘Real Poperty Sales,’ and ‘Residential Building.’
3. We loaded these three files into a PostgreSQL database and merged all three tables by forming a unique ID from the concatenation of the ‘Major’ and ‘Minor’ columns present in all three tables.
4. We then created a unique filter statement that removed data that wasn’t from 2018. We also selected 40 or so ‘Independent’ variables that we believed may influence house price in some capacity or another. We also specifically targeted features that would allow us to reject/fail to reject our null hypothesis. These features were: lot size, attached garage size, total living space, the area of the first floor, whether or not the unit was a duplex and the 2018 sale price of the unit. There was no specific value for house to lot ratio so we created one by summing the square footage of the first floor and attached garage and then divided by the lot size to procure the ratio.
5. We first queried these values into a Python Pandas DataFrame. Within this DataFrame, we filtered out houses that were above two standard deviations of the mean house price and were below $100,000 dollars.  From this, we ran a basic linear regression exclusively on the targeted variables. When doing this, we got an R2 of .31 indicating an extremely week correlation:
6. We decided to further investigate by creating a more complex model that looked included factors such as location, number of rooms, number of bathrooms, whether or not the property is situated next to a water front and much more. We looked at what values most highly correlated with sale price while also correlating the least with the other ‘independent variables.’ We used several methods such as recursive functions and correlation matrices to determine which values to include in our new model
7. To include non-continuous variables in our data - such as zip code - we one hot encoded before using them in our linear regression model.
8. We then calculated the R2 of this new model and determined it to be more accurate than the previous model - see Analyzation section for further discussion. We used this to determine the coefficients for each independent variable and the constant value as well.
9. After being satisfied with he R2 of the model, we performed several sanity checks to see how or model compared to the values listed on Zillow. 

## Analyzation/Conclusion:
### Comparison of the two models: 
The second mode was a substantially better predictor of housing price. It had an R2 value of .71 adjusted compared to the first model of .31. Both models rejected were able to reject the first and second null hypothesis - that total square footage and ratio had no effect on sale price - and reaffirmed the initial claims that there is a positive correlation between square footage and home sale price and a negative correlation between house footprint and lot ratio. The first model was able to reject the third claim. However, it supported the opposite of the original claim and suggested that the price per square foot of a duplex was actually greater than for a single family home. The more advanced model failed to reject the null hypothesis of third claim however. Because the second model had a significantly higher R2 value, we concluded that we were unable to reject the null hypothesis for the third claim. 

### Variables Used:
We found that the houses zip code correlated the best with the housing prices. This makes intuitive sense as zip codes are a good measure of a given neighborhood which tend to be very homogenous. This helps fix the discrepancy in price that would exist from two similar houses located in different socio economic regions. After this, the next best correlator was the building grade. However, there was a huge correlation between this and the zip code which made it not ideal when including it in our new model. We chose to omit it for this reason. After this, the variable that correlated best with housing prices and was sufficiently orthogonal to the zip code variable was the the total living space. We also concluded that the view played a significant role as well in our model. In addition to this, we included the values that would allow us to address the initial claims: the ratio of the footprint to the loft, and whether or not it was a duplex. 

### Checking our model:
Using our model, we did a quick search of houses on sale in Zillow, the vast majority of the houses we checked fell within a standard deviation of the estimate our model produced. The ones that didn’t fall into one standard deviation had several factors in common. The first was that the price was over 1.5 million dollars. This makes intuitive sense from our filter of houses below 100,000 dollars and above 2 standard deviations of the mean price ~ 2.5million dollars.  This lower filter was applied to address in-family sales that were way below market value and to remove items such as mobile homes. The filter of 2.5 million had to do more with the demographic we are addressing. Houses above 2.5million are not indicative of current market trends as the factors that appeal to the top 1 percent are not the same as the factors that first time homeowners are concerned with.

### Future Research:
Due to the success of predicting the sale price base purely on the zip code. We would hope to further improve this model by reanalyzing theses factors normalized to specific zip codes. For instances, while lot size wasn’t a good indicator of house price on its own - there are many small lots in West Seattle that are worth more than giant lots in Monroe - grouping theses variables by zip codes may make them better predictors of overall house prices.



### Additional info and links
1. Download the King County Data here:
Links for data:
https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip
https://aqua.kingcounty.gov/extranet/assessor/Parcel.zip
https://aqua.kingcounty.gov/extranet/assessor/Lookup.zip

links for data documents:
https://info.kingcounty.gov/assessor/datadownload/desc/Real%20Property%20Sales.doc
https://info.kingcounty.gov/assessor/datadownload/desc/Residential%20Building.doc
https://www.kingcounty.gov/assessor~/media/depts/assessor/documents/WebAppImages/Parcel_v2.ashx
https://info.kingcounty.gov/assessor/datadownload/desc/Lookup.doc

2. Unzip the .zip files and place the .csv files into ../data/raw/ directory within the repo

3. run the following terminal command from your maind repo directory:
sh setup.sh 


