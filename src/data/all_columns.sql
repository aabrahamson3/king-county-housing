
/*  
    AUTHOR:     Aaron Abrahamson
    Date:       Dec 2, 2019
    NOTE:       Selects relevant data from 3 King County House Sale Data
                for our analysis of house sale price data
*/

CREATE TABLE kc_data AS
(SELECT sqfttotliving AS SqFt, 
       s.saleprice AS SalePrice, 
       b.address,
       districtname, 
       b.nbrlivingunits AS NumberOfUnits,
       p.sqftlot,
       (b.sqft1stfloor+b.sqftgarageattached) AS HouseFootprint,
       ROUND((sqftgarageattached + sqft1stfloor)/sqftlot,4) AS ProportionHouse
FROM rpsale s
INNER JOIN resbldg b ON CONCAT(s.Major,s.Minor) = CONCAT(b.Major, b.Minor)
INNER JOIN parcel p ON CONCAT(s.Major,s.Minor) = CONCAT(p.Major,p.Minor)
WHERE documentdate LIKE '%2018%'
AND p.proptype = 'R');