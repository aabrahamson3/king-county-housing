/*
    Author:     Aaron C. Abrahamson
    Purpose:    Import CSV file into a PSQL table
    Date:       December 2, 2019
*/

-- Drop any existing tables
DROP TABLE IF EXISTS RPSale;
DROP TABLE IF EXISTS ResBldg;
DROP TABLE IF EXISTS Parcel;

-- Create a table from King Country Real Property Sale 
CREATE TABLE RPSale(
"ExciseTaxNbr" NUMERIC(7),
"Major" CHAR(6),
"Minor" CHAR(4),
"DocumentDate" DATE,
"SalePrice" NUMERIC(9),
"RecordingNbr" CHAR(14),
"Volume" CHAR(3),
"Page" CHAR(3),
"PlatNbr" CHAR(6),
"PlatType" CHAR(1),
"PlatLot" CHAR(14),
"PlatBlock" CHAR(7),
"SellerName" CHAR(300),
"BuyerName" CHAR(300),
"PropertyType" NUMERIC,
"PrincipalUse" NUMERIC(2),
"SaleInstrument" NUMERIC(2),
"AFForestLand" CHAR(1),
"AFCurrentUseLand" CHAR(1),
"AFNonProfitUse" CHAR(1),
"AFHistoricProperty" CHAR(1),
"SaleReason" NUMERIC(2),
"PropertyClass" NUMERIC(2),
"SaleWarning" CHAR(25)

);


-- Copy the csv contents of KC Real Property Sale into table
COPY RPSale
FROM '/Users/tree/ds/proj2/EXTR_RPSale.csv'
DELIMITER ',' CSV HEADER; 