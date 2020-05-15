# Success of Email Marketing Campaigns

# Motivation

SME (Small to Medium Businesses) are often trying to make use of Email campaigns to target their prospective customers and to promote/advertise their products to these customers. While advertising campaigns can help to promote the product, it is always essential to understand and quantify the precise relationship between advertising cost and profits. Thus, the business will know how much to invest into these campaigns. 

# Data

68,000 rows >> separated into training (80%) and test (20%)  sets
11 features
  - 1 identifier column (email ID)
  - 5 categorical : Type of Email, Type of Campaign, Email Source Type, Customer Location, Time Email Sent Category
  - 5 numerical: Email Word Count, Total links, Total Images, Total Past Communications, Subject Hotness Score

Each email was originally categorized into: Ignored, Read, Acknowledged
For the purpose of analyzing the success of email advertising, we are treating read and acknowledged as one group. Because as long as the customer reads/acknowledged the email, the purpose of this advertising campaign has been served. 

# EDA

Dataset was highly imbalanced even after combinging Read & Acknowledged. 

![image](./email_status.png)
