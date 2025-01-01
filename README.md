# Recommender System

This was a challenge of the BOOST HACKHATON organized by INDITEX. The goal of this project is to design a recommender system for an e-commerce retailer. First, we had to analyze the data of the customer's organization in order to understand what kind of products they buy according to their preferences. For this task, there are three datasets that I have used. Their sizes are too large to upload to GitHub. So , we will provide you the URL of these objects  from our AWS bucket:

* [Users](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/users.csv)[18 MB]: Customers's information. To gather this data we had to design an script (`scripts/users_api.py`) to automate the sending of requests to the API that they provide us.

* [Products](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/products.pkl)[186 MB]: The list of products that the organization sells.

* [Train](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/train.csv)[2.8 GB]: The training data that we have used to design this system. 


