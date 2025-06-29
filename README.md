# Recommender System

<p align = "center">
<img src = "img/logo.png" width = 200>
</p>

<p align = "center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Bytesized&size=25&duration=6000&pause=1000&color=497DF7&width=435&lines=Customizing+exclusive+selections..." alt="Typing SVG" /></a>
</p>


This was a challenge of the BOOST HACKHATON organized by INDITEX. The goal of this project is to design a recommender system for an e-commerce retailer. First, we had to analyze the data of the customer's organization in order to understand what kind of products they buy according to their preferences. For this task, there are three datasets that I have used. Their sizes are too large to upload to GitHub. So , we will provide you the URL of these objects from our AWS bucket:

* [Users](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/users.csv) [17 MB]: Customers's information. To gather this data we had to design an script (`scripts/users_api.py`) to automate the sending of requests to the API that they provide us.
    * K-means (Uploaded to the repository): The file containing the results when the K-means algorithm in order to reduce the computing time required to reproduce the experiment.

* [Products](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/products.pkl) [186 MB]: The list of products that the organization sells.
    * [Embeddings](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/embeddings.csv) [582 MB]: The embeddings of the product's flat image obtained from computer vision techniques.

* [Train](https://cfolstorage.s3.eu-west-3.amazonaws.com/data_recommender_system/train.parquet) [605 MB]: The training data that we have used to design this system.



## Development of the model

Dataset and Methodology
The dataset we have does not provide explicit user ratings for the products offered by this virtual store; therefore, what is used to measure user preferences is the implicit feedback they generate from variables such as:

* How many times has the user viewed the product? (Mild interest)
* How many times did they add the product to the cart? (Strong interest)
* How many times did they purchase the product? (Maximum interest)

Then, to recommend a product to a user, their purchase history is analyzed, as well as that of other customers with similar purchasing behaviors. This information reveals latent patterns that allow representing user preferences and product/item characteristics (Hu, Y. et al., 2008, p. 1).

For the development of this recommendation system, the ALS (Alternating Least Squares) algorithm was chosen. This consists of creating an interaction matrix between products and customers. However, not all users interact with all products, so what we have is a sparse matrix. Therefore, ALS performs a decomposition of this matrix into two dense matrices whose initial values are generated pseudo-randomly, and due to this nature, a seed was used in training to ensure reproducibility.

The main variables defined in the training stage according to Hu, Y. et al. (2008):

* Number of Factors ($f$): This determines the dimensionality of the latent space. Think of it as the number of hidden "concepts" or "topics" that you want to capture about user preferences and item characteristics.
    * Sweet spot: 20-200 for large volumes of data

* Regularization Parameter ($\lambda$): Controls how much we penalize large factor values to prevent overfitting.

    * Sweet spot: 0.1-1.0

* Confidence Multiplier ($\alpha$): Determines how much more we trust frequent interactions compared to single interactions. We won't use it, because the we have defined the way to calculate the confidence values.

    * Typical range: 1-40

* Number of Iterations: The balance between convergence and overfitting.

    * Typical range: 15-30

La explicación a detalle sobre el algoritmo el ALS se puede encontrar el siguiente notebook: `notebooks/recomm_system.ipynb`.

## References:

* Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE International Conference on Data Mining (pp. 263-272).
