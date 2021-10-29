# Product recommendation
This repository contains all code and file that were used to build a recommendation system
in the context of the Dataton challenge at PedidosYa.

The main idea is build a Implicit Recommendation System using the interactions between user and product,
in particular, we use the data of all order of Markets in Argentina.


# Project Structure

* [**data:**](data) It contains queries that create our data source tables.
* [**experiments:**](experiments) It has all jupyter notebooks that train our models.
* [**exploratory_data_analysis:**](exploratory_data_analysis) All the different analysis in notebooks that we made to support our hypothesis. Additional EDA was made in [Data Studio](https://datastudio.google.com/reporting/baa6cf57-a39c-4be7-8de5-f73d2cf36976/page/rMrcC/edit)
* [**metrics:**](metrics) It has a notebook that calculate hit_rate and MAP@k for a specific model in test.
* [**models:**](models) It contains our trained models. Some of them are too big for Github, so we uploaded them to Google Drive [here](https://drive.google.com/drive/folders/18Tm1gSydWOFJkPzdiI6MXMsVPCcKPEWm?usp=sharing).
* [**serving:**](serving) It contain examples of how to deploy the models using docker and tensorflow serving. 


# Model Framework
We use [Tensorflow Recommenders](https://www.tensorflow.org/recommenders/) to build a Two-tower model and we experiment
with a simple collaborative filtering using user_id and product_id and also try with a more complex model that has
the user context (days of the week and hour of the day) and product attributes (brand and categories).


<img src="https://1.bp.blogspot.com/-ww8cKT3nIb8/X2pdWAWWNmI/AAAAAAAADl8/pkeFRxizkXYbDGbOcaAnZkorjEuqtrabgCLcBGAsYHQ/s0/TF%2BRecommenders%2B06.gif" width=500>


# Team

* Juan Andreani
* Alexis De Los Santos
* Fiorella Di Rosario
* Cesar Reyes
* Patricio Woodley