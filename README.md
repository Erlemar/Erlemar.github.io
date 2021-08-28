Now I have a personal [website](https://andlukyane.com/)!


# Data science portfolio by Andrey Lukyanenko

This portfolio is a compilation of notebooks which I created for data analysis or for exploration of machine learning algorithms. A separate category is for separate projects.

## Stand-alone projects.

### Handwritten digit recognition

This is my own project using image recognition methods in practice. This is a site (also works on mobile) where user can draw a digit, and machine learning models (FNN and CNN) will try to recognize it. After than models can use the drawn digit for training to improve their accuracy. Live version is [here](https://digits-draw-recognize.herokuapp.com/). The code can be found [here](https://github.com/Erlemar/digit-draw-recognize).

### Chatbot in telegram

A conversational chatbot in [telegram](http://t.me/amlnlpbot) which was created for an honor assignment of [nlp course by Higher School of Economics](https://www.coursera.org/learn/language-processing/home/welcome). The main functionality of the bot is to distinguish two types of questions (questions related to programming and others) and then either give an answer or talk using a conversational model.

## Kaggle competitions.

### Avito demand prediction

[Avito demand prediction](https://www.kaggle.com/c/avito-demand-prediction) was a competition on Kaggle where we tried to predict something like demand based on ads content. This competition was very interesting because it had tabular data, texts and images. On the other hand this was the reason the competition was quite difficult. My team reached 131th place and got bronze medal! Here is a [link](https://github.com/Erlemar/Avito_demand_prediction_2018) to my solution.

### Categorization of purchases

[This](https://www.kaggle.com/c/receipt-categorisation) was a Russian inclass Kaggle competition in the third session of ODS ml course. It sounded interesting and I took part in it reaching a 3rd place. [Here](https://www.kaggle.com/artgor/3rd-place-solution) is my kaggle kernel with a solution.

## Kaggle kernels.

### 2018 Kaggle ML & DS Survey Challenge

Some time ago Kaggle launched a big online survey for kagglers and now this data is public. There were multiple choice questions and some forms for open answers. Survey received 23k+ respondents from 147 countries. As a result we have a big dataset with rich information on data scientists using Kaggle.
In this [kernel](https://www.kaggle.com/artgor/russia-usa-india-and-other-countries) I compare DS in USA, Russia, India and other countries.

### DonorsChoose.org Application Screening

DonorsChoose.org empowers public school teachers from across the country to request much-needed materials and experiences for their students. DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. This is a competition on Kaggle where people can create a machine learning model to help this fund with auto-approving of applications. Prizes are given to the authors with the most upvoted kernels. Here is my [kernel](https://www.kaggle.com/artgor/eda-feature-engineering-and-xgb-lgb) with extensive EDA, feature engineering and building models.
This kernel got 2nd place by the number of votes and I won Google Pixelbook for it!

### Avito Demand Prediction Challenge

Avito challenge is about predicting demand for an online advertisement based on its full description (title, description, images, etc.), its context (geographically where it was posted, similar ads already posted) and historical demand for similar ads in similar contexts. The competition is interesting due to many types of data in it which allows to build various models. Here is my [kernel](https://www.kaggle.com/artgor/eda-features-engineering-and-lightgbm) with EDA, creating features and building models.

### Home Credit Default Risk

Home Credit Bank offers a challenge of credit scoring. There is a lot of data about applicants and their previous behavior. Here is my [kernel](https://www.kaggle.com/artgor/eda-in-progress).

### Movie Review Sentiment Analysis

Some time ago Kaggle has launched several "remakes" of old competitions. It means that datasets are the same, but now we are offered an opportunity to simply explore the data and create kernels with new methods. One of these competitions is sentiment analysis of Rotten Tomatoes dataset with 5 classes (negative, somewhat negative, neutral, somewhat positive, positive). I have created a [kernel](https://www.kaggle.com/artgor/movie-review-sentiment-analysis-eda-and-models/notebook) with EDA and modern NN architecture: LSTM-CNN. Currently this kernel shows the 5th result of leaderboard.

### Two Sigma: Using News to Predict Stock Movements

In this competition Reuters provide unique data, which can't be obtained outside of this competition. We can see a 10 years worth of news and market data on many companies. This competition is kernel-only, which means that everyone has the same amount of computational power for this competition. In my [kernel](https://www.kaggle.com/artgor/eda-feature-engineering-and-everything) I have analysed the data and showed trends of market data.

### Santander Value Prediction Challenge

In this competition we got an anonymized dataset, later it was found that it had a certain structure. In my [kernel](https://www.kaggle.com/artgor/satander-eda-nn-features-lgb) I tried to analyze the data and created new features using NN model.

### Google Analytics Customer Revenue Prediction

RStudio hosted this competition to prove that machine learning algorithms can impact business and help marketing. In my [kernel](https://www.kaggle.com/artgor/eda-on-basic-data-and-lgb-in-progress) I did an extensive EDA and build an interesting LGB model.

### Data Science for Good: Center for Policing Equity

This dataset was provided by The Center for Policing Equity. They hope that kagglers will help to create better models, find some unique insights and improve geo-analytics. In my [kernel](https://www.kaggle.com/artgor/what-can-be-said-about-justice) I try to do such things.

## Classification problems.

### Titanic: Machine Learning from Disaster

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Titanic.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Titanic.ipynb)

Titanic: Machine Learning from Disaster is a knowledge competition on Kaggle. Many people started practicing in machine learning with this competition, so did I. This is a binary classification problem: based on information about Titanic passengers we predict whether they survived or not. General description and data are available on [Kaggle](https://www.kaggle.com/c/titanic).
Titanic dataset provides interesting opportunities for feature engineering.

### Ghouls, Goblins, and Ghosts... Boo!

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/GGG.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/GGG.ipynb)

Ghouls, Goblins, and Ghosts... Boo! is a knowledge competition on Kaggle. This is a multiple classification problem: based on information about monsters we predict their types. A fun competition for Halloween. General description and data are available on [Kaggle](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo).
This dataset has little number of samples, so careful feature selection and model ensemble are necessary for high accuracy.

### Otto Group Product Classification Challenge

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Otto_Group.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Otto_Group.ipynb)

Otto Group Product Classification Challenge is a knowledge competition on Kaggle. This is a multiple classification problem. Based on information about products we predict their category. General description and data are available on [Kaggle](https://www.kaggle.com/c/otto-group-product-classification-challenge).
The data is obfuscated, so the main questionlies in the selection of the model for prediction.

### Imbalanced classes

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Imbalanced.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Imbalanced.ipynb)

In real world it is common to meet data in which some classes are more common and others are rarer. In case of a serious disbalance prediction rare classes could be difficult using standard classification methods. In this notebook I analyse such a situation. I can't share the data, used in this analysis.

### Bank card activations

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Card_activation.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Card_activation.ipynb)

Banks strive to increase the efficiency of their contacts with customers. One of the areas which require this is offering new products to existing clients (cross-selling). Instead of offering new products to all clients, it is a good idea to predict the probability of a positive response. Then the offers could be sent to those clients, for whom the probability of response is higher than some threshold value.
In this notebook I try to solve this problem.

## Regression problems.

### House Prices: Advanced Regression Techniques

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/House_Prices.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/House_Prices.ipynb)

House Prices: Advanced Regression Techniques is a knowledge competition on Kaggle. This is a regression problem: based on information about houses we predict their prices. General description and data are available on [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
The dataset has a lot of features and many missing values. This gives interesting possibilities for feature transformation and data visualization.

### Loan Prediction

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Loan_Prediction.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Loan_Prediction.ipynb)

Loan Prediction is a knowledge and learning hackathon on Analyticsvidhya. Dream Housing Finance company deals in home loans. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. Based on customer's information we predict whether they should receive a loan or not. General description and data are available on [Analyticsvidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/).


### Caterpillar Tube Pricing

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Caterpillar.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Caterpillar.ipynb)

Caterpillar Tube Pricing is a competition on Kaggle. This is a regression problem: based on information about tube assemblies we predict their prices. General description and data are available on [Kaggle](https://www.kaggle.com/c/caterpillar-tube-pricing).
Dataset consists of many files, so there is an additional challenge in combining the data snd selecting the features.

## Natural language processing.

### Bag of Words Meets Bags of Popcorn

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Bag_of_Words.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Bag_of_Words.ipynb)

Bag of Words Meets Bags of Popcorn is a sentimental analysis problem. Based on texts of reviews we predict whether they are positive or negative. General description and data are available on [Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial).
The data provided consists of raw reviews and class (1 or 2), so the main part is cleaning the texts.

### NLP with Python: exploring Fate/Zero

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Fate_Zero_explore.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Fate_Zero_explore.ipynb)

Natural language processing in machine learning helps to accomplish a variety of tasks, one of which is extracting information from texts. This notebook is an overview of several text exploration methods using English translation of Japanese light novel "Fate/Zero" as an example.

### NLP. Text generation with Markov chains

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Markov_chain_nlp.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Markov_chain_nlp.ipynb)

This notebook shows how a new text can be generated based on a given corpus using an idea of Markov chains. I start with simple first-order chains and with each step improve model to generate better text.

### NLP. Text summarization

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Summarize.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Summarize.ipynb)

This notebook shows how text can be summarized choosing several most important sentences from the text. I explore various methods of doing this based on a news article.

## Clustering

### Clustering with KMeans

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Clustering_with_K-Means.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Clustering_with_K-Means.ipynb)

Clustering is an approach to unsupervised machine learning. Clustering with KMeans is one of algorithms of clustering. in this notebook I'll demonstrate how it works. Data used is about various types of seeds and their parameters. It is available [here](https://archive.ics.uci.edu/ml/datasets/seeds).

## Neural networks

### Feedforward neural network with regularization

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/NN_GGG.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/NN_GGG.ipynb)

This is a simple example of feedforward neural network with regularization. It is based on Andrew Ng's lectures on Coursera. I used data from Kaggle's challenge "Ghouls, Goblins, and Ghosts... Boo!", it is available [here](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo).

## Data exploration and analysis

### Telematic data

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Devices_analysis.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Devices_analysis.ipynb)

I have a dataset with telematic information about 10 cars driving during one day. I visualise data, search for insights and analyse the behavior of each driver. I can't share the data, but here is the notebook. I want to notice that folium map can't be rendered by native github, but nbviewer.jupyter can do it.

## Recommendation systems.

### Collaborative filtering

[Github](https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/Collaborative_filtering.ipynb) [nbviewer](http://nbviewer.jupyter.org/github/Erlemar/Erlemar.github.io/blob/master/Notebooks/Collaborative_filtering.ipynb)

Recommenders are systems, which predict ratings of users for items. There are several approaches to build such systems and one of them is Collaborative Filtering. 
This notebook shows several examples of collaborative filtering algorithms.
