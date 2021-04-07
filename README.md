# User Factor Adaptation for User Embedding via Multitask Learning

Data and code repository to build user embeddings for the Adapt-NLP 2021 paper [User Factor Adaptation for User Embedding via Multitask Learning](https://arxiv.org/abs/2102.11103).

![Visualization of our proposed user embeddings](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/user_viz.png)


## Review data
* Amazon (Health and Personal care): https://jmcauley.ucsd.edu/data/amazon/
* Yelp: https://www.yelp.com/dataset
* IMDb: https://www.imdb.com/interfaces/


We listed the summary of the Amazon, Yelp and IMDb review datasets. Amazon-Health refers to health- related reviews. Tokens mean the number of average tokens per document. We present the data split for the evaluation task of text classification on the right side.
![Data Statistics](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/data.png)

## Data analysis
To illustrate language variations across user groups, we present two sections of data analysis in the following. 

### Word Usage Variations

![Classification Variations Amaong User Groups](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/word_variation.png)

Word feature overlaps between every two user groups. A value of 1 means no variations of top features between two user groups, while values less than 1 indicate more feature variations. The overlap varies significantly across genre domains. This indicates that the word usage and its contexts of users change across user interests and preferences. Since the training of user embeddings relies heavily on the language features of users, this suggests that it is important to consider the language variations in user interests for the user embeddings.


### Classification Performance Variations

![Classification Variations Amaong User Groups](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/clf_variation.png)
The figure shows document classification performance when training and testing on different groups of users. The datasets come from Amazon health, IMDb and Yelp reviews. Darker red indicates better classification performance, while darker blue means worse performance. We can observe that classification performance varies across the grouped users. Higher performance variations between in- and out- user groups suggest higher user variations and vice versa. If no variations of user language exist, the performance of classifiers should be similar across the domains. The performance variations suggest that user behaviors vary across the categories of user interests. We can also observe that classification models generally perform better when tests within the same user groups while worse in the other user groups. This suggests a variability connection between the user interests and language usage, which derives user embeddings.



## How to run

Some scripts are for submitting jobs to the CLSP sever cluster, which usually have the keyword `grid` in their file names.

1. Just make sure that you have installed `conda` and `Python 3.6`;
2. Install Tensorflow, Keras and Pytorch accordingly;
3. Any data analysis scripts are in the directory of `analysis/`;
    * `uemb_analysis.py` will reduce dimensions of user embeddings and generate visualizations of users in each dataset;
    * `user_analysis.py` will calculate world and user variations and generate analysis visualizations with heatmaps.
4. Baselines are in the directory of `baselines`, including doc2user, lda2user, word2user and bert2user;
5. Main entrance will be `word_user_product.py`, which jointly train language, user and product embeddings. You can run the script by `python word_user_product.py your_task your_data_name and mode`. We use `global` as the default sampling strategy. To run all methods in a sequence, you can run `python submit_jobs.py`. The script will train our proposed user embeddings for all datasets in a HPC cluster.
6. Pretrained word/lda/doc embeddings: this will pre-train embeddings for baselines. You can run `python embeddings.py data_name model_name`.
7. To evaluate models, we have two different evaluations:
    * Intrinsic evaluation (clustering): please run the `python run_evaluator_desktop.py` and change the model and data names accordingly before running the script. If you test our model on a HPC cluster, you can run `run_evaluator_grid.py`.
    * Extrinsic evaluation (classification): please go to the directory of `[personalize](https://github.com/xiaoleihuang/UserEmbedding/tree/master/personalize)`, and you can run each classifier in the directory.


## Model Diagram

![Model Diagram](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/model.png)

The figure provides illustrations of User Embedding via multitask learning framework on the left and personalized document classifiers using trained embedding models on the right. The arrows and their colors refer to the input directions and input sources respectively. We use the logos of people, shopping cart and ABC to represent users, reviewed items and word inputs. The `\bigoplus` is the concatenation operation.


## Conference Poster
![Poster](https://github.com/xiaoleihuang/UserEmbedding/blob/master/images/poster.png)

## Contact
If you have any issues, please email xiaolei.huang@memphis.edu.


## Citation

If you feel this repository useful, please kindly cite this [paper]():

```
@inproceedings{huang2021user,
    title = "User Factor Adaptation for User Embedding via Multitask Learning",
    author = "Huang, Xiaolei  and
      Paul, Michael J.  and
      Burke, Robin and
      Dernoncourt, Franck  and
      Dredze, Mark
      ",
    booktitle = "Proceedings of the Second Workshop on Domain Adaptation for Natural Language Processing (Adapted-NLP)",
    month = april,
    year = "2021",
    address = "Kyiv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2102.11103",
    abstract = "Language varies across users and their interested fields in social media data: words authored by a user across his/her interests may have different meanings (e.g., cool) or sentiments (e.g., fast). However, most of the existing methods to train user embeddings ignore the variations across user interests, such as product and movie categories (e.g., drama vs. action). In this study, we treat the user interest as domains and empirically examine how the user language can vary across the user factor in three English social media datasets. We then propose a user embedding model to account for the language variability of user interests via a multitask learning framework. The model learns user language and its variations without human supervision. While existing work mainly evaluated the user embedding by extrinsic tasks, we propose an intrinsic evaluation via clustering and evaluate user embeddings by an extrinsic task, text classification. The experiments on the three English-language social media datasets show that our proposed approach can generally outperform baselines via adapting the user factor.",
}
```
