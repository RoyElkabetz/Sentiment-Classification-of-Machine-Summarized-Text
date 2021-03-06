# Machine Summarized-Text Sentiment Classification

> This repository contains my project assignement made for the Technion's Electrical Engineering (EE) 046211 course "Deep Learning" by Prof. Daniel Soudry.
---

## Introduction
In this project I will attempt to examine the idea of classifying machine summarized text with the objective of exploring the capabilities of state-of-the-art machine summarization and its biases towards objective information in short summarizations. For that I will use Transfer Learning to fine-tune Google's T5 model into a text summarizer, which will later be used in a summarization pipeline for the IMDB reviews dataset in varying summarization lengths. Here is a flowchart of the project steps


<img src="https://github.com/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text/blob/d78a640a460f2639f5a7392512bde75b2287d5eb/Figures/Project_flowchart.png" width="700" height="">

> The project flowchart: (1) Fine-tune Google’s T5 text model using Transfer Learning. (2) Having a state-of-the-art coherent T5-summarizer. (3) Constricting a Summarization Pipeline using the T5-summarizer. (4) Transferring the IMDB dataset to the summarization pipeline for. (5) Constructing and processing the summarized IMDB reviews into a batch of different datasets. (6, 7) Training a Text Classifier on the complete IMDB reviews. (8) Comparing performance of trained Classifier on complete and summarized IMDB reviews.



## The Code
The code for this project is devided between 3 main parts:

### List of Notebooks

All the notebooks in this repository are contained in the [`notebooks`](/notebooks) folder, and are compatible and ready to run on google colab.

| #   | Subject                                         | Colab             | Nbviewer               |
|:----:|------------------------------------------------|:-----------------:|:---------------------:|
| 1   | Fine-tune T5 into a summarizer with Transfer Learning                   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text/blob/main/notebooks/Text_summarization_using_T5.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_summarization_using_T5.ipynb)|
| 2   | IMDB Dataset summarization pipeline                  | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text/blob/main/notebooks/T5_Summarizer_pipeline.ipynb)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/T5_Summarizer_pipeline.ipynb)|
| 3   | Train and test a text classifier for sentiment classification on summarized-IMDB datasets                   | [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_Classification_full_vs_machine_summarized.ipynb#scrollTo=blFvDB1GBvDP)        | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_Classification_full_vs_machine_summarized.ipynb)

These three notebooks are supported by the `utils.py` and `models.py` files in the [`src`](/src) folder.

## Datasets
### News Summary
The **News Summary** dataset can be found on Kaggle in the next link [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/sunnysai12345/news-summary)

The T5 model can take inputs of up to 512 tokens, so any sequence with more tokens was truncated in the training process. Here is a token count histograms of the News Summary train dataset. 



<img src="https://github.com/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text/blob/bf1a5393147178c51102edb05fa93c33c172d82d/Figures/news_summary_histogram.png" width="700" height="">

### Summarized-IMDB
The summarized-IMDB datasets I generated with T5 can be downloaded by runing the following lines of code:



#### List of IMDB datasets

| #   | Dataset                                         | Link             | 
|:----:|------------------------------------------------|:-----------------:|
| 1   | Train                   | `!gdown --id '1EJD8f_PiymNmhaDuvxj27qTo2OMMruiP' -O 'train.csv'`       
| 2   | Validation                   | `!gdown --id '1--t5cZIL81qBOLjbHxj6iDeXUez5zAvV' -O 'valid.csv'`        
| 3   | Test (with sampling)                   | `!gdown --id '1-2-nT7vtLNMoiUMXuLcWp2wdCryIS9Ep' -O 'test_with_sampling.csv'`|
| 4   | Test (without sampling)                   | `!gdown --id '1-3SD5xYj_R8VaxT15NWs_HbnR5tz7fmP' -O 'test_without_sampling.csv'`|

The Train dataset is identical to the original IMDB Train from Pytorch. The Validation and Test were generated ni the following way:
-  The original IMDB Test dataset was splited to (1000, 24000) samples.
-  The 1000 samples were taken for test (They were the first 1000 in the original IMDB Test) and 24000 for validation.
-  The 1000 Test samples were shuffled and summarized with two different methods: with and without sampling (see [`generate()`](https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate)), and with 11 different summarization lengths of [120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20] tokens.



Here are histograms of the token counts for each dataset. The title of each plot for the summarized-IMDB word count histograms describes the maximal length parameter value set to the T5 summarizer. Notice that the maximal length parameter of 100 causes a generated average word count of 70 words per summary. In general, the mean word count for each dataset is typically a few dozens of words less than the maximal length value that was set.

![image](Figures/IMDB_histogram.png)

## Model

The text classifier architecture that was used for the sentiment classification is illustarted in the figure below. The model is comprised of a single embedding-bag layer followed by a linear fully-connected layer followed by a fully-connected layer with a single unit (binary layer). The $W_k$ variables stand for single words / tokens where ik are their index representation in the model’s vocabulary.

![image](Figures/classifier_model.png)

## Results


The classifier was trained on full-text IMDB reviews and then tested on full-texts and their summarizations in different lengths. Here are the Loss and Accuracy measured through the training process of the text classifier model 

![image](Figures/loss_acc.png)


Each point in the classifier's test accuracy i the graph below describes the accuracy of predicting the correct classes for the **same** 1000 test samples sharing the same maximal summary length. The orange data belong to the test sets generated without "sampling" and the blue data belong the ones generated with sampling. An explanation of the summarization process can be found in the project paper [Sentiment Classification of Machine Summarized-Text](/paper/Sentiment_Classification_of_Machine_Summarized_Text.pdf).

![image](Figures/test_acc.png)

## Conclusions
In this project, I attempted to find the relation between the length of a summarized text piece generated with a state-of-the-art machine summarizer to the amount of subjective information preserved in the machine summarization process. To quantify the amount of subjective information in a text piece I used a sentiment text classifier, and through its performance in classifying correctly the given text, I infer indirectly the quality of the summarization process. From the results I got, it seems that there is a tight connection between the length of the summary to the amount of subjective information kept in the summarized text through the summarization process. We see that shorter summary pieces contain less subjective information which makes them harder to classify in a positive/negative framework. The results are not unambiguous, that is because of a bias towards summaries of lengths longer than 60 words which have been used in the transfer learning process of the T5 fine-tune. To verify if this bias is the main cause of the results we got, we need to use a dataset of shorter summaries in the transfer learning process of the T5 model.

---
## References
- [[1]](https://arxiv.org/abs/1910.10683) Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.

## Contact


Roy Elkabetz - [elkabetzroy@gmail.com](mailto:elkabetzroy@gmail.com)

Project link - [https://github.com/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text](https://github.com/RoyElkabetz/Sentiment-Classification-of-Machine-Summarized-Text)

Project paper - [Sentiment Classification of Machine Summarized Text](/paper/Sentiment_Classification_of_Machine_Summarized_Text.pdf)
 


