# Machine Summarized Text Sentiment Classification

> This repository contains my project assignement made for the Technion's Electrical Engineering (EE) 046211 course "Deep Learning" by Prof. Daniel Soudry.
---

## Introduction
In this project I will attempt to examine the idea of classifying machine summarized text with the objective of exploring the capabilities of state-of-the-art machine summarization and its biases towards objective information in short summarizations. For that I will use Transfer Learning to fine-tune Google's T5 model into a text summarizer, which will later be used in a summarization pipeline for the IMDB reviews dataset in varying summarization lengths.

## The Code
> The code for this project split in to 3 main parts:

__1. Fine-tune T5 into a summarizer with Transfer Learning:__
* [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_Classification_full_vs_machine_summarized.ipynb#scrollTo=blFvDB1GBvDP)
* [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_summarization_using_T5.ipynb)

__2. IMDB Dataset summarization pipeline:__
* [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lin-Sinorodin/Technion_046211-Deep_Learning_Project/blob/master/Project/train_classifier.ipynb)
* [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/T5_Summarizer_pipeline.ipynb)

__3. Train and test a text classifier for sentiment classification on summarized-IMDB datasets:__
* [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lin-Sinorodin/Technion_046211-Deep_Learning_Project/blob/master/Project/train_classifier.ipynb)
* [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/RoyElkabetz/Text-Summarization-with-Deep-Learning/blob/main/notebooks/Text_Classification_full_vs_machine_summarized.ipynb)

## Datasets
The **News Summary** dataset can be found on Kaggle [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/sunnysai12345/news-summary)

The T5 model can take inputs of up to 512 tokens, so any sequence with more tokens was truncated in the training process. Here is a token count histograms of the News Summary train dataset. 
![image](Figures/news_summary_histogram.png)

The summarized-IMDB datasets I generated with T5 can be downloaded by runing the following lines of code:


**Train:** 

`!gdown --id '1EJD8f_PiymNmhaDuvxj27qTo2OMMruiP' -O 'train.csv'`

**Validation:**

`!gdown --id '1--t5cZIL81qBOLjbHxj6iDeXUez5zAvV' -O 'valid.csv'`

**Test (with sampling):** 

`!gdown --id '1-2-nT7vtLNMoiUMXuLcWp2wdCryIS9Ep' -O 'test_with_sampling.csv'`

**Test (without sampling):** 

`!gdown --id '1-3SD5xYj_R8VaxT15NWs_HbnR5tz7fmP' -O 'test_without_sampling.csv'`

The title of each plot for the summarized-IMDB word count histograms describes the maximal length parameter value set to the T5 summarizer. Notice that the maximal length parameter of 100 causes a generated average word count of 70 words per summary. In general, the mean word count for each dataset is typically a few dozens of words less than the maximal length value that was set.

![image](Figures/IMDB_histogram.png)

## Model

The text classifier architecture that was used for the sentiment classification is illustarted in the figure below. The model is comprised of a single embedding-bag layer followed by a linear fully-connected layer followed by a fully-connected layer with a single unit (binary layer). The $W_k$ variables stand for single words / tokens where ik are their index representation in the model’s vocabulary.

![image](Figures/classifier_model.png)

## Results


Loss and Accuracy measured through the training process of the text classifier model.

![image](Figures/loss_acc.png)


The classifier accuracy over the test sets. Each point in the graph describes the classifier’s accuracy of predicting the correct classes for the same 1000 test samples sharing the same maximal summary length. The orange data belong to the test sets generated without "sampling" and the blue data belong the ones generated with sampling.

![image](Figures/test_acc.png)

## Conclusion
In this project, I attempted to find the relation between the length of a summarized text piece generated with a state-of-the-art machine summarizer to the amount of subjective information preserved in the machine summarization process. To quantify the amount of subjective information in a text piece I used a sentiment text classifier, and through its performance in classifying correctly the given text, I infer indirectly the quality of the summarization process. From the results I got, it seems that there is a tight connection between the length of the summary to the amount of subjective information kept in the summarized text through the summarization process. We see that shorter summary pieces contain less subjective information which makes them harder to classify in a positive/negative framework. The results are not unambiguous, that is because of a bias towards summaries of lengths longer than 60 words which have been used in the transfer learning process of the T5 fine-tune. To verify if this bias is the main cause of the results we got, we need to use a dataset of shorter summaries in the transfer learning process of the T5 model.

---
## References
- [[1]](https://arxiv.org/abs/2010.11929) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
- [[2]](https://arxiv.org/abs/1409.1556) Very Deep Convolutional Networks for Large-Scale Image Recognition.
 

## TODO

## Report:
- [ ] write down a project report (up to 8 pages):
- [ ] 1. Introduction 
- [ ] 2. Methods
- [ ] 3. Experiments
- [ ] 4. Conclusions
- [ ] 5. Future work
- [ ] 6. References


============================================================================================================





<!--  ## The Code
> The code for this project split in to 2 parts:

__1. Data exploration and feature extraction:__
* Notebook can an be found on Kaggle (the dataset is large ~ 28GB, so it's more convenient to access the data from Kaggle directly):
  * [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/linsinorodin/feature-extraction-vit-vgg)
* Used to export the features files used in the next part:
    * ```train_predictions_vgg.csv```
    * ```train_true_values_vgg.csv```
    * ```train_predictions_vit.csv```
    * ```train_true_values_vit.csv```
    * ```val_predictions_vgg.csv```
    * ```val_true_values_vgg.csv```
    * ```val_predictions_vit.csv```
    * ```val_true_values_vit.csv```
__2. Train classifier models from the generated features:__
* Notebook can be found in this repository at ```Project/train_classifier.ipynb```
* Also, after obtaining the features files (previous part), the code can be used in Google Colab: 
  * [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lin-Sinorodin/Technion_046211-Deep_Learning_Project/blob/master/Project/train_classifier.ipynb)

## Dataset
The dataset [COVIDx CT](https://www.kaggle.com/hgunraj/covidxct) from Kaggle used for this project. 
This dataset contains CT scans for both positive and negative COVID19 cases.
 
> __Example images from the dataset:__
> * The original dataset contains Pneumonia class, which wasn't considered here for simplicity.
> * The number of data samples for positive and negative class isn't balanced, so the custom dataloader 
keeps equal number of samples of both positive and negative samples. 

![image info](Project/Figures/preview_data.png)

> __Preprocessing steps:__
> * All the images provided with a bounding box, allowing one to crop the images to contain only the relevant areas.
> * In order to use the ViT model, the images should be with size 384x384.

![image](Project/Figures/preprocessing.png)

## Image Classification

For the classification, two models will be compared as feature extractors:
* ViT [1] - pretrained model from [lukemelas/PyTorch-Pretrained-ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT) 
on GitHub (it has very detailed and useful readme). This model is trained on ImageNet-21K. ![image info](Project/Figures/Presentation1_vit.png)
* VGG [2] - pretrained model from ```torchvision.models```. This model is trained on ImageNet. ![image info](Project/Figures/Presentation1_vgg.png)

### Feature Extraction
For each model, the features obtained by replacing the last layer in the original model with an identity layer, 
because these models trained in ImageNet and therefore outputs 1000 class while for this stage only the learned features 
of the model are wanted (and not it's classification). 

Example for VGG:
```python
import torch.nn as nn
from torchvision import models

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Identity()
```

Now inference with ```model``` will yield features insted of classification.

---
## Results

### VGG
![image](Project/Figures/results_vgg.png)

  Time   | Epoch | Learning rate |  Loss  | Val accuracy |  Val (tp, tn, fp, fn)   
:---------:|:-------:|:---------------:|:--------:|:----:|:---------------------------:
00:00:16 | 01/10 |    5.0e-05    | 0.4137 |  80.89%|    42.4%,   38.5%,   11.5%,   07.6%
00:00:32 | 02/10 |    5.2e-04    | 0.3501 |  84.06%|    41.8%,   42.3%,   07.7%,   08.2%
00:00:48 | 03/10 |    1.0e-03    | 0.2412 |  84.47%|    39.7%,   44.7%,   05.3%,   10.3%
00:01:04 | 04/10 |    5.2e-04    | 0.1306 |  84.56%|    38.9%,   45.6%,   04.4%,   11.1%
00:01:20 | 05/10 |    5.0e-05    | 0.0594 |  85.71%|    41.7%,   44.0%,   06.0%,   08.3%
00:01:36 | 06/10 |    2.9e-04    | 0.0479 |  85.45%|    41.9%,   43.5%,   06.5%,   08.1%
00:01:52 | 07/10 |    5.2e-04    | 0.0259 |  85.30%|    43.5%,   41.8%,   08.2%,   06.5%
00:02:09 | 08/10 |    2.9e-04    | 0.0105 |  85.58%|    42.0%,   43.5%,   06.5%,   08.0%
00:02:25 | 09/10 |    5.0e-05    | 0.0066 |  85.78%|    41.5%,   44.3%,   05.7%,   08.5%
00:02:41 | 10/10 |    1.7e-04    | 0.0060 |  __85.75%__|    41.5%,   44.3%,   05.7%,   08.5%



### ViT
![image](Project/Figures/results_vit.png)

  Time   | Epoch | Learning rate |  Loss  | Val accuracy |  Val (tp, tn, fp, fn)   
:---------:|:-------:|:---------------:|:--------:|:----:|:---------------------------:
00:00:10 | 01/10 |    5.0e-04    | 0.6887 |  53.07%|    15.4%,   37.6%,   12.4%,   34.6%
00:00:20 | 02/10 |    5.2e-03    | 0.5822 |  67.68%|    31.9%,   35.8%,   14.2%,   18.1%
00:00:31 | 03/10 |    1.0e-02    | 0.4923 |  70.24%|    32.4%,   37.8%,   12.2%,   17.6%
00:00:41 | 04/10 |    5.2e-03    | 0.4508 |  72.05%|    34.6%,   37.4%,   12.6%,   15.4%
00:00:51 | 05/10 |    5.0e-04    | 0.4225 |  72.31%|    33.5%,   38.8%,   11.2%,   16.5%
00:01:02 | 06/10 |    2.9e-03    | 0.4265 |  72.66%|    34.0%,   38.7%,   11.3%,   16.0%
00:01:13 | 07/10 |    5.2e-03    | 0.4229 |  74.11%|    32.5%,   41.6%,   08.4%,   17.5%
00:01:23 | 08/10 |    2.9e-03    | 0.4042 |  74.37%|    35.9%,   38.5%,   11.6%,   14.1%
00:01:34 | 09/10 |    5.0e-04    | 0.3863 |  75.15%|    36.8%,   38.4%,   11.6%,   13.2%
00:01:44 | 10/10 |    1.7e-03    | 0.3872 |  __76.01%__|    39.9%,   36.1%,   13.8%,   10.1%

---
## Conclusion
Although ViT pretrained on a larger dataset, and achieves better accuracy on most of the common datasets (compares to CNNs),
on this dataset the VGG model (CNN based) performed much better than the ViT model. 

>For future work, it would be interesting to:
> * Train both models from scratch (instead of using transfer learning) and see if the CNN model still achieves superior results.
> * Use data augmentation in order to provide more general features.
> * Check another datasets from the medical field and check if CNN based models still outperforms ViT.

---
## References
- [[1]](https://arxiv.org/abs/2010.11929) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
- [[2]](https://arxiv.org/abs/1409.1556) Very Deep Convolutional Networks for Large-Scale Image Recognition.


 -->
 

