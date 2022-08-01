<!-- ## Classifying Toxic Memes with Artificial Intelligence -->

## Introduction

In recent years, memes have grown into one of the most widespread forms of content on social media. While generally intended for humor, hateful and misleading content has been on the rise (Fig. 1).  Hence, an effective method of classifying memes into toxic or non-toxic is a major issue to be tackled to ensure a healthy atmosphere online. 

We intend to build a meme classifier that can augment the human moderators in filtering out the toxic ones. The dataset used for the project would include the data from Meta’s Hateful Memes Challenge [3].


## Problem

Due to the massive number of memes being posted on the internet daily, no human team of moderators can effectively filter out every harmful meme containing hate speech, cyberbullying, propaganda, and other toxic behaviors. Therefore, high-bandwidth machine learning algorithms capable of detecting such memes can be extremely helpful in reporting such harmful content. 

Frequently, the harmfulness of a meme is due to the combination of both image and text, e.g. the text may be harmless with a certain background image but harmful when used with another image. Therefore, unimodal models can’t perform the above-mentioned tasks with high accuracy. To tackle this problem, we intend to develop a multimodal model similar to those used in [1,2,4,5] that is capable of classifying harmful memes based on both the background image and embedded text.

## Data Collection

The meme dataset was obtained from Meta’s AI Hateful Meme Challenge Dataset.  The 12,140 memes are pre labeled as 0 (not hateful) or 1 (hateful).  Training set includes 8500 memes and the rest are testing.  Each meme contains a single image and caption.  For clustering, only the training images are used.  External measures are based on the 0 or 1 binary hate classification. 

## Methods

Feature extraction from captions was performed using BERT, a pre-trained neural network to get sentence vectors. Similarly, feature extraction from images was performed using ResNet50, another pre-trained neural network.  Additionally, the layer outputs of both neural networks were saved at each stage of applying the respective model to the dataset.

Upon generating features, we fuse the features from different stages of the pre-trained neural network. We have 6 combinations in our case. 
1. Early-Early: Early layer features of images and captions.
2. Early-Mid: Early layer features of images and mid layer features of captions.
3. Early-Late: Early layer features of images and late layer features of captions. 
4. Late-Early: Late layer features of images and early layer features of captions.
5. Late-Mid: Late layer features of images and mid layer features of captions.
6. Late-Late: Late layer features of images and late layer features of captions. 

Next, we performed PCA separately on the image and the text features. After reducing the number of components, while preserving 95% of the variance, we tried various algorithms like KMeans, DBSCAN and GMM. We performed a full analysis with GMM as results obtained with DBSCAN algorithm are poor and KMeans is a special case of GMM.

Feature reduction and clustering was explored with PCA, KMeans, DBSCAN, and GMM.  The purpose of this was to reduce features and visualize the high dimensional dataset. For GMM, we performed clustering with both, a full covariance matrix and a spherical covariance matrix. 

<p align="center">
    <img align="center" src="/Screen Shot 2022-07-16 at 6.44.31 PM.png" />
</p>
<p align="center">
    <em>When memes go extreme. Example taken from [4]</em>
</p>

 



## Results

# unsupervised

We take features obtained from early and later stage layers from ResNET-50 and early, middle and later stage layers from BERT. Using GMM for clustering, we conclude that the features from the 11th layer of the BERT produce the best results individually and results from the fully-connected layer(FC) of ResNET-50 give the best results without data fusion. The results are compiled in Table 1.



<center>
<table class="center">

    <caption><b>Table 1.</b> Homogeneity results for clustering early and late stage features from BERT applied to captions. </caption>

<tr>

<th>BERT Layer </th>

<th>Homogeneity score
</th>

</tr>

<tr>

<td>Layer 1</td>

<td>0.0056</td>

</tr>

<tr>

<td> Layer 7 </td>

<td>0.0078</td>

</tr>
 
<tr>

<td>Layer 11</td>

<td><b>0.0169</b></td>

</tr>

<tr>

<td>Layer 12</td>

<td>0.0147</td>

</tr>

</table>
</center>


Next, we fuse features from various layers. The fusion techniques employed also improve the results obtained individually just from image or text data. We report homogeneity scores for each of the cases.  The results can be seen in Table 2.

<center>

<table class="center">

    <caption><b>Table 2.</b>  Homogeneity results for clustering fused features from BERT and ResNet50.</caption>

<tr>

<th> </th>
<th>Text only</th>
<th>Image only</th>
<th>Fusion</th>

</tr>
 
<tr>
 <td> FC, Layer 11</td>
 <td> 0.0169 </td>
 <td> 0.016 </td>
 <td> <b>0.022</b> </td>
</tr>

</table>
</center>

Further, we try various fusion techniques as described in the section above. As expected, concatenating results from the FC layer of ResNET and the 11th layer of BERT gives the best results. Late-Late fusion technique gives the best results because of the fact that later stage features are much more abstract. Conventional algorithms like GMM tend to perform poorly on complex and less abstract features. Using neural networks for Early stage fusion techniques will be part of future work. The results are compiled in Table 3.

<center>
<table class="center">

<caption><b>Table 3.</b>  Homogeneity results for clustering early and late stage features from fused feature layers.</caption>

<tr>

<th> </th>
<th>Early (Layer 1)</th>
<th>Middle (Layer 7)</th>
<th>Late (Layer 11)</th>

</tr>
 
<tr>
 <td> Early (Layer 2) </td>
 <td> 2.38e-5 </td>
 <td> 2.1e-5 </td>
 <td> 2.91e-6 </td>
</tr>
 
<tr>
 <td> Late (Layer 10 FC)</td>
 <td> 0.0070 </td>
 <td> 0.0103 </td>
 <td> <b>0.022</b> </td>
</tr>

</table>
</center>    
    

For visualization, we tried different covariance types in the GMM algorithm. We observed that GMM with spherical covariance matrix worked better than the full covariance matrix in all of our runs. From this, we can conclude that the features are much less correlated and hence, by adding extra covariance terms, the model might not converge to the optimum solution. Instead, the spherical covariance matrix gives better and faster results. 

Finally, we tried concatenating features obtained from various layers of BERT. For example, we concatenated features obtained from layer 11,12,13 from BERT. However, we report a decrease in the homogeneity score, as shown in Table 4. Exploring concatenation from various layers of the network will be part of future work.

<center>
<table class="center">

<caption><b>Table 4.</b>  Homogeneity results for clustering concatenated features from BERT layers applied to captions.</caption>

<tr>

<th> Layers </th>

<th>Homogeneity score</th>

</tr>

<tr>

<td>11</td>

 <td><b>0.0169</b></td>

</tr>

<tr>

<td> 11, 12, 13 </td>

<td> 0.0157 </td>

</tr>
 
<tr>

<td>11, 12</td>

<td>0.0167</td>

</tr>

</table>
</center>

Finally, to visualize a meaningful representation of the training dataset, we concatenated pairwise image and text layers from ResNet50 and BERT, respectively.  tSNE was implemented to reduce the 1768 features pertaining to the 8500 training samples to a 2-dimensional embedding (Fig. 2).  Subsequently, KMeans clustering (n = 2) was performed on the embedding to attempt partitioning of the samples into Not Hateful (0) or Hateful (1) categories.  Fig. 2 illustrates various examples of feature concatenation from early or late BERT and ResNet50 features.  Upon closer examination and combined with the results above, we see that unsupervised learning has performed poorly. This is expected as we have not optimized any parameters for the Hateful Memes dataset. This forms our motivation to do supervised learning with the fused features.


fig2

# supervised learning




## Conclusions

In this work, we report improvement in accuracy when features from image and text are fused. We explore various fusion techniques and conclude that Late-Late fusion techniques give the best results. Next to visualize the data, we tried various types of covariance matrix and concluded that features are much less correlated and hence giving better results for spherical objects. Finally, we try concatenating features from various layers and report a drop in accuracy.

## Future work

We will build a supervised machine learning model which will be trained on the fusion feature combos yielding the best homogeneity score. There will also be a “not so good” fusion feature combination trained supervised machine learning model. Finally, we will have golden standard results, which would be obtained from VisualBERT trained on an uncleaned dataset. A comparison of ML metrics among the three will be presented. We will further try to have multiple classes of toxicity and train the model to predict the type of toxicity. 

## References

1. Pramanick, S., et al. "MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets." arXiv preprint arXiv:2109.05184 (2021).
2. Dimitrov, D., et al. "Detecting propaganda techniques in memes." arXiv preprint arXiv:2109.08013 (2021).
3. Kiela, D., et al. "The hateful memes challenge: Detecting hate speech in multimodal memes." Advances in Neural Information Processing Systems 33 (2020): 2611-2624.
4. Sharma, S., et al. "Detecting and Understanding Harmful Memes: A Survey." arXiv preprint arXiv:2205.04274 (2022).
5. Mogadala, A. et al. "Trends in integration of vision and language research: A survey of tasks, datasets, and methods." Journal of Artificial Intelligence Research 71 (2021): 1183-1317.
