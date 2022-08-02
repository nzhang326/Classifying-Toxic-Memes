<!-- ## Classifying Toxic Memes with Artificial Intelligence -->

## Introduction

In recent years, memes have grown into one of the most widespread forms of content on social media. While generally intended for humor, hateful and misleading content has been on the rise across all major social media platforms (**Fig. 1**).  To ensure a healthy atmosphere online, an effective method of classifying memes into toxic or non-toxic is a major issue to be tackled.

We intend to build a meme classifier that can assist human moderators in filtering out the toxic memes posted to social media sites. The dataset used for the project includes the data from Meta’s Hateful Memes Challenge [3].

<p align="center">
    <img align="center" src="/meme_extreme.png" />
</p>
<p align="center">
    <em> Figure 1. When memes go extreme. Example taken from [4]</em>
</p>

## Problem

Due to the massive number of memes being posted on the internet daily, no human team of moderators can effectively filter out every harmful meme containing hate speech, cyberbullying, propaganda, and other toxic content. Therefore, high-bandwidth machine learning algorithms capable of detecting such memes can be extremely helpful in reporting such harmful content. 

Frequently, the harmfulness of a meme is due to the combination of the image and the text, e.g. the text may be harmless with a certain background image but harmful when used with another image. Therefore, unimodal models can’t perform the above-mentioned tasks with high accuracy. To tackle this problem, we intend to develop a multimodal model similar to those used in [1,2,4,5] that is capable of classifying harmful memes based on both the background image and embedded text.


## Data Collection

The meme dataset was obtained from Meta’s AI Hateful Meme Challenge Dataset.  The 12,140 memes are pre labeled as 0 (not hateful) or 1 (hateful).  Training set includes 8500 memes and the testing set includes 1000.  Each meme contains a single image and caption.  For clustering, only the training images are used.  External measures are based on the same 0 or 1 binary hate classification. 

## Methods

Feature extraction from captions was performed using BERT, a pre-trained neural network to get sentence vectors [5]. Similarly, feature extraction from images was performed using ResNet50, another pre-trained neural network [6]. Additionally, the layer outputs of both neural networks were saved at each stage of applying the respective model to the dataset.

To create a multimodal model, we used several pre-trained models to extract features from the text and image portions of the memes. The pre-trained models used include BERT, ResNet50 and CLIP, a “zero-shot” network [7]. BERT is a pre-trained transformer network which was used to extract features from meme captions, and ResNet50, is a pre-trained convolutional neural network which was used to extract features from meme images.

In addition to these two networks, CLIP was used to generate these features as well as provide a baseline for classifying memes as hateful or not-hateful. CLIP is an image classifying network that was trained on a very diverse set of images, allowing it to accurately classify images from any dataset. In this project CLIP was modified to classify the memes as hateful or not-hateful (Table 5). It was also used to extract image and text features from internal image and text encoders. CLIP has multiple internal image encoders, the ones used for this project are a modified ResNet50 model and a custom vision transformer. They provide a single internal text encoder which is a custom transformer-based model. We include these encoders in our project to examine the effect of encoder choice on the ability to classify memes.

Once we extracted the features from these models, we designed our own neural network to fuse both the image and text features and trained it on the Hateful Memes Dataset. To investigate the success of multiple fusion techniques, we used features extracted from several stages of computation in the pre-trained models. 

We have 6 possible fusion combinations in our case:

<p align = 'left'>
1. Early-Early: Early layer features of images and captions.  <br>
2. Early-Mid: Early layer features of images and mid layer features of captions. <br>
3. Early-Late: Early layer features of images and late layer features of captions. <br>
4. Late-Early: Late layer features of images and early layer features of captions. <br>
5. Late-Mid: Late layer features of images and mid layer features of captions. <br>
6. Late-Late: Late layer features of images and late layer features of captions. <br>
</p>

Our neural network was composed of multiple dense layers, with skip connections like those used in ResNet to allow for faster training and to reduce overfitting. Next, we employ a technique similar to bagging where we randomly choose, with replacement, the output of a few layers of text and images and train a neural network. We do the random sampling 10 times and obtain the outputs. Instead of a strict majority vote, we take an average of the sigmoid outputs obtained and make a decision based on that. For the first fusion model where we use outputs from ResNet50 and BERT, we choose 3 layers of text embeddings and the last image layer. In the second fusion model, we choose output of 3 layers of CLIP model. Table 6 shows the results of various feature extractions from the different pre-trained models and fusion techniques.

<p align="center">
    <img align="center" src="/skip_mdl.png" />
</p>
<p align="center">
    <em> Figure 2.  Neural networking with dense layers employing skip connections to reduce training time and overfitting while mitigating the vanishing gradient problem. </em>
</p>

We also investigated clustering of the features. We used PCA separately on the image and the text features then, while preserving 95% of the variance, we tried various clustering algorithms like KMeans, DBSCAN and GMM. The purpose of this was to reduce features and visualize the high dimensional dataset. For GMM, we performed clustering with both, a full covariance matrix and a spherical covariance matrix. GMM provided the best results, however we note that the results are quite poor, largely to the very high-dimensionality of the features.  

Additionally, we implemented T-Distributed Stochastic Neighbor Embedding (tSNE), a manifold technique, to reduce and visualize the high dimensional features in hopes of retaining non linear relationships among the data.  tSNE minimizes the Kullback-Leibler divergence between the data points in joint probability form.  After creating the embedding, KMeans (n=2) clustering was performed to classify Not Hateful (0) or Hateful (1).  Similarly to before, clustering performed poorly due to almost all predictions being positive, resulting in true positives and also false positives.  

Thus, we next considered a supervised fusion model to improve accuracy by incorporating training labels.  To extract more meaningful features, we used CLIP, a powerful pre-trained neural network reliant on zero-shot transfer and multimodal learning, to learn our dataset.  Performance was measured against ResNet50 using accuracy and area under the receiver operating characteristic (AUROC) curves.  Overfitting issues were corrected using dropout layers and skip connections, which also mitigated the issue of vanishing gradients during backward propagation.  

## Results

### Unsupervised Learning

We take features obtained from early and later stage layers from ResNET-50 and early, middle and later stage layers from BERT. Using GMM for clustering, we conclude that the features from the 11th layer of the BERT produce the best results individually and results from the fully-connected layer (FC) of ResNET-50 give the best results without data fusion. The results are compiled in Table 1.

<p align="center">
<center>
<table class="center">
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
        <em><b>Table 1.</b> Homogeneity results for clustering early and late stage features from BERT applied to captions. </em> <br>
</center>
</p>

Next, we fuse features from various layers. The fusion techniques employed also improve the results obtained individually just from image or text data. We report the best homogeneity result in Table 2. 
<center>

<p align = 'center'>
<table class="center">
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
    <em><b>Table 2.</b>  Table 2.  Homogeneity results for clustering fused features from BERT and ResNet50.</em>
</p>

Further, we try various fusion techniques as described in the methods section above. As expected from the homogeneity scores, concatenating results from the FC layer of ResNET and the 11th layer of BERT gives the best results. We hypothesize the Late-Late fusion technique gives the best results because the later stage features are much more abstract. Conventional algorithms like GMM tend to perform poorly on complex and less abstract features. The results are compiled in Table 3.

<p align = 'center'>
<center>
<table class="center">
    
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
<em><b>Table 3.</b>  Homogeneity results for clustering early and late stage features from fused feature layers.</em>
</p>
    

For visualizing the clusterings, we tried different covariance types in the GMM algorithm. We observed that GMM with spherical covariance matrix worked better than the full covariance matrix in all of our runs. From this, we can conclude that the features are much less correlated and hence, by adding extra covariance terms, the model might not converge to the optimum solution. Instead, the spherical covariance matrix gives better and faster results. 

Next, we concatenated features obtained from various layers of BERT. For example, we concatenated features obtained from layer 11,12,13 from BERT. However, we report a decrease in the homogeneity score, as shown in Table 4.

<p align = 'center'>
<center>
<table class="center">

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
<em><b>Table 4.</b>  Homogeneity results for clustering concatenated features from BERT layers applied to captions.</em>

</p>

Finally, to visualize a meaningful representation of the training dataset, we concatenated pairwise image and text layers from ResNet50 and BERT, respectively.  tSNE was implemented to reduce the 1768 features pertaining to the 8500 training samples to a 2-dimensional embedding (Fig. 3).  Subsequently, KMeans clustering (n = 2) was performed on the embedding to attempt partitioning of the samples into Not Hateful (0) or Hateful (1) categories.  Fig. 3 illustrates various examples of feature concatenation from early or late BERT and ResNet50 features.  Upon closer examination and combined with the results above, we see that unsupervised learning has performed poorly. This is expected as we have not optimized any parameters for the Hateful Memes dataset. This forms our motivation to do supervised learning with the fused features.

<p align="center">
    <img align="center" src= 'unsupv.png')
 />
</p>
<p align="center">
    <em> Figure 3. Examples of (A) early-early, (B) early-late, (C) late-early, and (D) late-late fusion clustering from a 2D tSNE embedding followed by KMeans = 2 clustering. Clustering evaluation is high from Fowlkes-Mallows but not Rand due to the large number of predicted positives, resulting in many true positives but also false positives.  
 </em>
</p>


### Supervised Learning

Since the features are extracted from pre-trained models, we directly use a Fully connected model for our use case. Our initial model has 6 Fully Connected layers and all leaky ReLU activations, except for the last activation which is a sigmoid (Fig. 4). 

<p align="center">
    <img align="center" src="/init_model_arch.png" />
    </p>
<p align="center">
    <em> Figure 4. Initial neural network model architecture with 6 fully connected layers and leaky ReLU activations except for the last sigmoid activation. </em>
</p>

<p align="center">
    <img align="center" src="/loss_init.png" />
</p>
<p align="center">
    <em> Figure 5.  Overfitting in our initial model with cross entropy loss. However, the maximum AUROC score obtained 0.706. </em>
</p>

Fig. 5 demonstrates overfitting in all the fused features. Next, dropout is used as a means of regularization to avoid this. The updated model is seen in Fig. 6.

<p align="center">
    <img align="center" src="/drpt_arch.png" />
</p>
<p align="center">
    <em> Figure 6. Neural network model architecture with dropout layers. In our case, dropout layers act as a means of regularization. The value of drop is our hyperparameter which is tuned to 0.5 upon doing cross validation. </em>
</p>

<p align="center">
    <img align="center" src="/loss_drpt.png" />
    </p>
<p align="center">
    <em> Figure 7.  Slight overfitting still exists in our model with cross entropy loss. We can see an increase in the max AUROC score which is 0.7174. </em>
</p>

With this updated model, higher training time and slight overfitting is observed (Fig. 7). To overcome this, we use skip connection based models (Fig. 8) as described in the Methods section. Along with faster learning, this alleviates the problem of vanishing gradients.

<p align="center">
    <img align="center" src="/skip_mdl_arch.png" />
    </p>
<p align="center">
    <em> Figure 8. Neural network model architecture with skip layer connections. </em>
</p>

<p align = 'center'>
<table>
    <tr>
        <td>Metric\Model</td>
        <td>Late-Early</td>
        <td>Late-Middle</td>
        <td>Late-Late</td>
    </tr>
    <tr>
        <td>Max AUROC score</td>
        <td>0.7172</td>
        <td>0.7172</td>
        <td>0.746</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>0.6068</td>
        <td>0.5882</td>
        <td>0.6319</td>
    </tr>
    <tr>
        <td>Recall</td>
        <td>5504</td>
        <td>0.5943</td>
        <td>0.5633</td>
    </tr>
</table>
<em><b>Table 5.  Performance metrics for fused CLIP-based models.</em>
</p>
    
    
<p align="center">
    <img align="center" src="/loss_skip.png" />
    </p>
<p align="center">
    <em> Figure 9.  The model reaches good accuracy before overfitting. We can see an increase in the max AUROC score to 0.7460. </em>
</p>

Next, we tried out a bagging-based method as described in the Methods section. We saw a slight increase in accuracy as bagging relies on majority vote. A slight increase indicates that the features represent nearly the same amount of information for classification.  This gives max AUROC accuracy of 0.7624. For visualization, tSNE plots are obtained as shown (Figs 10, 11).

<p align="center">
    <img align="center" src="/tSNE_bag.png" />
    </p>
<p align="center">
    <em> Figure 10.  tSNE embedding of strong performing model with 0.7624 AUROC. </em>
</p>

Until now, features extracted from pre-trained ResNet50 & BERT are used. To examine the adaptability of zero-shot multimodal learners to the Hateful Memes dataset and to provide a baseline metrics for classification we modified CLIP model to classify memes in the challenge. The baseline metrics were obtained by classifying the memes using the image only (with embedded text), and using text only. The results are shown in Table 6.

<p align = 'center'>
<table>
    <tr>
        <td>Internal Image Encoder</td>
        <td>Text-Only Accuracy (%)</td>
        <td>Text-Only AUROC</td>
        <td>Image-Only Accuracy</td>
        <td>Image-Only AUROC</td>
    </tr>
    <tr>
        <td>Modified ResNet-50</td>
        <td>50.3</td>
        <td>0.49</td>
        <td>51.1</td>
        <td>0.29</td>
    </tr>
    <tr>
        <td>Custom Vision Transformer</td>
        <td>50.7</td>
        <td>0.41</td>
        <td>51.6</td>
        <td>0.2</td>
    </tr>
</table>
<em><b> Table 6.  Classification results from CLIP without fine-tuning. </em>
</p>

From these results, it is clear that even state-of-the art zero-shot learners cannot be directly applied to all applications they were not trained on. Next, we discuss the results of using CLIP embeddings in our supervised learning model.

Finally, extracted features from the state of the art pretrained CLIP model are used, which gives image, text pair features for a dataset. Late-Late based fusion is done using CLIP embeddings to obtain a higher accuracy. We use last layer embeddings obtained from both the CLIP models for images and the last layer text embeddings from the CLIP model for text (Fig. 11). We obtained a final AUROC score of 0.7744.

To conclude our experimentation, we attempted a bagging experiment with CLIP. But a slight decrease in the AUROC accuracy down to 0.7638 is observed. This is because the models were trained with unimportant fused features obtained from initial layers that corrupted the prediction.  This is verified by giving individual layer outputs of text as input, a decrease in accuracy for all but the last layer is observed. 

<p align="center">
    <img align="center" src="/tSNE_bag_CLIP.png" />
    </p>
<p align="center">
    <em> Figure 11.  CLIP embedding of best performing model with 0.7748 AUROC. </em>
</p>

With this we see the best AUROC score of 0.7748 and our score is within top 20 in the hateful memes challenge.


## Conclusions

In our work, we explored various experiments with feature fusion in both unsupervised and supervised learning phases. By using data-fusion techniques our model can classify hateful memes with fewer parameters and shorter train-times than other methods. Next, we see that ensemble learning, where a majority vote considering multiple layers that is used to make a prediction, further improves results. Finally we use features from the powerful, pre-trained CLIP model and conclude our experimentation with accuracy numbers within top 20 in the hateful meme challenge.  

By performing bagging experiments, we observed that results improve much less when ResNet and BERT models are used which implies that the features obtained from several layers have similar information. In case of CLIP, the accuracy with bagging experiment as the final features give more clear information. Such experiments can be extended for other models to quantify and compare information obtained from various layers.

## Future work

Employing convolution layers for image features and recurrent layers for text would be our focus for the future. We would also aim at a multiclass classification of a meme’s toxicity with multiple labels (discrimination, violence, mature etc).

## References

1. Pramanick, S., et al. "MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets." arXiv preprint arXiv:2109.05184 (2021).
2. Dimitrov, D., et al. "Detecting propaganda techniques in memes." arXiv preprint arXiv:2109.08013 (2021).
3. Kiela, D., et al. "The hateful memes challenge: Detecting hate speech in multimodal memes." Advances in Neural Information Processing Systems 33 (2020): 2611-2624.
4. Sharma, S., et al. "Detecting and Understanding Harmful Memes: A Survey." arXiv preprint arXiv:2205.04274 (2022).
5. Devlin, J., et. al. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” arXiv, May 24, 2019. 
6. He, K., et. al.. “Deep Residual Learning for Image Recognition.” arXiv, December 10, 2015. 
7. Radford, A., et al. “Learning Transferable Visual Models From Natural Language Supervision.” arXiv, February 26, 2021.
8. Mogadala, A. et al. "Trends in integration of vision and language research: A survey of tasks, datasets, and methods." Journal of Artificial Intelligence Research 71 (2021): 1183-1317.
