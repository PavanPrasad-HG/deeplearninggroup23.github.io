## TASK DIVISION
**CODE**
- LGBM (Pavan)
- Wavelet + FC layers (Tishar)

**BLOG**
- Explain data split (Pavan)
- Update model layout (Marcus)
- Results
- Dicussion: Not enough data, Decrease or split channels (Marnix)




## Structure
- Background
- Intro
- Explaining the data: Pavan
- Explanation (General architecture: TCN: Marcus, Triplet loss: Marnix, Haars: Tishar, LGBM classifier: Tishar)
- Approach taken (Same architecture but Data missing -> Cross-validation: Pavan, What do we have / Assumptions used: Marcus)
- Results / If it doesn't work -> our process so far / where we got stuck 
- Discussion (can't replicate)

## Introduction
### Background:
With privacy taking a center stand in all identification operations, alternative methods for driver identification have gained importance over time. 
Drivers’ use of turn indicators, following distance, rate of acceleration, etc. can be transformed to an embedding that is representative of their behavior and identity as it is very individualistic when compared to each driver. 

We try to harness this in order to create digital fingerprint of a driver based on these behaviors to assist in driver identification. But, drivers behave differently according to changing conditions and different road types and hence lot of training data is required to develop a highly capable system.The goal of the trained model is to detect the driver based on a short 10 second snippet of driving data. 

To achieve this goal,we design a customized deep learning architecture that leverages the advantages of temporal convolution with cross-correlation, the Haar wavelet transform, triplet loss and gradient boosted decision trees’. We train this model on a dataset of driving data collected from a driving simulator designed by Nervtech, a high-end driving simulation company.

### Overview of whole model:

![Image](Model.png)


## Dataset

The dataset used was collected froma driving simulator built by Nervtech that is demonstrated to reproduce an environment that invokes realistic driver reactions. 

The original dataset used by the paper contained a dataset where each driver spent approximately 15 minutes on the simulator, accumulating to more than 15 hours of driving in total. However, the entire dataset was unavailable for recreation and a sample dataset containing 10 seconds of data of each driver is available. Hence, complete recreation of the paper becomes impossible but a try of that is attempted.

The dataset is divided into groups as shown in the below figure to assess the impact of each group on the performance of the model. Pandas dataframe is used to load the model and divide into the sub-groups.

<p align="center">
<img src= Dataset_groups.png/>
</p>

The dataset is further constructed as triplets(x<sub>r</sub>, x<sub>p</sub> and x<sub>n</sub>) as required in triplet loss. Here x<sub>r</sub> denotes an anchor point, x<sub>p</sub> denotes a positive sample of the same driver as x<sub>r</sub>,x<sub>r</sub>  


## Network Architecture

### TCN

The neural network used in this paper is a Temporal Convolutional Networks. This network makes use multiple 1D fully convolutional network stacked on top of each other. A key characteristic of the TCN is that each ouput at time _t_ is only convolved with elements which are before _t_. In practise this means the last element in the series can see the whole series from the beginning. The following image shows how this structure looks.

<p align="center">
<img src= TCN_layer.png/ width=70% height=70%>
</p>

### Triplet loss
The Loss function used for the TCN is triplet loss. With this loss function the reference input called the anchor <img src="https://render.githubusercontent.com/render/math?math=x_a"> is compared with a matching positive pair <img src="https://render.githubusercontent.com/render/math?math=x_p"> and a matching negative pair <img src="https://render.githubusercontent.com/render/math?math=x_n">. This is done by feeding these data points through the current model and computing the distance between the anchor and its matching outputs. The loss is then defined as:

<img src="https://render.githubusercontent.com/render/math?math=l(x_a,x_p,x_n) = max(0,D^2_{ap} - D^2_{an}+\alpha)">

Here <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is the margin and <img src="https://render.githubusercontent.com/render/math?math=D"> is the distance. In this model, the loss optimization objective is then to achieve <img src="https://render.githubusercontent.com/render/math?math=D^2_{ap} \gg D^2_{an}">.

### Haars
The haar wavelet transform is used as a method of indexing time series. Also known as DB1. This method is often better than discrete Fourier transform. The advantage it has over fourier transform is temporal resolution. It captures both frequency and  location information (location in time).  We use a Haar wavelet transformation to generates two vectors in the frequency domain. These vectors are cA and cD. Here cA is approximation coefficients vector and cD  is detail coefficients vector of the discrete wavelet transform The haar wavelet returns a tuple of cA and cD. Other application of Haar wavelet are de-noising and compression of signals and images. The vectors from Haar wavelet is then used as an input to a fully connected linear layer. The output then gets concatenated with the TCN output and fed to the  LightGBM classifier.

### LGBM
Tishar


## Approach taken

#### Model assumptions
The structure of our TCN was chosesn as follows: 
We have 5 TCN blocks in total. Each block contains 2 times a 1D Convolution, Chomp(remove extra padding), Relu and Dropout. A visual representation is seen in the image below:

<p align="center">
<img src= TCN_block.png/ width=40% height=40%>
</p>

The input of each block has 38 channels corresponding to each input signal and the final output also has 38 channels. Each channels has 10 seconds of driver data which amounts to a 1000 data points. The rest of the parameters used are:

- Kernel size = 16
- Padding = 2
- Stride = 1
- Dropout = 0.1

This is how the whole TCN sequence looks like:

<p align="center">
<img src= TCN_sequence.png/ width=75% height=75%>
</p>


## Results
At the moment we are able to train the model but unfortunately there is an issue with the computation of the loss. It seems the losses very large and the network is never able to optimize. Upon inspection of the outputs we think it'd due to the weights getting very large. 


## Discussion


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/PavanPrasad-HG/deeplearninggroup23.github.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/PavanPrasad-HG/deeplearninggroup23.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
