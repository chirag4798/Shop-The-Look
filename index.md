# Shop The Look

[Webapp Demo](https://www.youtube.com/watch?v=YlWKfkpZ9h0)
<iframe width="560" height="315" align="center" src="https://www.youtube.com/embed/YlWKfkpZ9h0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

End-to-end content based fashion product recommendation engine based on visual similarity of items.

**This blog is inspired by the research paper published by Myntra**  
<p align="center">
<a href="https://arxiv.org/pdf/2008.11638.pdf"><img src="https://i.imgur.com/FEklYUC.png">Buy Me That Look: An Approach for Recommending Fashion Products</a>
</p> 
          
## Introduction
<p align="justify">
Have you ever looked at a model and wondered if you could ever look like him/her? Ever wondered how to develop a good fashion sense? Don't worry this blog has a very elaborate (a bit much if you ask me) solution to both of those questions. The rise of e-commerce websites in India has led to an increase in online shopping throughout the country. Fashion items are the most sought after in the e-commerce market. A robust, effective, and efficient recommendation system for fashion items could lead to an increase in revenue, better customer engagement, and experience. What makes this approach novel is the ability to detect multiple products from a given Product Display page and recommend similar items to the user. A Product Display Page (PDP), is a web-page that presents an image of the product/products worn by a model or a standalone picture of the product in a plain background. This approach helps the user to not only look for the relevant item in a single category but also to match those items with other items that go well with the specified product. In simple words, thanks to the advancements in computer vision, you can copy the look of your favourite celebrity by just using an image of that look. 
</p>

## Proposed Solution
<p align="justify">
The proposed solution is a Content-Based recommendation system that involves various modules that carry specific tasks that help in achieving the overall goal. For the system proposed in the published article by Myntra the first module consists of the Human Key Point Detection, looks for specific human key points like the ankle and the eyes in order to classify the image as a full-shot image i.e. an image that consists of the full body of a model, however the solution proposed in this blog skips this step, instead we'll recommend items using all the fashion products detected in an image. For better product recommendation we'll first try to determine the gender of the person in the query image. The pose can either be front, back, left, right or detailed. After determining the gender, the input is fed is then fed to the next module, it’s an Object Detection module trained to identify objects from these broad categories - top-wear, bottom-wear, and foot-wear. The goal of this module is not just the identification of these objects in the image but also to localize the objects with bounding-boxes. Once the bounding boxes are obtained for a certain object, that part of the image is cropped and the semantic embeddings for the cropped image are obtained, which will help in identifying similar products from the catalog/database of products whose semantic embeddings are already known. We'll train a Siamese Network for generating the embeddings for the catalog of images. This system helps in automatically and efficiently recommending new items to a user who has selected or shown interest in a certain product display page.
</p>

## Data Acquisition
<p align="justify">
To get started with the Fashion Recommendation Engine, we'll need a catalog/database consisting of fashion products to recommend. Unless you have a database of these items lying around we'll need to acquire this data using simple web scraping tools in python. Since we'll focus on recommending items accross various categories we'll need a diverse set of fashion products to populate the database. We can use Myntra's e-commerce website to query images using specific keywords and later save the product details for the results obtained. This way we can categorize products into various sections which would make it easier while recommeding items. We'll use Selenium package from python, its the most popular browser automation tool mainly used to carry out web-product testing. For that we'll require webdriver, for chrome users, you can get it from <a href="https://chromedriver.chromium.org/">here</a>,  make sure the version that you download matches with the version of chrome on your computer. For more details on webscraping e-commerce websites using python follow this article "<a href="https://medium.com/analytics-vidhya/web-scraping-e-commerce-sites-using-selenium-python-55fd980fe2fc">Webscraping e-commerce websites using python</a>".  After scraping the image and product urls, you either use it to download the images into your local system or save the image and product urlspair to a flat file and use it later.
</p>

```python
# import necessary libraries
import time
from selenium import webdriver as wb

# Enter the path to your webdriver's .exe file here
wd = wb.Chrome('chromedriver.exe')

# List of search terms to use for querying from the e-commerce sites
search_terms = ["Mens Topwear", "Womens Topwear", "Mens Bottomwear", "Womens Bottomwear", "Mens Footwear", "Womens Footwear"]

# Define the Maximum number of products you want to scrape per search term
MAX_PRODUCTS_PER_CATEGORY = 1000
image_urls, product_urls = list(), list()
for search_term in search_terms:
    search_term = search_term.lower()
    query = f"https://www.myntra.com/{search_term.replace(' ', '-')}"
    total = 0
    wd.get(query)
    while total < MAX_PRODUCTS_PER_CATEGORY:
        time.sleep(3)
        products = wd.find_elements_by_class_name("product-base")
        for product in products:
            try:
                image = product.find_element_by_tag_name("img")
            except:
                continue
            image_urls.append(image.get_property("src"))
            product_urls.append(product.find_element_by_tag_name("a").get_property("href"))
    try:
        wd.find_element_by_class_name("pagination-next").click()
    except:
        break
```
## Exploratory Data Analysis
<p align="justify">
Now that we've gathered the catalog, lets do some exploratory analysis on it. The dataset consists of 10310 fashion products in various categories. The categories being:
</p>    

<p align="center"><a href="https://imgur.com/bdbH1xs"><img src="https://i.imgur.com/bdbH1xs.png" title="source: imgur.com" /></a></p>

<p align="center"><a href="https://imgur.com/D1DzBdb"><img src="https://i.imgur.com/D1DzBdb.png" title="source: imgur.com" /></a></p>

<p align="center"><a href="https://imgur.com/uQM1cYf"><img src="https://i.imgur.com/uQM1cYf.png" title="source: imgur.com" /></a></p>

<p align="center"><a href="https://imgur.com/w4Upxdr"><img src="https://i.imgur.com/w4Upxdr.png" title="source: imgur.com" /></a></p>

<p align="center"><a href="https://imgur.com/TczHf8t"><img src="https://i.imgur.com/TczHf8t.png" title="source: imgur.com" /></a></p>

<p align="center"><a href="https://imgur.com/KKcSvpO"><img src="https://i.imgur.com/KKcSvpO.png" title="source: imgur.com" /></a></p>
  
<p align="justify">
The fashion products are not equally split among the two genders, there are 5992 Mens Products whereas as just 4318 Womens products. The plot below is a visual representation of the products split by genders.
</p> 

<p align="center"><a href="https://imgur.com/btlsNp5" align="center"><img src="https://i.imgur.com/btlsNp5.png" title="source: imgur.com" /></a></p>

<p align="justify">
Distribution across the categories of Topwear - 3770, Bottomwear - 3524 and Footwear - 3016.
</p> 

<p align="center"><a href="https://imgur.com/FqAAPjC"><img src="https://i.imgur.com/FqAAPjC.png" title="source: imgur.com" /></a></p>

<p align="justify">
There is huge disparity in the distribution among the type of clothing. Around 60% of the fashion items are of type casual whereas formal and party wear occupy just 30% and 10% respectively. The recommendations for casual would obviously be better considering the variety of items present in them. Here's a visual representation.
</p> 

<p align="center"><a href="https://imgur.com/nQ1iLwx"><img src="https://i.imgur.com/nQ1iLwx.png" title="source: imgur.com" /></a></p>


## Gender Classification
<p align="justify">
In order to recommend relevant items to the user we'll have to first detect the gender of the person present in the query image. Doing so will allow us to generate and query embeddings only on certain subsets of the catalog instead of the whole database. This will also help us in reducing irrelevant recommendations to the user based on their gender. For this task we'll make use of the <a href="https://www.kaggle.com/playlist/men-women-classification">Man/Woman classification dataset from Kaggle</a>. This is a manually collected and cleaned dataset containing 3354 pictures (jpg) of men (1414 files) and women (1940 files) that includes full body and upper body images of men and women. This is ideal for our use case since we'll mostly have full body images of models wearing various fashion items. For classification we can use a ResNET50 backbone initialized with 'Imagenet' weights attached to a fully connected network that outputs the probability scores for the two classes. We'll freeze all the layers from the backbone since the task is coparitively easy to achieve and inorder to avoid overfitting to this dataset. Below is a code snippet for the model used for classification. We are abe to achieve accuracy of 95% on the test set which is decent for our use case.
</p>          

```python
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNET50
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

base_model = ResNet50(weights="imagenet", input_shape=(224,224,3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)

output = Dense(1, activation="sigmoid")(x)

gender_classifier = Model(base_model.input, output, name="Gender_Classifier")
```

## Object Detection & Localization

<p align="justify">
In order to recommend similar products we'll first need to detect all the products in a given image, and for generating the embeddings we'll need to crop out the product from n image. Object detection is one of the most interesting aspect of Computer Vision. Most object detection systems have some kind of trade off between inference speed and detection accuracy. In order to better understand object we'll need to understand a metric called 'mAP' or Mean Average Precision, for object detection whether a detection is considered correct or not depends on the IoU (Intersection over Union) threshold. The IoU score for a predicted bounding box and actual bounding box is defined as
</p>
<p align="center">
<a align="center" href="https://www.codecogs.com/eqnedit.php?latex=IoU&space;=&space;\frac{\text{Area&space;of&space;Intersection}}{\text{Area&space;of&space;Union}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?IoU&space;=&space;\frac{\text{Area&space;of&space;Intersection}}{\text{Area&space;of&space;Union}}" title="IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}" /></a>  
    
<a href="https://miro.medium.com/max/960/1*FrmKLxCtkokDC3Yr1wc70w.png"><img src="https://miro.medium.com/max/960/1*FrmKLxCtkokDC3Yr1wc70w.png"></a>
</p>
<p align="justify">
For a pre-defined IoU threshold we can define if a detection was accurate or not if the IoU is greater than the thresold. Based on the detections Precision and Recall is defined as.
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Precision}&space;=&space;\frac{\text{TP}}{\text{TP&space;&plus;&space;FP}}&space;\text{&space;\&&space;Recall}&space;=&space;\frac{\text{TP}}{\text{TP&space;&plus;&space;FN}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Precision}&space;=&space;\frac{\text{TP}}{\text{TP&space;&plus;&space;FP}}&space;\text{&space;\&&space;Recall}&space;=&space;\frac{\text{TP}}{\text{TP&space;&plus;&space;FN}}" title="\text{Precision} = \frac{\text{TP}}{\text{TP + FP}} \text{ \& Recall} = \frac{\text{TP}}{\text{TP + FN}}" /></a>
</p>
<p align="justify">
Where TP = True Positive, FP = False Positive & FN = False Negative
</p>
<p align="justify">
The average precision is calculated by using the Area Under the Precision and Recall Curve.
</p>
<p align="center">
<a href="https://miro.medium.com/max/632/0*4lq6j_CHfvqQuvqn.png"><img src="https://miro.medium.com/max/632/0*4lq6j_CHfvqQuvqn.png"></a>
</p> 
<p align="justify">
The Mean Average Precision is calculated by taking mean of the Average Precision values over different values of threshold, for example in the COCO Primary Challenge the mAP was calculated by averaging the AP scores over a range of IoU thresolds from 0.5 to 0.95 with a step size of 0.05 and finally taking the mean of the AP scores over all the classes. For more detailed explanation of mAP follow this blog - "<a href="https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52">Breaking Down Mean Average Precision</a>". Now that we know how Object Detection Systems are evaluated let's checkout various methods for object detection. We'll compare various object detection model's performance against the MS COCO Dataset that contains 80 classes. We'll look at the Inference speed and mAP scores. Below is a plot showing the highest and lowest Frames Per Second (FPS) values reported in their respective papers.
</p>
<p align="center">
<a href="https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359"><img src="https://miro.medium.com/max/1200/1*EQKFp_c6jMYcDZbIwbEOzA.png">Source: Object Detection Speed and Accuracy Comparision</a>
</p>  
<p align="justify">
It is clear that YOLO object detection are faster by a huge margin compared to other object detection methods, however an important thing to note is the plot doesn't mention the mAP scorees for the models with the highest inference speeds. Now lets look at mAP scores for various methods against the MS COCO Dataset.  
</p>
<p align="center">
<a href="https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359"><img src="https://miro.medium.com/max/1200/1*tOkQQ5g2Tp5xWShaO4VUpQ.jpeg">Source: Object Detection Speed and Accuracy Comparision</a>
</p>  
<p align="justify">
From the plot above we can see that RetinaNET has the highest mAP score of 40.8 followed by Faster-RCNN, FPN and YOLO. The difference between the latter three methods is not huge compared to their difference with RetinaNET which is SOTA in object detection. We can safely assume that YOLO will provide us with the right balance of Accuracy and Speed for our task of Fashion product detection and localisation. The research paper by Myntra used Mask RCNN for their object detection module since they had the luxury to run that part offline, however we do not have that luxury if we want to recommend similar products in real time using a web application. Hence we'll use YOLO object detection for detecting fashion products in three categories - topwear, bottomwear and footwear for our simplicity, we can easily improve our recommendation engine by including more and better categories for object detection like classifying the type of shirts, t-shirts, bags, sunglasses, trousers, jeans etc. YOLO refers to “You Only Look Once” is a versatile object detection model. YOLO algorithms divide a given input images into the SxS grid system. Each grid is responsible for object detection. Now the Grid cells predict the boundary boxes for the detected object. For every box, we have five main attributes: x and y for coordinates, w and h for width and height of the object, and a confidence score for that the box containing the object. 
Follow this blog for training an object detector using your custom dataset - "<a target="_blank" href="https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/">YOLOv5 Training for Custom Dataset</a>"</p>
<p align="center"><img src="https://www.pyimagesearch.com/wp-content/uploads/2018/11/yolo_design.jpg"></p> 
<p align="center"><a  target="_blank" href="https://arxiv.org/pdf/1506.02640v1.pdf">Source: YOLO - You Only Look Once</a></p>  



<p align="justify">
For training the custom Object Detection model we'll use Transfer Learning. We'll load YOLOv5s (s stands for small variant of YOLOv5) weights trained on the COCO Dataset and use transfer learning to fine-tune the model for our use case. I've created a dataset of 180 images (135 train and 45 validation) with annotated topwear, bottomwear and footwear.  We were able to achieve a mAP score of 0.94 @ 0.5 IoU threshold on a test set of 45 images. Here are some object detection results on topwear, bottomwear and footwear detection.
</p>  

<p align="center">
<a href="https://imgur.com/fJdtXsB"><img src="https://i.imgur.com/fJdtXsB.png" width=312 height=500 title="source: imgur.com" /></a>
<a href="https://imgur.com/k4GxQ86"><img src="https://i.imgur.com/k4GxQ86.png" width=312 height=500 title="source: imgur.com" /></a>
<a href="https://imgur.com/DDlE3NB"><img src="https://i.imgur.com/DDlE3NB.png" width=312 height=500 title="source: imgur.com" /></a>
</p>

















































