# DEER MIGRATION PROJECT

## Project Ideas
1. Data Ingestion Pipeline – Script to import images from; say, Google Drive to a designated folder (say X) using some 
API such as [this](https://developers.google.com/drive/api/v3/manage-downloads) . We would also need to maintain a text 
file log for each batch of images we ingest for tracking purposes.

2. Image Preprocessing – Take all images from X, apply set of image transformations (depending on various times of the 
day) – I hope these images have time stamps. If not, we have to apply some logic to infer time of the day. After 
transformation drop transformed images to another folder (say Y). Erik – Feel free to make changes here as you deem fit

3. Neural Network – Possibly use a pretrained network (e.g. CIFAR) to classify each mage into 1 (Animal) vs 0 
(No Animal). Move all images classified as 1 into yet another folder (say Z)

4. YOLO v3 – Now use this network to perform object detection on all images in folder Z. The program should again create
 a log that gives the count of all deers in an image. We will try to output some other characteristics later to 
 understand animal movement dynamics
 
 5. Report – Use log generated by Step – 4 above to create a human readable report (Not sure if we can use Tableau 
 here / html perhaps) for human consumption. Will be good to have an email alert system that sends out emails when the 
 report is ready – there are plenty of APIs that connect to Google SMTP pretty easily to relay emails.
 

## User Steps from beginning to end (Check off when completed)
When we have completed an item we can check off that it is complete.   
- [x] Example item that has been checked off

List of items to complete:
- [ ] Load files from camera into folder &rarr; Potentially google drive folder  
    - [ ] If it is a google drive folder create a script to connect to the google drive api and transfer files to an
    internal folder for storage.  
- [ ] Maintain a text file log for each batch of images for tracking purposes  
- [ ] Classify the images based on time of day  
    - [ ] If time stamp is available use data to separate the images into separate folders.  
    - [ ] If no time stamp is available create a script to classify the images based on the image properties.  
- [ ] Preprocess the photos e.g. apply image transformation logic.  
- [ ] Transfer prprocessed photos to a new staging folder  
    * Question - Do we want to maintain images as they go through their different stages or do we just want to keep
    raw images and finalized images? If we maintain photos in their staged form it will eat up available space much 
    quicker.  
- [ ] Feed the photos into the neural network to classify them into two groups. Animal or no animal.  
- [ ] Use the neural network to perform object detection on the images containing animals  
- [ ] Based on the data returned by the neural net create a log file to capture the data returned from the neural 
network.   
    * Animals Counted. Hierarchy total, location possibly by time.  
    * Total animals counted breakdown of how many of each animal total and by location or time of day.  
    * See if there is any other breakdown they would like us to provide.  
- [ ] Decide on how to return the results. Discussed options:  
    * Django Website  
    * Tableau  
    * Emailed Report  
- [ ] Other items to discuss  
    * If we collect the data in a database it will be easier to return reports and allow them to compare totals over
    time and locations from different batches of photo dumps.  

## Project Coding Instructions
We can discuss how we want the code to be integrated together. I have tried to set up the repository so we can build
separate modules such as transformations, preprocessing, and classification in separate modules then we can import them
all into the central app.py file and only include necessary functions.0 

# Research and Articles
## Reference Links:
* [CHPC Hosting Services](https://www.chpc.utah.edu/resources/hosting.php)
* [Deep Learning Basics Video](https://www.youtube.com/watch?v=j-3vuBynnOE&vl=en) Provided by Ronak in 12/26/19 Meeting
* [PNAS Research Article on Automatically Classifying Animals in Images](https://www.pnas.org/content/115/25/E5716) Work
conducted on the Serengeti Project.
* [Medium Article on PNAS Project](https://medium.com/coinmonks/automated-animal-identification-using-deep-learning-techniques-41039f2a994d)
* [Serengeti Project](https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti/classify) Camera Trap project
with images that have been classified and an established structure.
* [Lila BC](http://lila.science/datasets/snapshot-serengeti) More work done on the Serengeti Project

## Steps for image pre-processing
[Image Pre-processing Article](https://towardsdatascience.com/image-pre-processing-c1aec0be3edf)  
* Read image
* Resize image
* Remove noise(De-noise)
* Segmentation  
    * Note: The segmentation step is only useful for segmentation problems, if your AI -Computer Vision problem does 
    not include segmentation, just skip this step. 
* Morphology(smoothing edges)

