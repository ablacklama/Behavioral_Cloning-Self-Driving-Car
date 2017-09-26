#**Behavioral Cloning Using Data Augmentation** 



---




###Model Architecture and Training Strategy

####1. Architecture







In making my model, I started with the [nvidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ "Nvidia blog"). It's proven in the real world and about as big as i can fit in my gpu's memory (8gb). 
From there i played around with different strides, activation functions, and dropout before settling on a medium amount of dropout, and 'relu' activations. For strides i used (3,3) on the first 3 layers and (1,1) on the last two. I also apply (2,2) subsampling on every layer but the last.
I used a batch size of 64 and 6 epochs as training seemed to stop after that. And an Adam optimizer which picked it's own learning rate.

My training data is where it gets somewhat interesting. I used 4 laps of track one and one of track 2 to train. A comparitively small amount to what some other people used. I was able to get away with this because i used a large amount of data augmentation.
First, I tripled my data by using the left and right images as well and applying a +/- 0.25 offset to the steering angle based on what side the image was from. 
After that i removed some of the data with straight angles to make a better balance between turning data and straight data. This solved a problem i was having where the car didn't turn hard enough on sharper turns.
Then i doubled that data by horizontally flipping the images and reversing the steering angle.
At this point i started modifying the images in major ways. I applied two transormations to random samples of the training data, sometimes overlapping:

##1. Changing the Brightness
I reduced the brightness by a random amount to some of the images. This helps the model generalize better for night conditions and shadows.

##2. Change the Colors
I changed the hue of some images by a random amount (ie +20 for the first image, -62 for the second). I also adjusted the saturation a little bit but not much. This had a great effect because it changed what the area around the road looked like. The road, being mostly grey, was the least modified element. This helped the model identify the edges of the road when they changed color in the simulator, and made it rely on the line between the road and the side rather than relying on the color of the sides.

I left the validation data alone so that i could see how the model was impoving on the real data whenever i experimented with different architectures.


I found there were several critical points in the track where you could fail:
##1. The bridge
While almost all models made it over the bridge successfully, it did present an initial challenge for my models because it was colored differently than the rest of the road.

##2. The dirt offramp
the dirt offramp provided more of a challenge. This is where i saw the biggest change once i implemented the color changing step. 
The model had very little data next to that turn since it was such a small portion of the track so changing the colors of half the database randomly forced the model to no longer rely on color.

##3. the sharp turn.
This turn was a problem because the model was really optimized to go straight with small turns. Sharp turns once again represented a tiny portion of the dataset. 
To rectify this i removed large amounts of images with straight or near straight driving angles. This evened out the data somewhat.


###Future improvements
I still feel the model has some optimizing i can do to it's architecture. I haven't tried batch normalization yet and while "relu's" seemed to preform better than "elu's" or a more linear model, i still am not sure they are the best.
The model can sometimes have trouble encountering shadows, and it would be nice to implement a function to artificially introduce shadows into the test set to try to furthur generalize the model.
Lastly, i would love to try to do this same thing with a network containing LSTM cells. It would require almost completely reworking the project, but if done right i believe it would greatly improve performance. Where we drive and how we're turning is inherently linked to what we just did a second ago, and and LSTM network would be able to account for that.
