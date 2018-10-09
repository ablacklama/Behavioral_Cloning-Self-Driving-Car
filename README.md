# Behavioral_Cloning-Self-Driving-Car

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

This project is modeled after the paper above in which Nvidia traines a car to drive itself with only a camera.

A Convolutional Neural Network is used to take in images from a camera onboard the car and produce the steering command for the car. The model was trained on images and steering commands from a human driver for a few laps of both test courses. These images were then augmented in many ways to help the model generalize better. In the simulation videos below you can see how the model has no problem determaining which way to turn. The only problems it encounters are on the hard course because I've set the car to go so fast that it either doesn't have a lot of time to turn, or in some cases, the car jumps into the air.

Appologies for the video quality.

Easy Course: https://www.youtube.com/watch?v=vjSXBD6uuho

Hard Course: https://www.youtube.com/watch?v=vt5NwY5Tj_k


