# Q-learning Atari AI

(Emulator) ▼
Open AI gym used to include the atari roms but now they need to be installed with a license using ALE-py. You can use many Atari games and optimize the network for each, I use Mspacman because it runs well. 



(Main Scipt.py ▼)

This project is forked from an older one using Tensorflow 1.15. This has been updated to TF 2.12.0 which has much easier syntax. This is a very simple script that creates the learning environment in real time with no save state feature. This is more of a functional demo because the learning rate is limited by frame rate and other factors. The network is also not well optimized for this type of environment. 


(Q.py ▼)



(Packages) ▼

TensorFlow 2.12.0 (Keras)

Gymnasium 1.1.1 (Updated from OpenAI Gym)

ALE-py 0.10.1
