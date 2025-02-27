# Hidden Markov Model for Speech Recognition

This project is an assignment code compilation of CS 304 Course at Duke Kunshan University taught by Prof. Li

The project uses data from TI Dights, an ancient Dataset that contains several thousands of digit sequence recordings from dozens of speakers. Because of the data is recorded before standardized C is even a thing, the included conversation software in TI Dights CD-ROM may not work as expected. Instead, we can use FFmpeg to convert the recordings to modern standards.

## Results

At its ultimate form, unrestricted HMM with continuous speech sequence training, it can reach a accuracy of 85% for TI Dights test Dataset. The accuracy means the predicted sequence matches exactly the true sequence. An interactive script is provided for test purpose as well. 