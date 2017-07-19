# sincvideo

This project is a different approach to speeding up or slowing down video. Instead of spline-based approaches (blending in-between frames using linear interpolation) or more complicated optical flow methods, I tried to approximate each in-between frame as a convex linear combination of every frame of the video, [sinc](https://en.wikipedia.org/wiki/Sinc_function) weighted as is customary in signal processing. Treating a video as discrete-time samples of a bandlimited 3-dimensional signal is very mathematically simple and programmatically concise (NumPy's `tensordot` works wonders) but creates time-domain fringing artifacts as shadows and highlights that flash in and out at regions of the video with high temporal frequency and low spatial frequency.

[1x, 2x, 4x, and 16x interpolation of some DSLR footage](https://www.youtube.com/watch?v=P50nGtnLgNA)
