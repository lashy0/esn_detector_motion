## Motion Detector

Implementation of a motion detector in the specified area of interest in the cart.

The current implementation of MotionDetect does not work well with strong overexposure in the frame and obvious shadows in the area of interest, considering them to be movement.

- The **notebooks** folder contains notebooks with an example of using ESN and the old version of MotionDetect
- The **resources** folder contains videos with motion, without motion, and when the light is in the frame
- The **src** folder contains the implementation of the old MotionDetect and the implementation of the updated ESN with utilities for displaying graphs, metrics

### An example of the work

![](assets/example.gif)