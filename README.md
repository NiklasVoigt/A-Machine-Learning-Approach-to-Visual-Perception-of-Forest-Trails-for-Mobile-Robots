# A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots

This is a pytorch implementation inspired by the paper "A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots, A. Giusti et al."


## Source
|  |  |
|:---------|:---------|
| Paper | http://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf |
| Dataset | http://people.idsia.ch/~giusti/forest/web/ |

## Abstract
Abstract — We study the problem of perceiving forest or mountain trails from a single monocular image acquired from the viewpoint of a robot traveling on the trail itself. Previous literature focused on trail segmentation, and used low-level features such as image saliency or appearance contrast; we propose a different approach based on a deep neural network used as a supervised image classifier. By operating on the whole image at once, our system outputs the main direction of the trail compared to the viewing direction. Qualitative and quantitative results computed on a large real-world dataset (which we provide for download) show that our approach outperforms alternatives, and yields an accuracy comparable to the accuracy of humans that are tested on the same image classification task. Preliminary results on using this information for quadrotor control in unseen trails are reported. To the best of our knowledge, this is the first letter that describes an approach to perceive forest trials, which is demonstrated on a quadrotor micro aerial vehicle.

## Results of this implementation

| Training Epochs | Main (Training) Accuracy | Validation Accuracy | Test Accuracy |
|:---------|:---------|:---------|:---------|
| 90 | 97.04% | 96.27% | 96.97% |
