## ON LARGE BATCH TRAINING FOR DEEP LEARNING GENERALIZATION GAP AND SHARP MINIMA
- Large-Batch (LB): 
    + sharp minimizers
    + poor generalization
    + flat minimum can be described with low precision
    + noise in the stochastic gradient is not sufficient to cause ejection from the initial basin leading to convergence to sharper a minimizer

- Small-Batch (SB):
    + converge to flat minimizers
    + better generalization
    + noisy gradients
    + a sharp minimum requires high precision
    + noise in the gradient pushes the iterates out of the basin of attraction of sharp minimizers and encourages movement towards a flatter minimizer where noise will not cause exit from that basin

- a flat minimum can be described with low precision, whereas a sharp minimum requires high precision. This can be explained through
the lens of the Minimum description length (MDL) theory, which states that statistical models that require fewer bits to describe (i.e., are of low complexity) generalize better 
- generalization gap is not due to over-fitting or over-training as commonly
observed in statistics: **early stopping does not help**
- when increasing the batch size for a problem, there exists a threshold after which there is a deterioration in the quality of the model
- it has been speculated that LB methods tend to be attracted to minimizers close to the starting point x0 , whereas SB methods move away and locate minimizers that are farther away
- data augmentation, conservative training and robust optimization: do not correct the problem; they improve the generalization of large-batch methods but still lead to relatively sharp minima
- loss function of deep learning models is fraught with many local minimizers and that many of these minimizers (both sharp and flat) correspond to a similar loss function value
- warm-starting / piggybacked with SB for some epochs, then use LB -> better accuracy than SB-only
