# Loss Functions in PyTorch (2.2)

## nn.L1Loss

- Creates a criterion that measures the mean absolute error (MAE) between each element in the input `x` and target `y`.

## nn.MSELoss

- Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input `x` and target `y`.

## nn.CrossEntropyLoss

- This criterion computes the cross entropy loss between input logits and target.

## nn.CTCLoss

- The Connectionist Temporal Classification loss.

## nn.NLLLoss

- The negative log likelihood loss.

## nn.PoissonNLLLoss

- Negative log likelihood loss with Poisson distribution of target.

## nn.GaussianNLLLoss

- Gaussian negative log likelihood loss.

## nn.KLDivLoss

- The Kullback-Leibler divergence loss.

## nn.BCELoss

- Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.

## nn.BCEWithLogitsLoss

- This loss combines a Sigmoid layer and the BCELoss in one single class.

## nn.MarginRankingLoss

- Creates a criterion that measures the loss given inputs `x1`, `x2`, two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor `y` (containing 1 or -1).

## nn.HingeEmbeddingLoss

- Measures the loss given an input tensor `x` and a labels tensor `y` (containing 1 or -1).

## nn.MultiLabelMarginLoss

- Creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input `x` (a 2D mini-batch Tensor) and output `y` (which is a 2D Tensor of target class indices).

## nn.HuberLoss

- Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.

## nn.SmoothL1Loss

- Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.

## nn.SoftMarginLoss

- Creates a criterion that optimizes a two-class classification logistic loss between input tensor `x` and target tensor `y` (containing 1 or -1).

## nn.MultiLabelSoftMarginLoss

- Creates a criterion that optimizes a multi-label one-versus-all loss based on max-entropy, between input `x` and target `y` of size `(N,C)`.

## nn.CosineEmbeddingLoss

- Creates a criterion that measures the loss given input tensors `x1`, `x2` and a Tensor label `y` with values 1 or -1.

## nn.MultiMarginLoss

- Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input `x` (a 2D mini-batch Tensor) and output `y` (which is a 1D tensor of target class indices, `0 ≤ y ≤ x.size(1)−1`).

## nn.TripletMarginLoss

- Creates a criterion that measures the triplet loss given an input tensors `x1`, `x2`, `x3` and a margin with a value greater than `0`.

## nn.TripletMarginWithDistanceLoss

- Creates a criterion that measures the triplet loss given input tensors `a`, `p`, and `n` (representing anchor, positive, and negative examples, respectively), and a nonnegative, real-valued function ("distance function") used to compute the relationship between the anchor and positive example ("positive distance") and the anchor and negative example ("negative distance").
