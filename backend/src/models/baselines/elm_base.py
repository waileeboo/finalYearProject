import numpy as np 


class ELMBase: 
    def __init__(self, hidden_neurons: int, seed=42):
        self.hidden_neurons = hidden_neurons
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.input_weights: np.ndarray | None = None
        self.output_weights: np.ndarray | None = None 
    
    # add bias to each window of input data 
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([X, np.ones(X.shape[0])])
    
    # first by adding the bias then multiplying the input weights and then applying the activation function to get the hidden layer output and finally calculating the output weights using the pseudo-inverse of the hidden layer output and the target values
    def train(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> None:
        X_biased = self._add_bias(X)
        
        if weights is None:
            # create input weight of the shape (window + 1, hidden_neurons)(matrix)
            self.input_weights = self.rng.standard_normal((X_biased.shape[1], self.hidden_neurons)) 
        else:
            self.input_weights = weights 
            
            
        # tranform teh data to complex pattern
        #without tanh the model can only learn lenear relationship. H is a matrix 
        H = np.tanh(X_biased.dot(self.input_weights))
        
        # instead of doing backpropagation we can directly calculate the output weights using the pseudo-inverse of the hidden layer output and the target values. This is a key feature of ELMs that allows for fast training. H * B = y we are trying to find B (output weight). This calculation give weight to each neurons. we use pseudo inverse becasue H is not a square matrix and we want to find the best fit solution for the output weights. The pseudo-inverse allows us to find a solution even when H is not invertible or when there are more hidden neurons than training samples.
        self.output_weights = np.linalg.pinv(H).dot(y)
        
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_biased = self._add_bias(X)
        H = np.tanh(X_biased.dot(self.input_weights))
        return H.dot(self.output_weights)
    
    def predict_phase_adjusted(self, X: np.ndarray) -> np.ndarray:
        first_prediction = self.predict(X)
        adjusted_input = np.column_stack([X[:,1:], first_prediction])
        return self.predict(adjusted_input)
        
    
    
    
    