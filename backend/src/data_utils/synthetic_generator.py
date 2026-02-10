import numpy as np 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

SYNTHETIC_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# value to prevent overflow during concept generation
value_limit = 0.5 

class SyntheticDataGenerator:
    def __init__(self, seed:int = 42):
        self.rng = np.random.default_rng(seed)
        self.initialised: bool = False # initalised 
        self.split_series: list[list[float]] = [] # list of series per concept 
        self.final_series: list[float] = [] # list of all series
        # self.time_index = 1 # global time step
        self.concept_counter: int = 0 # concept counter

    # update hte sliding window with the new observation by removing the first observation and adding the new observation to the end of the window
    def update_window(self, value: float) -> None:
        self.observations[:-1] = self.observations[1:]
        self.observations[-1] = value
        
    
    # Prepare the sliding window before generating the series . If initial window is provided, use it. Otherwise, use the last p observations of the previous concept series as the initial window for the next concept series.
    def check_inputs(self, parameters: list[float], initial_window: np.ndarray | None = None) -> None:
        p = len(parameters)
        if not self.initialised:
            if initial_window is None:
                raise ValueError("Initial observation valuese must be provided for the first concept")
            if len(initial_window) != p:
                raise ValueError(f"Initial observation values do not match the order p={p}.")
            self.observations = initial_window.copy()
            self.initialised = True
        else:
            # get the last concept generated and use the last p observations as the intial window for the next concept 
            last_concept = self.split_series[-1]
            self.observations = np.array(last_concept[-p:])

    
    # Linear Autoregressive (AR) model of order p:
    # Computes the next value in the time series as a weighted sum of the last p observations plus Gaussian white noise.
    # Formula:
    #   x_t = sum_{i=1}^{p} (phi_i * x_{t-i}) + w_t
    
    def linear_ar_formula(self, parameters: list[float], noise_variance: float) -> float:
        x = 0 
        x= np.dot(parameters, self.observations) 
        w = self.rng.normal(0, noise_variance)
        
        return x + w 
    
    
    # Formula for Non-linear model belongs to smooth transtion autoregressive (STAR) models, which are a class of non-linear time series models that allow for smooth transitions between different regimes or states. The non-linear factor is defined as 1.0 / (1.0 - np.exp(-10.0 * self.observations[0])), which introduces non-linearity into the model based on the value of the first observation in the sliding window. The noise is added to introduce randomness into the series, and the clipping mechanism ensures that the generated values do not deviate too much from the previous observation, keeping the series bounded.
    def nonlinear_ar_formula(self, parameters: list[float], noise_variance: float) -> float:
        linear_combination = np.dot(parameters, self.observations)
    
        nonlinear_factor = 1.0 / (1.0 - np.exp(-10.0 * self.observations[0]))
    
        noise = self.rng.normal(0, noise_variance)

        # apply non linear transformation to the linear combination and add noise
        x = (linear_combination * nonlinear_factor) + noise
    
        last_obs = self.observations[-1]
    
        # Clip large jumps to keep the series bounded
        if x - last_obs >= value_limit:
            x = last_obs - noise
        elif x - last_obs <= -value_limit:
            x = -last_obs + noise
    
        return float(x)
    
    
    
     # Formula for Linear AR model 
    # generate one time step of a linear AR model 
    def ar_model(self, parameters: list[float], noise_variance: float, concept_length: int, initial_window: np.ndarray | None = None) -> None:
        self.check_inputs(parameters, initial_window)
        series = []
        for _ in range(concept_length):
            # generate one time step of series using the linear AR formula and add it to the series list
            x = self.linear_ar_formula(parameters, noise_variance)
            series.append(x)
            # update the sliding window with the new observation
            self.update_window(x)
        # add hte generated series to the split series list    
        self.split_series.append(series)
        
            
    
    
    # generate one time step of a non-linear AR model 
    def non_linear_ar_model(self, parameters: list[float], noise_variance: float, concept_length: int, initial_window: np.ndarray | None = None) -> None:
        self.check_inputs(parameters, initial_window)
        series = []
        
        for _ in range(concept_length):
            # generate one time step of series using the non-linear AR formula and add it to the series list
            x = self.nonlinear_ar_formula(parameters, noise_variance)
            series.append(x)
            # update the sliding window with the new observation
            self.update_window(x)
        self.split_series.append(series)
    
    
        
    def series_linear_gradual_drift(self, concept_length: int, plot_series: bool = False)-> tuple[str, str, list[float]]:
        initial_window = self.rng.uniform(0,0.5, size=4)
        noise_variance = 0.02 
        
        self.ar_model(parameters=[0.006607488803146307, -0.2529881594354167, 0.8552562304577513, 0.3905674250981309], noise_variance=noise_variance, concept_length=concept_length, initial_window=initial_window)
        self.ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.7151247572018152], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.6335599337412476, 0.334852076313965, 1.3595185287185048, -0.07363691675509275], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.44071503228306824, 0.07407529241129636, 1.2573688275751191, 0.1076310711909298], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.6335599337412476, 0.334852076313965, 1.3595185287185048, -0.07363691675509275], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.7151247572018152], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        
        series = self.get_series()
        series_name = "linear_gradual_drift"
        output_folder = "linear_gradual_drift"
        
        if plot_series:
            self.plot()
            
        return output_folder, series_name, series
        
    
    def series_linear_abrupt_drift(self, concept_length: int, plot_series: bool = False) -> tuple[str, str, list[float]]:
        initial_window = self.rng.uniform(0, 0.5, size=4)
        noise_variance = 0.02
        
        self.ar_model(parameters=[0.14876092573738822, 0.05087244788237593, 0.4330193805067835, 0.3667339588762431], noise_variance=noise_variance, concept_length=concept_length, initial_window=initial_window)
        self.ar_model(parameters=[-0.318229212036593, 0.4133521130815502, 1.14841972367221, -0.24486472090297637], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.026851477518947557, 0.22016898814054223, -0.03814933593273199, 0.8447046999175475], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.4785914515620302, 0.8558602481837317, 0.024539136949191378, 0.5980008075169353], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.026851477518947557, 0.22016898814054223, -0.03814933593273199, 0.8447046999175475], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], noise_variance=noise_variance, concept_length=concept_length)
        self.ar_model(parameters=[-0.318229212036593, 0.4133521130815502, 1.14841972367221, -0.24486472090297637], noise_variance=noise_variance, concept_length=concept_length)
        
        series = self.get_series()
        series_name = "linear_abrupt_drift" 
        output_folder = "linear_abrupt_drift"
        
        if plot_series:
            self.plot()
        
        return output_folder, series_name, series
    
    def series_nonlinear_gradual_drift(self, concept_length: int, plot_series: bool = False) -> tuple[str, str, list[float]]:
        initial_window = self.rng.uniform(0, 0.5, size=4)
        noise_variance = 0.02
        
        self.non_linear_ar_model(parameters=[0.0203825939140348, 0.14856960377126693, 0.12154840218302701, 0.6913077037309644], noise_variance=noise_variance, concept_length=concept_length, initial_window=initial_window)
        self.non_linear_ar_model(parameters=[0.21432208811179806, 0.1747177586312132, 0.25627781880181116, 0.34924372007037097], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.6747722679575656, 0.0400499490190765, 0.12859434021708172, 0.1411115580708043], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.2592127990366997, 0.18679044833178132, 0.2510160243812225, 0.29144511960870556], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.715124757201], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.1779748207049134, -0.09139762327444532, 0.3628849251594744, 0.5451838112044337], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.715124757201], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.2592127990366997, 0.18679044833178132, 0.2510160243812225, 0.29144511960870556], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.6747722679575656, 0.0400499490190765, 0.12859434021708172, 0.1411115580708043], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.21432208811179806, 0.1747177586312132, 0.25627781880181116, 0.34924372007037097], noise_variance=noise_variance, concept_length=concept_length)
        
        
        series = self.get_series()
        series_name = "nonlinear_gradual_drift" 
        output_folder = "nonlinear_gradual_drift"
        
        if plot_series:
            self.plot()
            
        return output_folder, series_name, series
    
    def series_nonlinear_abrupt_drift(self, concept_length: int , plot_series: bool = False) -> tuple[str, str, list[float]]:
        initial_window = self.rng.uniform(0, 0.5, size=4)
        noise_variance = 0.02
        
        self.non_linear_ar_model(parameters=[-0.06658679980732536, 0.23421353635081468, 0.15495114023325046, 0.6768101219541569], noise_variance=noise_variance, concept_length=concept_length, initial_window=initial_window)
        self.non_linear_ar_model(parameters=[-0.506870130353138, 0.2589111765633722, 1.3970013340136547, -0.14964763809967313], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.4387715888915295, 0.3747437070432394, 1.3330941335780706, -0.26908562619916504], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.06975366774909564, -0.05196107339800573, 0.6352865482608727, 0.3344985733604905], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.2763541765783329, 0.3343598857377247, 0.4102952504128611, 0.5315753100371876], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[0.06975366774909564, -0.05196107339800573, 0.6352865482608727, 0.3344985733604905], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.506870130353138, 0.2589111765633722, 1.3970013340136547, -0.14964763809967313], noise_variance=noise_variance, concept_length=concept_length)
        self.non_linear_ar_model(parameters=[-0.06658679980732536, 0.23421353635081468, 0.15495114023325046, 0.6768101219541569], noise_variance=noise_variance, concept_length=concept_length)
        
        series = self.get_series()
        series_name = "nonlinear_abrupt_drift" 
        output_folder = "nonlinear_abrupt_drift"
        
        if plot_series:
            self.plot()
            
        return output_folder, series_name, series
    
    
    
    def get_series(self) -> list[float]:
        
        self.concept_counter = len(self.split_series)
         
        # loop over each concept series and add the length of each concept series to the series length 
        series_length = sum(len(i) for i in self.split_series)
            
        # create a list of zeros with the total length of all concept series
        series = [0] * series_length
        
        idx = 0 
        for concept in self.split_series:
            series[idx:idx+len(concept)] = concept
            idx += len(concept)
            
        self.final_series = series
        return series
 
        
    
    # method to write series to csv file
    def write_series_csv(self, output_folder: str, series_name: str, series: list[float]) -> None:
        output_dir = SYNTHETIC_DATA_DIR / output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{series_name}.csv"
        
        df = pd.DataFrame({"series": series})
        df.to_csv(output_path, header=False, index=False, float_format='%.3f')
        del df
        
    def plot(self, series: list[float] | np.ndarray = None, detections: list[float] | np.ndarray = None) -> None:
        if series is None:
            plt.plot(self.final_series, label = "Series", color = "blue")
            # add vertical lines ad boundary between concepts 
            boundary = 0
            for concept in self.split_series:
                boundary += len(concept)
                plt.axvline(boundary, linewidth=1, color='red', linestyle='--', zorder=-1)
                    
            plt.title(f"Series with {self.concept_counter} concepts")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return          
        
        else: 
            plt.plot(series, label = "Series", color = "blue")
            
            if detections is not None:
                for d in detections: 
                    plt.axvline(d, linewidth=1, color='red', linestyle='--', zorder=-1)
                
            plt.title("Manual Series with detected drifts")
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        


def generate_series(series_type: int, concept_length:int, num_series: int, plot_series: bool = False) -> None:
    
    for i in range(num_series):
        generator = SyntheticDataGenerator(seed=42+i) # use different seed for each series to generate different series for the same concept type
        
        print(str(i+1))
        
        match series_type:
            case 4:
                print("Linear Gradual Drift")
                output_folder, series_name, series = generator.series_linear_gradual_drift(concept_length, plot_series)
            case 5:
                print("Linear Abrupt Drift")
                output_folder, series_name, series = generator.series_linear_abrupt_drift(concept_length, plot_series)
            case 6:
                print("Non-linear Gradual Drift")
                output_folder, series_name, series = generator.series_nonlinear_gradual_drift(concept_length, plot_series)
            case 7:
                print("Non-linear Abrupt Drift")
                output_folder, series_name, series = generator.series_nonlinear_abrupt_drift(concept_length, plot_series)
            case _:
                raise ValueError("Invalid series type. Must be between 4 and 7.")
        
        generator.write_series_csv(output_folder, f"{series_name}{i+1}", series)

    


def main():
    # number of consecutive time steps generated by one fixed data generating process (concept) before switching to another one
    concept_length = 2000
    # number of csv file to generate for each ceoncept 
    num_series = 30
    plot_series = False
    
    for i in range(4,8):
        generate_series(i, concept_length, num_series, plot_series)
    
    # generate_series(4, concept_length, num_series, plot_series)

if __name__ == "__main__":
    main()
