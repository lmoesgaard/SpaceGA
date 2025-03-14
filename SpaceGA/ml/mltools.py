import sys
import os
import numpy as np
import importlib
import joblib
import numpy as np
from sklearn.model_selection import train_test_split


def get_model(modelname):
    """
    import SciKit model from ml.models
    """
    module = importlib.import_module("SpaceGA.ml.models")
    try:
        model = getattr(module, modelname)()
        return model
    except:
        print(f"Could not find model: {modelname}")
        sys.exit()

class MLtools:
    def setup_ml(self):
        self.model_path = os.path.join(self.log, "model_r{}.pkl")
        model = get_model(self.model_name)
        joblib.dump(model, self.model_path.format(self.gen))

    def update_model(self, offspring):
        """Updates the ML model using new offspring data and saves it."""
        # Combine features and scores into a single array
        array = np.hstack([np.vstack(offspring["fp"].to_list()), offspring["scores"].to_numpy().reshape(-1, 1)])

        # Save new training data
        npz_path = os.path.join(self.train_dir, f'round{self.gen}.npz')
        np.savez_compressed(npz_path, array=array)

        # Load ML model
        model = joblib.load(self.model_path.format(self.gen - 1))

        # Load and concatenate all training data
        files = [os.path.join(self.train_dir, f'round{gen}.npz') for gen in range(1, self.gen + 1)]
        dataset = np.concatenate([np.load(f, allow_pickle=True)['array'] for f in files])

        # Split into features (X) and labels (y)
        X, y = dataset[:, :-1], dataset[:, -1].flatten()

        # Train model
        model.model.fit(X, y)

        # Save updated model
        joblib.dump(model, self.model_path.format(self.gen))

        return offspring.drop(columns=["fp"])
