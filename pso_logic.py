import numpy as np
import joblib

class PSOOptimizer:
    def __init__(self, model_path, bounds_min, bounds_max, feature_names=None):
        # Memuat model Random Forest yang kamu buat di Colab
        self.model = joblib.load(model_path)
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.dim = len(bounds_min)
        self.feature_names = feature_names

    def optimize(self, n_particles=30, n_iterations=50):
        # Parameter PSO standar (Inersia, Kognitif, Sosial)
        w, c1, c2 = 0.7, 1.5, 1.5
        
        # Inisialisasi posisi dan kecepatan partikel
        X = np.random.uniform(self.bounds_min, self.bounds_max, (n_particles, self.dim))
        V = np.random.uniform(-1, 1, (n_particles, self.dim))
        
        pbest_pos = X.copy()
        pbest_val = np.full(n_particles, np.inf)
        gbest_pos = np.zeros(self.dim)
        gbest_val = np.inf
        history = []

        for _ in range(n_iterations):
            # Meminta model memprediksi kadar silika di posisi partikel saat ini
            if self.feature_names is not None:
                import pandas as pd
                input_df = pd.DataFrame(X, columns=self.feature_names)
                fitness = self.model.predict(input_df)
            else:
                fitness = self.model.predict(X)
            
            # Update Personal Best (PBest)
            better_mask = fitness < pbest_val
            pbest_pos[better_mask] = X[better_mask]
            pbest_val[better_mask] = fitness[better_mask]
            
            # Update Global Best (GBest)
            if np.min(pbest_val) < gbest_val:
                gbest_val = np.min(pbest_val)
                gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()
            
            history.append(gbest_val)
            
            # Update Kecepatan dan Posisi untuk iterasi berikutnya
            r1, r2 = np.random.rand(n_particles, self.dim), np.random.rand(n_particles, self.dim)
            V = w * V + c1 * r1 * (pbest_pos - X) + c2 * r2 * (gbest_pos - X)
            X = np.clip(X + V, self.bounds_min, self.bounds_max)
            
        return gbest_pos, gbest_val, history