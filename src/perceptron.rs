pub struct Instance {
    pub label: f64,
    pub features: Vec<f64>,
}

pub fn encode_nucleotide(nuc: &str) -> [f64; 4] {
    match nuc.to_lowercase().as_str() {
        "a" => [1.0, 0.0, 0.0, 0.0],
        "g" => [0.0, 1.0, 0.0, 0.0],
        "t" => [0.0, 0.0, 1.0, 0.0],
        "c" => [0.0, 0.0, 0.0, 1.0],
        _ => [0.0, 0.0, 0.0, 0.0],
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Model {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    pub fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let dot: f64 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        Self::sigmoid(dot + self.bias)
    }

    pub fn train(&mut self, dataset: &[Instance], epochs: usize, learning_rate: f64) {
        let m = dataset.len() as f64;
        for epoch in 0..epochs {
            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;

            for instance in dataset {
                let y_pred = self.predict(&instance.features);
                let error = y_pred - instance.label;

                for i in 0..self.weights.len() {
                    grad_w[i] += error * instance.features[i];
                }
                grad_b += error;
            }
            for i in 0..self.weights.len() {
                self.weights[i] -= learning_rate * grad_w[i] / m;
            }
            self.bias -= learning_rate * grad_b / m;

            if epoch % 100 == 0 {
                let total_error: f64 = dataset
                    .iter()
                    .map(|inst| (self.predict(&inst.features) - inst.label).abs())
                    .sum();
                println!("Epoch {}: Erreur moyenne = {:.4}", epoch, total_error / m);
            }
        }
    }
}
