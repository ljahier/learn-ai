mod perceptron;
use perceptron::Model;

use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_dataset(path: &str) -> std::io::Result<Vec<perceptron::Instance>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() != 3 {
            println!(
                "Ligne {} incomplète ({} champs)",
                line_num + 1,
                fields.len()
            );
            continue;
        }
        let label = if fields[0] == "+" { 1.0 } else { 0.0 };
        let sequence_str = fields[2];
        if sequence_str.len() != 57 {
            println!(
                "Ligne {}: séquence de longueur {} (attendu 57)",
                line_num + 1,
                sequence_str.len()
            );
            continue;
        }
        let mut features = Vec::with_capacity(57 * 4);
        for c in sequence_str.chars() {
            let one_hot = perceptron::encode_nucleotide(&c.to_string());
            features.extend_from_slice(&one_hot);
        }
        dataset.push(perceptron::Instance { label, features });
    }
    Ok(dataset)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = load_dataset("promoters.data")?;
    println!("Dataset chargé avec {} instances", dataset.len());

    let mut model = Model::new(228);
    model.train(&dataset, 1000, 0.1);
    println!("Modèle entraîné : biais = {:.4}", model.bias);
    println!("Premiers poids : {:?}", &model.weights[..5]);

    for (i, instance) in dataset.iter().enumerate() {
        let y_pred = model.predict(&instance.features);
        println!(
            "Instance {}: label attendu = {:.0}, prédiction = {:.4}",
            i + 1,
            instance.label,
            y_pred
        );
    }

    Ok(())
}
