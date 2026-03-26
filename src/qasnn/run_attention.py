from typing import Optional
from qasnn.attention.trainer import train
from qasnn.attention.inference import inference

def run_q_attention(
        mode: str, dataset: str, model_name: str, saved_dir: str,
        num_qubits: int, num_layers: int, depth_ebd: int,
        depth_query: int, depth_key: int, depth_value: int,
        batch_size: int, num_epochs: int, learning_rate: float,
        using_validation: bool, text: str,
        metrics_log_path: Optional[str] = None,
):
    if mode == 'train':
        print(f"Starting training on dataset from: {dataset}")
        train(
            model_name=model_name,
            dataset=dataset,
            num_qubits=num_qubits,
            num_layers=num_layers,
            depth_ebd=depth_ebd,
            depth_query=depth_query,
            depth_key=depth_key,
            depth_value=depth_value,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            saved_dir=saved_dir,
            using_validation=using_validation,
            metrics_log_path=metrics_log_path,
        )
    elif mode == 'inference':
        # Example inference classes (update based on your dataset)
        classes = ["Negative", "Positive"] 
        
        prediction = inference(
            text=text,
            model_path=f"{saved_dir}/{model_name}.pt",
            vocab_path=f"{dataset}/vocab.txt",
            classes=classes,
            num_qubits=num_qubits,
            num_layers=num_layers,
            depth_ebd=depth_ebd,
            depth_query=depth_query,
            depth_key=depth_key,
            depth_value=depth_value
        )
        print(f"Prediction for '{text}': {prediction}")
