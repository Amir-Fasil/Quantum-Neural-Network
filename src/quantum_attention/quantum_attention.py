import argparse
from attention.trainer import train
from attention.inference import inference

def main(
        mode: str, dataset: str, model_name: str, saved_dir: str,
        num_qubits: int, num_layers: int, depth_ebd: int,
        depth_query: int, depth_key: int, depth_value: int,
        batch_size: int, num_epochs: int, learning_rate: float,
        using_validation: bool, text: str   
):
    if args.mode == 'train':
        print(f"Starting training on dataset from: {args.dataset}")
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
            using_validation=using_validation
        )
    elif args.mode == 'inference':
        # Example inference classes (update based on your dataset)
        classes = ["Negative", "Positive"] 
        
        prediction = inference(
            text=text,
            model_path=f"{saved_dir}/{model_name}.pt",
            vocab_path=f"{dataset}/vocab.txt",
            classes=classes,
            num_qubits=args.num_qubits,
            num_layers=args.num_layers,
            depth_ebd=args.depth_ebd,
            depth_query=args.depth_query,
            depth_key=args.depth_key,
            depth_value=args.depth_value
        )
        print(f"Prediction for '{args.text}': {prediction}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantum Self-Attention Neural Network (QSANN)")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help="Mode to run: train or inference")
    parser.add_argument('--dataset', type=str, required=True, help="Directory containing train.txt, test.txt, and vocab.txt")
    parser.add_argument('--model_name', type=str, default='qsann_model', help="Name for the saved model")
    parser.add_argument('--saved_dir', type=str, default='./models/', help="Directory to save the trained model")
    
    # Model Hyperparameters
    parser.add_argument('--num_qubits', type=int, default=4, help="Number of qubits")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of self-attention layers")
    parser.add_argument('--depth_ebd', type=int, default=1, help="Depth of embedding circuit")
    parser.add_argument('--depth_query', type=int, default=1, help="Depth of query circuit")
    parser.add_argument('--depth_key', type=int, default=1, help="Depth of key circuit")
    parser.add_argument('--depth_value', type=int, default=1, help="Depth of value circuit")
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--using_validation', action='store_true', help="Use validation dataset (dev.txt)")
    
    # Inference arguments
    parser.add_argument('--text', type=str, default="Good movie", help="Text to classify during inference")
    
    args = parser.parse_args()
    main(
        mode=args.mode,
        dataset=args.dataset,
        model_name=args.model_name,
        saved_dir=args.saved_dir,
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        depth_ebd=args.depth_ebd,
        depth_query=args.depth_query,
        depth_key=args.depth_key,
        depth_value=args.depth_value,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        using_validation=args.using_validation,
        text=args.text  
    )
