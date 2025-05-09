import os
import argparse
import json
from typing import Dict, Any, Optional, List, Union
import numpy as np

from learning_module import LearningModule
from application_module import ApplicationModule


def setup_directories():
    """Set up necessary directories for the project."""
    os.makedirs("data", exist_ok=True)


def train_model(dataset_path: str, limit: Optional[int] = None):
    """Train the model using the Learning Module."""
    print("=== STARTING LEARNING PROCESS ===")
    learning_module = LearningModule(
        dataset_path=dataset_path,
        vector_db_path="data/vector_db.pkl"
    )
    
    learning_module.process_dataset(limit=limit)
    print("=== LEARNING PROCESS COMPLETED ===")


def solve_problem(problem: str) -> Dict[str, Any]:
    """Solve a problem using the Application Module."""
    print("=== STARTING PROBLEM SOLVING ===")
    app_module = ApplicationModule(
        vector_db_path="data/vector_db.pkl",
        k_neighbors=3
    )
    
    result = app_module.solve_problem(problem)
    print("=== PROBLEM SOLVING COMPLETED ===")
    
    return result


def evaluate_model(test_problems_path: str) -> dict[str, Any]:
    """Evaluate the model performance on test problems."""
    print("=== STARTING MODEL EVALUATION ===")
    
    # Load test problems
    with open(test_problems_path, "r") as f:
        test_problems = json.load(f)
    
    app_module = ApplicationModule(
        vector_db_path="data/vector_db.pkl",
        k_neighbors=3
    )
    
    results = []
    metrics: dict[str, Union[int, float]] = {
        "total": len(test_problems),
        "successful": 0,
        "with_similar_problems": 0,
        "successful_with_similar": 0,
        "successful_without_similar": 0,
        "global_accuracy": 0.0,
        "precision_accuracy": 0.0,
        "profitability": 0.0
    }
    
    for i, problem_data in enumerate(test_problems):
        problem = problem_data.get("problem", "")
        print(f"Evaluating problem {i+1}/{len(test_problems)}...")
        
        try:
            result = app_module.solve_problem(problem)
            results.append(result)
            
            # Update metrics
            if result["success"]:
                metrics["successful"] = metrics["successful"] + 1
            
            if result["has_similar_problems"]:
                metrics["with_similar_problems"] = metrics["with_similar_problems"] + 1
                if result["success"]:
                    metrics["successful_with_similar"] = metrics["successful_with_similar"] + 1
            elif result["success"]:
                metrics["successful_without_similar"] = metrics["successful_without_similar"] + 1
                
        except Exception as e:
            print(f"Error solving problem {i+1}: {str(e)}")
    
    # Calculate derived metrics
    if metrics["total"] > 0:
        metrics["global_accuracy"] = metrics["successful"] / metrics["total"]
    
    if metrics["with_similar_problems"] > 0:
        metrics["precision_accuracy"] = metrics["successful_with_similar"] / metrics["with_similar_problems"]
    
    # Calculate profitability/benefit
    non_similar_problems = metrics["total"] - metrics["with_similar_problems"]
    if non_similar_problems > 0:
        metrics["profitability"] = (
            metrics["precision_accuracy"] - metrics["successful_without_similar"] / non_similar_problems
        )
    
    print("=== MODEL EVALUATION COMPLETED ===")
    print(f"Global accuracy: {metrics['global_accuracy']:.2f}")
    print(f"Precision accuracy: {metrics['precision_accuracy']:.2f}")
    print(f"Profitability: {metrics['profitability']:.2f}")
    
    return {
        "metrics": metrics,
        "results": results
    }


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="MathLearner - Learn and solve math problems")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", required=True, help="Path to the dataset")
    train_parser.add_argument("--limit", type=int, help="Limit the number of problems to process")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a problem")
    solve_parser.add_argument("--problem", required=True, help="Problem to solve")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--test-problems", required=True, help="Path to test problems dataset")
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    if args.command == "train":
        train_model(args.dataset, args.limit)
    
    elif args.command == "solve":
        result = solve_problem(args.problem)
        
        # Print the result
        print("\nResults:")
        print(f"Problem: {result['problem']}")
        print(f"Found similar problems: {'Yes' if result['has_similar_problems'] else 'No'}")
        print(f"Solution generation successful: {'Yes' if result['success'] else 'No'}")
        
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown')}")
        
        print("\nPython code:")
        print(result["code"])
    
    elif args.command == "evaluate":
        evaluate_model(args.test_problems)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()