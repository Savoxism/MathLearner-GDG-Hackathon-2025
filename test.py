from learning_module import AIModelClient, MathProblemSolver, TrainingExampleManager, BatchProcessor
import os
import json
from dotenv import load_dotenv
from typing import Set

def get_existing_ids() -> Set[int]:
    """Get set of existing IDs from training_examples.json"""
    try:
        with open('training_examples.json', 'r') as f:
            data = json.load(f)
            return {entry['id'] for entry in data}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def get_problems_to_process(start_id: int, end_id: int, existing_ids: Set[int]) -> list[int]:
    """Get list of IDs that need to be processed"""
    all_ids = set(range(start_id, end_id + 1))
    return sorted(list(all_ids - existing_ids))

def main():
    # Load environment variables
    _ = load_dotenv()
    
    # Configuration
    START_ID = 500
    END_ID = 1000
    
    # Get existing IDs
    existing_ids = get_existing_ids()
    ids_to_process = get_problems_to_process(START_ID, END_ID, existing_ids)
    
    if not ids_to_process:
        print(f"All IDs from {START_ID} to {END_ID} have already been processed!")
        return
    
    print(f"Found {len(existing_ids)} existing examples")
    print(f"Need to process {len(ids_to_process)} new examples")
    print(f"Starting with ID: {ids_to_process[0]}")
    
    # Initialize AI client
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "qwen/qwen-2.5-coder-32b-instruct:free"  # or your preferred model
    
    model_client = AIModelClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )
    
    # Create solver and example manager
    solver = MathProblemSolver(model_client)
    example_manager = TrainingExampleManager(model_client)
    
    # Load all problems
    with open("data/sample_metamath.json", 'r') as f:
        all_problems = json.load(f)
    
    # Process each problem that needs to be generated
    for current_id in ids_to_process:
        print(f"\nProcessing ID {current_id}")
        
        if current_id >= len(all_problems):
            print(f"Warning: ID {current_id} is out of range of available problems")
            continue
        
        problem = all_problems[current_id]
        
        try:
            # Solve the problem
            success, code, message = solver.solve_problem(problem)
            
            if success:
                # Create and save training example
                example = example_manager.create_training_examples(current_id, problem, code)
                
                # Rename id to id in the example
                example['id'] = example.pop('id')
                
                # Load existing data
                try:
                    with open('training_examples.json', 'r') as f:
                        all_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_data = []
                
                # Add new example
                all_data.append(example)
                
                # Sort by id
                all_data = sorted(all_data, key=lambda x: x['id'])
                
                # Save back to file
                with open('training_examples.json', 'w') as f:
                    json.dump(all_data, f, indent=2)
                
                print(f"Successfully processed and saved example {current_id}")
            else:
                print(f"Failed to solve problem {current_id}: {message}")
                example_manager.log_error(current_id, problem, message)
            
        except Exception as e:
            print(f"Error processing problem {current_id}: {str(e)}")
            example_manager.log_error(current_id, problem, str(e))
        
        # Save checkpoint
        example_manager.save_checkpoint(current_id, current_id + 1)
        
        # Small delay to avoid hitting API rate limits
        import time
        time.sleep(0.5)
    
    print("\nProcessing complete!")
    print(f"Processed {len(ids_to_process)} new examples")

if __name__ == "__main__":
    main()