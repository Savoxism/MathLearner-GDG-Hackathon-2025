from learning_module import AIModelClient, MathProblemSolver, TrainingExampleManager
import os
import json
from dotenv import load_dotenv

_ = load_dotenv()


START_ID = 560
END_ID = 1000

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o-mini"

model_client = AIModelClient(
    base_url=BASE_URL,
    api_key=API_KEY,
    model_name=MODEL_NAME
)

with open('data/training_examples.json', 'r') as f:
    data = json.load(f)

existing_ids = {entry['id'] for entry in data}

all_ids = set(range(START_ID, END_ID + 1))
ids_to_process = list(all_ids - existing_ids)


print(f"Found {len(existing_ids)} existing examples")
print(f"Need to process {len(ids_to_process)} new examples")
print(f"Starting with ID: {ids_to_process[0]}")


solver = MathProblemSolver(model_client)
example_manager = TrainingExampleManager(model_client)


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
                with open('data/training_examples.json', 'r') as f:
                    all_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_data = []
            
            # Add new example
            all_data.append(example)
            
            # Sort by id
            all_data = sorted(all_data, key=lambda x: x['id'])
            
            # Save back to file
            with open('data/training_examples.json', 'w') as f:
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

print("\nProcessing complete!")
print(f"Processed {len(ids_to_process)} new examples")
