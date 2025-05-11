from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from typing import Any
import time
import datetime

from utils import extract_python_code, run_python_code, check_solutions_match, save_solution

_ = load_dotenv()

class AIModelClient:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def generate_code(self, system_prompt: str, user_prompt: str) -> str:
        """Generate code based on a system and user prompt"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=messages,
        )

        return extract_python_code(completion.choices[0].message.content)
    
    def improve_code(self, system_prompt: str, user_prompt: str,
                     code: str, error_message: str) -> str:
        """Improve code based on error feedback."""
        error_prompt = f"""
        Your Python code has the following issue:
        {error_message}
        
        Please provide a new version of the Python code to fix this issue.
        Make sure your code PRINTS the final answer as the last output.
        Original question: {user_prompt}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": code},
            {"role": "user", "content": error_prompt}
        ]
        
        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=messages,
        )
        
        return extract_python_code(completion.choices[0].message.content)
    
    def verify_solution(self, output: str, expected: str) -> bool:
        """Use AI to verify if the solution matches the expected answer."""
        verification_prompt = f"""
        Given a math problem solution, check if the Python output matches the expected answer.
        
        Expected answer: {expected}
        Python output: {output}
        
        Evaluate if they are mathematically equivalent. Only respond with "True" if they are equivalent, or "False" if they are not.
        """

        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise math verification assistant. Only respond with True or False."},
                {"role": "user", "content": verification_prompt}
            ]
        )

        verification_result = completion.choices[0].message.content.strip()
        return "true" in verification_result.lower()
    
    def extract_problem_features(self, question: str, solution_code: str) -> dict[str, Any]:
        """Extract features from the question and solution code using the LLM."""
        analysis_prompt = f"""
        Analyze this math problem and solution code to extract features.
        
        Question: {question}
        
        Solution code:
        ```python
        {solution_code}
        ```

        Extract the following information in JSON format:
        1. general_type: General math category (e.g., algebra, geometry, calculus, probability, etc.)
        2. specific_topics: List of specific math topics involved (e.g., quadratic equations, triangles, derivatives, etc.)
        3. operations: List of mathematical operations used in solution steps
        4. theorems: List of theorems, formulas or principles applied
        5. difficulty_level: Estimated difficulty (easy, medium, hard)
        
        Return ONLY the JSON object without any additional text.
        """
        messages = [
            {"role": "system", "content":"You are a math analysis assistant that extracts structured features from math problems and solutions."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=messages,
        )

        try:
            result_text = completion.choices[0].message.content.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            result = json.loads(result_text.strip())
            return result
        except:
            return {
                "general_type": "unknown",
                "specific_topics": [],
                "operations": [],
                "theorems": [],
                "difficulty_level": "unknown"
            }
    
class MathProblemSolver:
    """Class to solve math problems using AI-generated code"""

    def __init__(self, model_client: AIModelClient):
        self.model_client = model_client
        self.system_prompt = """
        You are a math expert. Firstly, given a math problem, you will divide the problem into smaller sub-problems. Then, for each sub-problem, you will build corresponding Python code to solve the sub-problem. Finally, you will combine all the sub-problem solutions into a Python file with multiple functions. 

        IMPORTANT: Make sure your code prints the final answer as the last output.

        Only return the Python code in a markdown code block with ```python tag. Do not include any other text.
        """
        self.max_attempts = 5

    def solve_problem(self, problem: dict[str, Any]) -> tuple[bool, str, str]:
        """Solve a math problem and return success status, solution, and message."""
        question = problem['question']
        expected_answer = str(problem['answer']).strip()

        # Generate initial solution
        code = self.model_client.generate_code(self.system_prompt, question)
        print(f"Generated code:\n{code}\n")
        
        # Test the solution
        success, message, code = self._test_and_improve(code, question, expected_answer)
        
        # Save final solution
        # save_solution(code)
        
        return success, code, message
    
    def _test_and_improve(self, code: str, question: str, expected_answer: str) -> tuple[bool, str, str]:
        """Test the solution and improve it if needed."""
        attempt = 1
        success = False
        message = ""
        
        while attempt <= self.max_attempts and not success:
            # Run the code
            run_success, output, error = run_python_code(code)
            
            if not run_success:
                print(f"Error running code (Attempt {attempt}/{self.max_attempts}):\n{error}")
                message = error
            else:
                print(f"Output: {output}")
                print(f"Expected: {expected_answer}")
                
                # Check if solutions match
                match_success, match_message = check_solutions_match(output, expected_answer)
                
                if match_success:
                    print(match_message)
                    return True, match_message, code
                
                # AI verification as last resort
                ai_verification = self.model_client.verify_solution(output, expected_answer)
                
                if ai_verification:
                    print("AI verified - Solution is correct")
                    return True, "AI verified - Solution is correct", code
                
                message = f"Expected {expected_answer}, but got {output}"
                print(f"Solutions don't match (Attempt {attempt}/{self.max_attempts}): {message}")
            
            # Improve the code if we have more attempts left
            if attempt < self.max_attempts:
                print(f"Attempting to improve solution (Attempt {attempt}/{self.max_attempts})")
                code = self.model_client.improve_code(self.system_prompt, question, code, message)
            
            attempt += 1
        
        return success, message, code  

class TrainingExampleManager:
    """Class to manage training examples, extract features, and save to JSON."""
    
    def __init__(self, model_client: AIModelClient):
        self.model_client = model_client
        self.output_file = "data/training_examples.json"
        self.checkpoint_file = "checkpoint.json"
        self.log_file = "error_log.txt"

    def create_training_examples(self, id: int, problem: dict[str, Any], solution_code: str) -> dict[str, Any]:
        """Create a training example with features extracted from the problem and solution."""
        # Extract features using the AI model
        features = self.model_client.extract_problem_features(problem['question'], solution_code)
        
        # Create the training example
        example = {
            "id": id,
            "question": problem['question'],
            "answer": problem['answer'],
            "modified_solution": solution_code,
            "features": features
        }
        
        return example
    
    def save_training_example(self, example: dict[str, Any], append: bool = True) -> None:
        """Save a training example to the JSON file."""
        if append and os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    examples = json.load(f)
            except:
                examples = []
        else:
            examples = []
        
        # Add the new example or update existing one with same id
        id_exists = False
        for i, ex in enumerate(examples):
            if ex.get('id') == example['id']:
                examples[i] = example
                id_exists = True
                break
        
        if not id_exists:
            examples.append(example)
        
        # Save the examples to the file
        with open(self.output_file, 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"Training example saved to {self.output_file}")

    def log_error(self, id: int, problem: dict[str, Any], error_message: str) -> None:
        """Log an error in processing a problem."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    errors = json.load(f)
            except:
                errors = []
        else:
            errors = []
        
        # Create error entry
        error_entry = {
            "id": id,
            "question": problem['question'],
            "answer": problem.get('answer', ''),
            "error": error_message,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add or update error entry
        id_exists = False
        for i, entry in enumerate(errors):
            if entry.get('id') == id:
                errors[i] = error_entry
                id_exists = True
                break
        
        if not id_exists:
            errors.append(error_entry)
        
        # Save errors to file
        with open(self.log_file, 'w') as f:
            json.dump(errors, f, indent=2)
        
        print(f"Error logged for problem {id} in {self.log_file}")

    def save_checkpoint(self, current_id: int, next_id: int) -> None:
        """Save the current processing checkpoint."""
        checkpoint = {
            "last_processed_id": current_id,
            "next_id": next_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Checkpoint saved: processed up to problem {current_id}, next is {next_id}")
    
    def load_checkpoint(self) -> tuple[int, int]:
        """Load the last processing checkpoint."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                last_id = checkpoint.get("last_processed_id", -1)
                next_id = checkpoint.get("next_id", 0)
                print(f"Loaded checkpoint: last processed {last_id}, next is {next_id}")
                return last_id, next_id
            except:
                print("Failed to load checkpoint, starting from the beginning")
        
        return -1, 0
    
    def get_failed_problems(self) -> list[int]:
        """Get a list of failed problem indices from the error log."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    errors = json.load(f)
                return [entry["id"] for entry in errors]
            except:
                pass
        
        return []
    
class BatchProcessor:
    """Class to process a batch of problems with checkpointing and error tracking."""
    
    def __init__(self, model_client: AIModelClient, solver: MathProblemSolver, example_manager: TrainingExampleManager):
        self.model_client = model_client
        self.solver = solver
        self.example_manager = example_manager
        self.data_file = "data/sample_metamath.json"
        self.max_problems = 1000
        self.retry_failed = True
        self.continue_from_checkpoint = True
    
    def load_problems(self) -> list[dict[str, Any]]:
        """Load problems from the data file."""
        with open(self.data_file, 'r') as f:
            return json.load(f)
    
    def process_batch(self) -> None:
        """Process a batch of problems with checkpointing and error tracking."""
        problems = self.load_problems()
        
        # Determine starting point
        last_processed, next_id = self.example_manager.load_checkpoint()
        
        # Decide on the problems to process
        indices_to_process = []
        
        # First, add failed problems if we're supposed to retry them
        if self.retry_failed:
            failed_indices = self.example_manager.get_failed_problems()
            indices_to_process.extend(failed_indices)
            print(f"Added {len(failed_indices)} failed problems to retry")
        
        # Then, continue from the checkpoint or start from the beginning
        if self.continue_from_checkpoint and next_id >= 0:
            # Add all indices from next_id to max_problems
            new_indices = list(range(next_id, min(len(problems), self.max_problems)))
            indices_to_process.extend(new_indices)
            print(f"Will continue from index {next_id}, adding {len(new_indices)} problems")
        else:
            # Start from the beginning
            indices_to_process.extend(range(min(len(problems), self.max_problems)))
            print(f"Starting from the beginning, processing {min(len(problems), self.max_problems)} problems")
        
        # Remove duplicates while preserving order
        seen = set()
        indices_to_process = [id for id in indices_to_process if not (id in seen or seen.add(id))]
        
        print(f"Processing {len(indices_to_process)} problems in total")
        
        # Process each problem
        for i, id in enumerate(indices_to_process):
            if id >= len(problems):
                print(f"Index {id} is out of range, skipping")
                continue
                
            problem = problems[id]
            print(f"\n[{i+1}/{len(indices_to_process)}] Processing problem {id}")
            
            try:
                # Solve the problem
                start_time = time.time()
                success, code, message = self.solver.solve_problem(problem)
                elapsed_time = time.time() - start_time
                
                print(f"Problem {id} processed in {elapsed_time:.2f} seconds: {'Success' if success else 'Failed'}")
                
                if success:
                    # Create and save training example
                    example = self.example_manager.create_training_examples(id, problem, code)
                    self.example_manager.save_training_example(example)
                else:
                    # Log the error
                    self.example_manager.log_error(id, problem, message)
                
                # Save checkpoint: current index and next index to process
                next_problem_id = max(id + 1, next_id) if id not in failed_indices else next_id
                self.example_manager.save_checkpoint(id, next_problem_id)
                
                # Small delay to avoid hitting API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing problem {id}: {str(e)}")
                self.example_manager.log_error(id, problem, str(e))
                # Don't update checkpoint on exception to allow retry
        
        print("\nBatch processing complete!")
        
        # Summary
        failed_indices = self.example_manager.get_failed_problems()
        print(f"Total problems processed: {len(indices_to_process)}")
        print(f"Failed problems: {len(failed_indices)}")
        if failed_indices:
            print(f"Failed indices: {failed_indices}")



# Initialize AI client
# API_KEY = os.getenv("OPENAI_API_KEY")
# BASE_URL = "https://api.openai.com/v1"
# MODEL_NAME = "gpt-4o-mini"

# model_client = AIModelClient(
#     base_url=BASE_URL,
#     api_key=API_KEY,
#     model_name=MODEL_NAME
# )

# solver = MathProblemSolver(model_client)
# example_manager = TrainingExampleManager(model_client)
# processor = BatchProcessor(model_client, solver, example_manager)
# processor.process_batch()

