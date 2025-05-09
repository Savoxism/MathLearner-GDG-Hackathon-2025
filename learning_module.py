import os
import json
from re import T
from typing import Any, Optional
import numpy as np

from utils import load_math_dataset, call_llm_api, extract_python_code, execute_python_code, save_to_vector_db, read_prompt_template
from embedding import EmbeddingGenerator

class LearningModule:
    def __init__(self, 
                 dataset_path: str,
                 vector_db_path: str = "data/vector_db.pkl",
                 model: str = "qwen/qwen3-32b:free"):
        """
        Initialize the Learning Module.
        
        Args:
            dataset_path: Path to the MATH dataset
            vector_db_path: Path to save the vector database
            model: LLM model to use
        """
        self.dataset_path = dataset_path
        self.vector_db_path = vector_db_path
        self.model = model
        self.embedding_generator = EmbeddingGenerator()
        
        # Create directory for vector database if it doesn't exist
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    
    def generate_modified_solution(self, problem: dict[str, Any], solution: str) -> tuple[str, str, dict[str, Any]]:
        """
        Generate a modified solution for a math problem.
        
        Args:
            problem: Problem dictionary
            solution: Original solution
            
        Returns:
            Tuple of (standardized solution, Python code, problem features)
        """
        # Read prompt template
        prompt_template = read_prompt_template("generate_modified_solution")
        
        # Format the prompt with problem and solution
        prompt = prompt_template.format(
            problem=problem["problem"],
            solution=solution
        )
        
        # Call LLM to generate modified solution
        response = call_llm_api(prompt, self.model)
        
        # Extract parts from the response
        standardized_solution = ""
        python_code = ""
        features = {}
        
        # Parse standardized solution
        if "## STANDARDIZED SOLUTION" in response:
            start_idx = response.find("## STANDARDIZED SOLUTION") + len("## STANDARDIZED SOLUTION")
            end_idx = response.find("## PYTHON CODE", start_idx) if "## PYTHON CODE" in response else len(response)
            standardized_solution = response[start_idx:end_idx].strip()
        
        # Extract Python code
        if "## PYTHON CODE" in response:
            start_idx = response.find("## PYTHON CODE") + len("## PYTHON CODE")
            end_idx = response.find("## PROBLEM FEATURES", start_idx) if "## PROBLEM FEATURES" in response else len(response)
            code_section = response[start_idx:end_idx].strip()
            python_code = extract_python_code(code_section)
        
        # Parse features
        features = self._parse_features(response)
        
        return standardized_solution, python_code, features
    
    def _parse_features(self, response: str) -> dict[str, Any]:
        """Parse features from LLM response."""
        features = {
            "problem_type": "",
            "solution_steps": [],
            "used_techniques": [],
            "theorems_used": []
        }
        
        if "## PROBLEM FEATURES" in response:
            features_section = response.split("## PROBLEM FEATURES")[1].strip()
            
            # Extract problem type
            if "Problem type:" in features_section:
                problem_type_line = features_section.split("Problem type:")[1].split("\n")[0].strip()
                features["problem_type"] = problem_type_line
            
            # Extract solution steps
            if "Solution steps:" in features_section:
                steps_section = features_section.split("Solution steps:")[1]
                if "Techniques used:" in steps_section:
                    steps_section = steps_section.split("Techniques used:")[0]
                
                steps = []
                for line in steps_section.strip().split("\n"):
                    if line.strip() and any(line.strip().startswith(str(i) + ".") for i in range(1, 20)):
                        step_text = line.strip().split(".", 1)[1].strip()
                        steps.append(step_text)
                
                features["solution_steps"] = steps
            
            # Extract techniques
            if "Techniques used:" in features_section:
                techniques_section = features_section.split("Techniques used:")[1]
                if "Theorems used:" in techniques_section:
                    techniques_section = techniques_section.split("Theorems used:")[0]
                
                techniques = []
                for line in techniques_section.strip().split("\n"):
                    if line.strip() and line.strip().startswith("-"):
                        technique = line.strip()[1:].strip()
                        if technique:
                            techniques.append(technique)
                
                features["used_techniques"] = techniques
            
            # Extract theorems
            if "Theorems used:" in features_section:
                theorems_section = features_section.split("Theorems used:")[1]
                
                theorems = []
                for line in theorems_section.strip().split("\n"):
                    if line.strip() and line.strip().startswith("-"):
                        theorem = line.strip()[1:].strip()
                        if theorem:
                            theorems.append(theorem)
                
                features["theorems_used"] = theorems
        
        return features
    
    def verify_solution(self, python_code: str, problem: dict[str, Any], max_attempts: int = 3) -> tuple[bool, str, Optional[str]]:
        """
        Verify the solution by executing the Python code.
        
        Args:
            python_code: Python code to execute
            problem: Problem dictionary
            max_attempts: Maximum number of attempts to correct the code
            
        Returns:
            Tuple of (success, final_code, error_message)
        """
        # First attempt with original code
        success, result, error_msg = execute_python_code(python_code)
        
        current_code = python_code
        attempts = 0
        
        # If failed, try to fix the code
        while not success and attempts < max_attempts:
            attempts += 1
            
            # Read fix code prompt template
            fix_prompt_template = read_prompt_template("fix_code")
            
            # Format the fix prompt
            fix_prompt = fix_prompt_template.format(
                problem=problem["problem"],
                code=current_code,
                error_msg=error_msg
            )
            
            # Get fixed code from LLM
            response = call_llm_api(fix_prompt, self.model)
            fixed_code = extract_python_code(response)
            
            # Try to execute fixed code
            success, result, error_msg = execute_python_code(fixed_code)
            
            if success:
                current_code = fixed_code
                break
            else:
                current_code = fixed_code
        
        if success:
            return True, current_code, None
        else:
            return False, current_code, error_msg
    
    def process_dataset(self, limit: Optional[int] = None):
        """
        Process the dataset to generate modified solutions and store them in the vector database.
        
        Args:
            limit: Optional limit on the number of problems to process
        """
        # Load dataset
        problems = load_math_dataset(self.dataset_path)
        
        # Limit the number of problems if specified
        if limit is not None:
            problems = problems[:limit]
        
        print(f"Processing {len(problems)} problems...")
        
        successful_count = 0
        for i, problem_data in enumerate(problems):
            problem_id = str(i)
            problem = problem_data.get("problem", "")
            solution = problem_data.get("solution", "")
            
            print(f"Processing problem {i+1}/{len(problems)}...")
            
            try:
                # Generate modified solution
                standardized_solution, python_code, features = self.generate_modified_solution(
                    {"problem": problem}, solution
                )
                
                # Verify the solution
                success, verified_code, error_msg = self.verify_solution(python_code, {"problem": problem})
                
                if success:
                    # Add the original problem to features
                    features["problem_description"] = problem
                    
                    # Generate embedding for features
                    embedding = self.embedding_generator.generate_features_embedding(features)
                    
                    # Save to vector database
                    save_to_vector_db(problem_id, features, embedding, self.vector_db_path)
                    
                    successful_count += 1
                    print(f"✅ Problem {i+1} processed successfully")
                else:
                    print(f"❌ Problem {i+1} could not be verified: {error_msg}")
            
            except Exception as e:
                print(f"❌ Error processing problem {i+1}: {str(e)}")
        
        print(f"Complete! Successfully processed {successful_count}/{len(problems)} problems.")


if __name__ == "__main__":
    # Example usage
    learning_module = LearningModule(
        dataset_path="raw_train.json",
        vector_db_path="data/vector_db.pkl"
    )
    
    learning_module.process_dataset(limit=5)  # Process first 5 problems for testing 