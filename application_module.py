import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from utils import call_llm_api, extract_python_code, execute_python_code, find_similar_problems, read_prompt_template
from embedding import EmbeddingGenerator

class ApplicationModule:
    def __init__(self, 
                 vector_db_path: str = "data/vector_db.pkl",
                 model: str = "qwen/qwen3-32b:free",
                 k_neighbors: int = 3):
        """
        Initialize the Application Module.
        
        Args:
            vector_db_path: Path to the vector database
            model: LLM model to use
            k_neighbors: Number of similar problems to retrieve
        """
        self.vector_db_path = vector_db_path
        self.model = model
        self.k_neighbors = k_neighbors
        self.embedding_generator = EmbeddingGenerator()
        
        # Check if vector database exists
        if not os.path.exists(vector_db_path):
            print(f"WARNING: Vector database not found at {vector_db_path}")
    
    def extract_features(self, problem: str) -> Dict[str, Any]:
        """
        Extract features from a new problem.
        
        Args:
            problem: Problem text
            
        Returns:
            Dictionary of problem features
        """
        # Read extract features prompt template
        prompt_template = read_prompt_template("extract_features")
        
        # Format the prompt with the problem
        prompt = prompt_template.format(problem=problem)
        
        # Call LLM to extract features
        response = call_llm_api(prompt, self.model)
        
        # Parse features from response
        features = {
            "problem_type": "",
            "solution_steps": [],
            "used_techniques": [],
            "theorems_used": [],
            "problem_description": problem
        }
        
        if "## PROBLEM FEATURES" in response:
            features_section = response.split("## PROBLEM FEATURES")[1].strip()
            
            # Extract problem type
            if "Problem type:" in features_section:
                problem_type_line = features_section.split("Problem type:")[1].split("\n")[0].strip()
                features["problem_type"] = problem_type_line
            
            # Extract solution steps
            if "Possible solution steps:" in features_section:
                steps_section = features_section.split("Possible solution steps:")[1]
                if "Techniques" in steps_section:
                    steps_section = steps_section.split("Techniques")[0]
                
                steps = []
                for line in steps_section.strip().split("\n"):
                    if line.strip() and any(line.strip().startswith(str(i) + ".") for i in range(1, 20)):
                        step_text = line.strip().split(".", 1)[1].strip()
                        steps.append(step_text)
                
                features["solution_steps"] = steps
            
            # Extract techniques
            if "Techniques that could be used:" in features_section:
                techniques_section = features_section.split("Techniques that could be used:")[1]
                if "Theorems" in techniques_section:
                    techniques_section = techniques_section.split("Theorems")[0]
                
                techniques = []
                for line in techniques_section.strip().split("\n"):
                    if line.strip() and line.strip().startswith("-"):
                        technique = line.strip()[1:].strip()
                        if technique:
                            techniques.append(technique)
                
                features["used_techniques"] = techniques
            
            # Extract theorems
            if "Theorems that could be applied:" in features_section:
                theorems_section = features_section.split("Theorems that could be applied:")[1]
                
                theorems = []
                for line in theorems_section.strip().split("\n"):
                    if line.strip() and line.strip().startswith("-"):
                        theorem = line.strip()[1:].strip()
                        if theorem:
                            theorems.append(theorem)
                
                features["theorems_used"] = theorems
        
        return features
    
    def find_similar_solutions(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find similar solutions from the vector database.
        
        Args:
            features: Problem features
            
        Returns:
            List of similar problems with their features
        """
        # Generate embedding for the features
        query_embedding = self.embedding_generator.generate_features_embedding(features)
        
        # Search for similar problems
        similar_problems = find_similar_problems(
            query_embedding, 
            self.vector_db_path, 
            k=self.k_neighbors
        )
        
        return similar_problems
    
    def generate_solution(self, problem: str, similar_problems: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, bool, Optional[str]]:
        """
        Generate solution for a new problem.
        
        Args:
            problem: Problem text
            similar_problems: Optional list of similar problems with their features
            
        Returns:
            Tuple of (Python code, success flag, error message)
        """
        if similar_problems and len(similar_problems) > 0:
            # Generate solution with similar problems as reference
            return self._generate_solution_with_reference(problem, similar_problems)
        else:
            # Generate solution without reference
            return self._generate_solution_without_reference(problem)
    
    def _generate_solution_with_reference(self, problem: str, similar_problems: List[Dict[str, Any]]) -> Tuple[str, bool, Optional[str]]:
        """Generate solution using similar problems as reference."""
        # Construct context from similar problems
        context = "Below are some similar problems and their features:\n\n"
        
        for i, sim_prob in enumerate(similar_problems):
            context += f"Similar problem {i+1} (Similarity: {sim_prob['similarity']:.2f}):\n"
            features = sim_prob["features"]
            
            if "problem_description" in features:
                context += f"Description: {features['problem_description']}\n"
            
            if "problem_type" in features:
                context += f"Problem type: {features['problem_type']}\n"
            
            if "solution_steps" in features and features["solution_steps"]:
                context += "Solution steps:\n"
                for j, step in enumerate(features["solution_steps"]):
                    context += f"  {j+1}. {step}\n"
            
            if "used_techniques" in features and features["used_techniques"]:
                context += "Techniques used:\n"
                for technique in features["used_techniques"]:
                    context += f"  - {technique}\n"
            
            if "theorems_used" in features and features["theorems_used"]:
                context += "Theorems used:\n"
                for theorem in features["theorems_used"]:
                    context += f"  - {theorem}\n"
            
            context += "\n"
        
        # Read prompt template
        prompt_template = read_prompt_template("generate_solution_with_reference")
        
        # Format the prompt
        prompt = prompt_template.format(
            problem=problem,
            context=context
        )
        
        # Call LLM to generate solution
        response = call_llm_api(prompt, self.model)
        code = extract_python_code(response)
        
        # Verify the solution
        success, result, error_msg = execute_python_code(code)
        
        # If failed, try to fix the code once
        if not success:
            # Read fix code prompt template
            fix_prompt_template = read_prompt_template("fix_code")
            
            # Format the fix prompt
            fix_prompt = fix_prompt_template.format(
                problem=problem,
                code=code,
                error_msg=error_msg
            )
            
            # Get fixed code from LLM
            response = call_llm_api(fix_prompt, self.model)
            fixed_code = extract_python_code(response)
            
            # Try to execute fixed code
            success, result, error_msg = execute_python_code(fixed_code)
            
            if success:
                code = fixed_code
                error_msg = None
        
        return code, success, error_msg
    
    def _generate_solution_without_reference(self, problem: str) -> Tuple[str, bool, Optional[str]]:
        """Generate solution without reference to similar problems."""
        # Read prompt template
        prompt_template = read_prompt_template("generate_solution_without_reference")
        
        # Format the prompt
        prompt = prompt_template.format(problem=problem)
        
        # Call LLM to generate solution
        response = call_llm_api(prompt, self.model)
        code = extract_python_code(response)
        
        # Verify the solution
        success, result, error_msg = execute_python_code(code)
        
        # If failed, try to fix the code once
        if not success:
            # Read fix code prompt template
            fix_prompt_template = read_prompt_template("fix_code")
            
            # Format the fix prompt
            fix_prompt = fix_prompt_template.format(
                problem=problem,
                code=code,
                error_msg=error_msg
            )
            
            # Get fixed code from LLM
            response = call_llm_api(fix_prompt, self.model)
            fixed_code = extract_python_code(response)
            
            # Try to execute fixed code
            success, result, error_msg = execute_python_code(fixed_code)
            
            if success:
                code = fixed_code
                error_msg = None
        
        return code, success, error_msg
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a new problem using the application module pipeline.
        
        Args:
            problem: Problem text
            
        Returns:
            Dictionary with solution results
        """
        # Step 1: Extract features from the problem
        print("Step 1: Extracting problem features...")
        features = self.extract_features(problem)
        
        # Step 2: Find similar problems
        print("Step 2: Finding similar problems...")
        similar_problems = self.find_similar_solutions(features)
        
        # Check if we found similar problems
        has_similar = len(similar_problems) > 0
        print(f"{'Found' if has_similar else 'Did not find'} similar problems.")
        
        # Step 3: Generate solution
        print("Step 3: Generating solution...")
        if has_similar:
            print(f"Generating solution based on {len(similar_problems)} similar problems.")
            code, success, error_msg = self.generate_solution(problem, similar_problems)
        else:
            print("Generating solution without reference problems.")
            code, success, error_msg = self.generate_solution(problem)
        
        # Return results
        result = {
            "problem": problem,
            "features": features,
            "similar_problems": similar_problems if has_similar else [],
            "has_similar_problems": has_similar,
            "code": code,
            "success": success
        }
        
        if error_msg:
            result["error"] = error_msg
        
        return result


if __name__ == "__main__":
    # Example usage
    app_module = ApplicationModule(
        vector_db_path="data/vector_db.pkl",
        k_neighbors=3
    )
    
    # Test problem
    test_problem = "Find the value of x that satisfies the equation x^2 - 5x + 6 = 0."
    
    # Solve the problem
    result = app_module.solve_problem(test_problem)
    
    # Print the result
    print("\nResults:")
    print(f"Problem: {result['problem']}")
    print(f"Found similar problems: {'Yes' if result['has_similar_problems'] else 'No'}")
    print(f"Solution generation successful: {'Yes' if result['success'] else 'No'}")
    
    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown')}")
    
    print("\nPython code:")
    print(result["code"]) 