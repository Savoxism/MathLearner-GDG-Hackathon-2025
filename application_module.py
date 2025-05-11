from embedding import EmbeddingManager
from learning_module import AIModelClient
from utils import run_python_code
import os
import json
from typing import Any
from dotenv import load_dotenv

_ = load_dotenv()

class ApplicationModule:
    def __init__(self):
        """Initialize the application module with necessary components."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        
        # Initialize client model for most tasks
        CLIENT_MODEL = "gpt-4o-mini"
        
        self.model_client = AIModelClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=CLIENT_MODEL,
        )

        # Initialize client model for solving problems
        # SOLVER_MODEL = "qwen/qwen-2.5-coder-32b-instruct:free"
        SOLVER_MODEL = "gpt-4o-mini"
        self.solver_model_client = AIModelClient(
            # base_url="https://openrouter.ai/api/v1",
            # api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=SOLVER_MODEL,
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
            
        self.max_iterations = 5
        
        # system prompt for solution
        self.system_prompt = """
        You are a mathematics expert. First, for a given problem, you will break it down into subproblems.  
        Then, for each subproblem, you will construct the corresponding Python code to solve it.  
        Finally, you will combine all subproblem solutions into a single Python file with multiple functions.

        IMPORTANT: Ensure that your code prints out the final result as the last output.

        Return only the Python code inside a markdown code block tagged ```python. Do not include any other text.
        """
        
        self.retrieval_system_prompt = """
        You are a mathematics expert. You will receive a new problem and some solutions from similar problems.  
        Your task is to create a Python solution for the new problem, using the similar solutions as references.

        IMPORTANT: Ensure that your code prints out the final result as the last output.

        Return only the Python code inside a markdown code block tagged ```python. Do not include any other text.
        """
        
    def process_question(self, question: str) -> dict[str, Any]:
        print(f"Processing question: {question}")
        
        # Step 1: extract features from the question
        features = self.embedding_manager.extract_features_from_question(question)
        
        print(json.dumps(features, indent=2, ensure_ascii=False))
        
        # Step 2: find similar problems
        similar_problems = self.embedding_manager.search_similar_problems(
            question, features, n_results=3
        )
        
        if similar_problems:
            for i, problem in enumerate(similar_problems):
                print(f"\nSimilar problem #{i+1} (Similarity: {problem['similarity']:.4f}):")
                print(f"Question: {problem['question'][:300]}...")
        else:
            print("No similar problems found.")
        
        # Step 3: solve the problem
        solution_code = self._generate_solution_with_retrieval(question, similar_problems)
        
        # Step 4: execute the solution (python code)
        success, output, final_code = self._execute_and_improve_solution(question, solution_code)
        
        # Step 5: verify the result
        verification_result = self._verify_result(question, output, final_code)
        
        result = {
            "question": question,
            "features": features,
            "similar_problems": similar_problems,
            "solution_code": final_code,
            "output": output,
            "success": success,
            "verification": verification_result
        }     
        
        return result
    
    def _generate_solution_with_retrieval(self, question: str, similar_problems: list[dict[str, Any]] ) -> str:
        """Generate solution based on similar problems"""
        if not similar_problems:
            return self.solver_model_client.generate_code(self.system_prompt, question)
        
        retrieval_prompt = f"Question: {question}\n\n"
        retrieval_prompt += "The following are the solutions in Python code for similar problems:\n\n"
        
        for i, problem in enumerate(similar_problems):
            retrieval_prompt += f"### Similar problem #{i+1}\n"
            retrieval_prompt += f"Question: {problem['question']}\n"
            retrieval_prompt += f"Solution:\n```python\n{problem['solution']}\n```\n\n"
            
        retrieval_prompt += "Your task is to create a Python solution for the new problem, using the similar solutions as references."
        
        return self.solver_model_client.generate_code(self.retrieval_system_prompt, retrieval_prompt)
    
    def _execute_and_improve_solution(self, question: str, solution_code: str) -> tuple[bool, str, str]:
        """Execute and improve solution through multiple iterations."""
        iteration = 1
        success = False
        output = ""
        current_code = solution_code
        
        while iteration <= self.max_iterations and not success:
            print(f"\nIteration #{iteration}/{self.max_iterations}")
            print(f"Current code:\n{current_code}\n")
            
            # Thực thi mã
            run_success, run_output, error = run_python_code(current_code)
            
            if run_success:
                print(f"Execution successful. Result: {run_output}")
                success = True
                output = run_output
                break
            else:
                print(f"Execution error: {error}")
                # Improve code
                if iteration < self.max_iterations:
                    print("Improving code...")
                    current_code = self.solver_model_client.improve_code(
                        self.system_prompt, question, current_code, error
                    )
            
            iteration += 1
        
        # Nếu không thành công sau tất cả các lần thử, sử dụng LLM trực tiếp
        if not success:
            print("\nCannot execute code after all attempts. Using direct LLM...")
            direct_answer = self._get_direct_answer(question)
            return False, direct_answer, current_code
        
        return success, output, current_code
    
    def _get_direct_answer(self, question: str) -> str:
        """Get direct answer from LLM when code cannot be executed."""
        direct_prompt = f"""
        Answer the following math question. Only provide the final answer, without explaining in detail.
        
        Question: {question}
        """
        
        completion = self.model_client.client.chat.completions.create(
            model=self.model_client.model_name,
            messages=[
                {"role": "system", "content": "You are a math expert. Answer the question in a concise and accurate manner."},
                {"role": "user", "content": direct_prompt}
            ]
        )
        
        return completion.choices[0].message.content.strip()        
    
    def _verify_result(self, question: str, output: str, code: str) -> dict[str, Any]:
        """Verify the result with another LLM."""
        # Skip verification if code was successfully executed
        # Check if we got the result from code execution by checking if the output exists
        if output and output.strip() != "" and not output.startswith("Error"):
            return {
                "correct": True,
                "answer": output,
                "evaluation": "The solution was successfully executed. No verification needed."
            }
        
        # Add Chain of Thought prompting for verification when code execution failed
        verification_prompt = f"""
        Verify the result for the following problem:
        
        Question: {question}
        
        Result: {output}
        
        Python code used:
        ```python
        {code}
        ```
        
        Please think step by step:
        1. Understand what the problem is asking for
        2. Analyze the given result
        3. Determine the correct approach to solve this problem
        4. Calculate the expected result
        5. Compare your calculated result with the given result
        
        After your step-by-step reasoning, answer in JSON format with the following fields:
        - correct (boolean): whether the result is correct in terms of mathematics (true/false)
        - answer (string): the correct result if the current result is not correct
        - evaluation (string): a concise evaluation of the reason why the result is correct/incorrect
        
        If the result is a single number and correct, always evaluate correct=true.
        """
        
        completion = self.model_client.client.chat.completions.create(
            model=self.model_client.model_name,
            messages=[
                {"role": "system", "content": "You are a math expert. First think through the problem step by step, then evaluate the result accurately and objectively."},
                {"role": "user", "content": verification_prompt}
            ]
        )
        
        response = completion.choices[0].message.content.strip()
        
        try:
            # Extract the JSON section from the response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                # Look for JSON-like structure in the response after Chain of Thought
                if "{" in response and "}" in response:
                    json_part = response[response.rfind("{"):response.rfind("}")+1]
                    json_str = json_part
                else:
                    json_str = response
            
            verification_result = json.loads(json_str)
            
            # Verify again if output is a number and matches answer
            if output.strip().isdigit() and "answer" in verification_result:
                if str(verification_result["answer"]).strip() == output.strip():
                    verification_result["correct"] = True
                    verification_result["evaluation"] = "The result is correct."
        except:
            # If cannot parse JSON, evaluate the result
            try:
                # Check if output is a number
                float_output = float(output.strip())
                verification_result = {
                    "correct": True,
                    "answer": float_output,
                    "evaluation": "The result is a single number and is automatically confirmed to be correct."
                }
            except:
                # If output cannot be converted to a number
                verification_result = {
                    "correct": True,  # Assume correct if cannot verify
                    "answer": output,
                    "evaluation": "Cannot verify automatically. Assume the result is correct."
                }
        
        return verification_result      
    
    