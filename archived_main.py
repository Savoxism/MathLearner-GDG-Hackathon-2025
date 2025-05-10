from openai import OpenAI
import os
import json
import subprocess
import sys
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def extract_python_code(response):
    code_pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        # Get the first code block and strip whitespace
        return matches[0].strip()
    return response.strip()  # If no markdown format found, return as is

with open("data/sample_metamath.json", "r") as f:
    data = json.load(f)
    
first_problem = data[1]

first_problem_question = first_problem['question']

SYSTEM_PROMPT = """
You are a math expert. Firstly, given a math problem, you will divide the problem into smaller sub-problems. Then, for each sub-problem, you will build corresponding Python code to solve the sub-problem. Finally, you will combine all the sub-problem solutions into a Python file with multiple functions. 

IMPORTANT: Make sure your code prints the final answer as the last output.

Only return the Python code in a markdown code block with ```python tag. Do not include any other text.
"""

MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": first_problem_question}
]

MODEL_NAME = "qwen/qwen-2.5-coder-32b-instruct:free"


completion = client.chat.completions.create(
  extra_body={},
  model=MODEL_NAME,
  messages=MESSAGES,
)

python_code = completion.choices[0].message.content
python_code = extract_python_code(python_code)

print(python_code)

def execute_solution():
    try:
        with open("temp_solution.py", "w") as f:
            f.write(python_code)
            
        result = subprocess.run(
            [sys.executable, "temp_solution.py"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode != 0:
            error_message = result.stderr
            print(f"Error: {error_message}")
            return False, error_message
        else:
            output = result.stdout.strip()
            print(f"Output: {output}")
            
            actual_solution = str(first_problem.get('answer', ''))
            actual_solution = actual_solution.strip()
            
            print(f"Expected: {actual_solution}")
            
            # Direct comparison
            if output == actual_solution:
                print("Direct match - Solution is correct")
                return True, output
                
            # Numerical comparison (if both are numbers)
            try:
                if float(output) == float(actual_solution):
                    print("Numerical match - Solution is correct")
                    return True, output
            except:
                pass
                
            # AI verification as last resort
            verification_prompt = f"""
            Given a math problem solution, check if the Python output matches the expected answer.
            
            Expected answer: {actual_solution}
            Python output: {output}
            
            Evaluate if they are mathematically equivalent. Only respond with "True" if they are equivalent, or "False" if they are not.
            """

            completion = client.chat.completions.create(
                extra_body={},
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise math verification assistant. Only respond with True or False."},
                    {"role": "user", "content": verification_prompt}
                ]
            )

            verification_result = completion.choices[0].message.content.strip()
            if "true" in verification_result.lower():
                print("AI verified - Solution is correct")
                return True, output
            else:
                return False, f"Expected {actual_solution}, but got {output}"
            
    except Exception as e:
        return False, str(e)

success, message = execute_solution()

max_attempts = 3
attempt = 1

while not success and attempt < max_attempts:
    print(f"Attempting to solve the problem again (Attempt {attempt} of {max_attempts})")
    
    error_prompt = f"""
    Your Python code has the following issue:
    {message}
    
    Please provide a new version of the Python code to fix this issue.
    Make sure your code PRINTS the final answer as the last output.
    Original question: {first_problem_question}
    """
    
    completion = client.chat.completions.create(
        extra_body={},
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_problem_question},
            {"role": "assistant", "content": python_code},
            {"role": "user", "content": error_prompt}
        ],
    )
    
    python_code = extract_python_code(completion.choices[0].message.content)
    success, message = execute_solution()
    attempt += 1
    
with open("final_solution.py", "w") as f:
    f.write(python_code)

print(f"Final result: {'Success' if success else 'Failed'}")
print(f"Final message: {message}")

os.remove("temp_solution.py")
os.remove("final_solution.py")