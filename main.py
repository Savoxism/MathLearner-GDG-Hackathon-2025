from openai import OpenAI
import os
import json
import subprocess
import sys
import re

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def extract_python_code(response):
    code_pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        # Lấy phần mã đầu tiên tìm được và loại bỏ khoảng trắng thừa
        return matches[0].strip()
    return response.strip()  # Nếu không tìm thấy định dạng markdown, trả về nguyên văn

with open("data/sample_metamath.json", "r") as f:
    data = json.load(f)
    
first_problem = data[0]
first_problem_question = first_problem['question']


SYSTEM_PROMPT = """
You are a math expert. Firstly, given a math problem, you will divide the problem into smaller sub-problems. Then, for each sub-probblem, you will build corresponding Python code to solve the sub-problem. Finally, you will combine all the sub-problem solutions into a Python file with multiple functions. Only return the Python code in a markdown code block with ```python tag. Do not include any other text.
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

def execute_solution():
    try:
        # Thêm print để debug
        print("Executing code:\n", python_code)
        
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
            output = result.stdout
            print(f"Output: {output}")
            
            actual_solution = first_problem.get('answer', '')
            if str(output.strip()) == str(actual_solution).strip():
                print("Solution is correct")
                return True, output
            else:
                print(f"Solution is incorrect. Expected: {actual_solution}")
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

        